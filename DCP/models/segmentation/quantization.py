import torch
import torch.nn as nn
import torch.quantization
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from tqdm import tqdm
import copy

from models.segmentation.segnet import SegNet
from models.segmentation.augmentation import SegmentationAugmentation

class SegmentationQuantizer:
    """分割模型量化器"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        input_size: Tuple[int, int] = (512, 512),
        batch_size: int = 1
    ):
        """
        初始化量化器
        Args:
            model_path: 模型路径
            device: 设备
            input_size: 输入图像大小
            batch_size: 批次大小
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.batch_size = batch_size
        
        # 创建数据增强器
        self.transform = SegmentationAugmentation(train_mode=False)
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self, model_path: str) -> SegNet:
        """
        加载模型
        Args:
            model_path: 模型路径
        Returns:
            model: 加载的模型
        """
        # 创建模型
        model = SegNet().to(self.device)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def prepare_static_quantization(
        self,
        calibration_data: List[np.ndarray],
        num_bits: int = 8
    ) -> nn.Module:
        """
        准备静态量化
        Args:
            calibration_data: 校准数据
            num_bits: 量化位数
        Returns:
            quantized_model: 量化后的模型
        """
        # 创建量化配置
        qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 准备模型
        model = copy.deepcopy(self.model)
        model.eval()
        
        # 融合模块
        model = torch.quantization.fuse_modules(
            model,
            [['conv', 'bn', 'relu']],
            inplace=True
        )
        
        # 准备量化
        model.qconfig = qconfig
        torch.quantization.prepare(model, inplace=True)
        
        # 校准
        with torch.no_grad():
            for image in tqdm(calibration_data, desc='Calibrating'):
                # 预处理
                image = self._preprocess_image(image)
                
                # 前向传播
                _ = model(image)
        
        # 转换模型
        torch.quantization.convert(model, inplace=True)
        
        return model
    
    def prepare_dynamic_quantization(
        self,
        num_bits: int = 8
    ) -> nn.Module:
        """
        准备动态量化
        Args:
            num_bits: 量化位数
        Returns:
            quantized_model: 量化后的模型
        """
        # 创建量化配置
        qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 准备模型
        model = copy.deepcopy(self.model)
        model.eval()
        
        # 准备量化
        model.qconfig = qconfig
        torch.quantization.prepare_dynamic(model, inplace=True)
        
        return model
    
    def _preprocess_image(
        self,
        image: np.ndarray,
        return_tensor: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        预处理图像
        Args:
            image: 输入图像
            return_tensor: 是否返回张量
        Returns:
            处理后的图像
        """
        # 调整大小
        image = cv2.resize(image, self.input_size)
        
        # 应用变换
        transformed = self.transform(image=image)
        processed_image = transformed['image']
        
        if return_tensor:
            return processed_image.unsqueeze(0).to(self.device)
        return processed_image.numpy()
    
    def evaluate_quantized_model(
        self,
        quantized_model: nn.Module,
        test_data: List[np.ndarray],
        test_masks: Optional[List[np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        评估量化模型
        Args:
            quantized_model: 量化后的模型
            test_data: 测试数据
            test_masks: 测试掩码
        Returns:
            评估指标
        """
        # 设置模型为评估模式
        quantized_model.eval()
        
        # 初始化指标
        total_loss = 0
        total_dice = 0
        total_iou = 0
        num_samples = len(test_data)
        
        # 评估
        with torch.no_grad():
            for i, image in enumerate(tqdm(test_data, desc='Evaluating')):
                # 预处理
                image_tensor = self._preprocess_image(image)
                
                # 前向传播
                outputs = quantized_model(image_tensor)
                
                # 计算损失
                if test_masks is not None:
                    mask = test_masks[i]
                    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
                    loss = self.model.get_loss(outputs, mask_tensor)
                    total_loss += loss.item()
                
                # 计算指标
                pred_mask = torch.argmax(outputs, dim=1)[0].cpu().numpy()
                if test_masks is not None:
                    dice = self._calculate_dice(pred_mask, mask)
                    iou = self._calculate_iou(pred_mask, mask)
                    total_dice += dice
                    total_iou += iou
        
        # 计算平均指标
        metrics = {
            'loss': total_loss / num_samples if test_masks is not None else None,
            'dice': total_dice / num_samples if test_masks is not None else None,
            'iou': total_iou / num_samples if test_masks is not None else None
        }
        
        return metrics
    
    def _calculate_dice(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        计算Dice系数
        Args:
            pred: 预测掩码
            target: 目标掩码
        Returns:
            dice: Dice系数
        """
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target)
        return 2 * intersection / (union + 1e-6)
    
    def _calculate_iou(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        计算IoU
        Args:
            pred: 预测掩码
            target: 目标掩码
        Returns:
            iou: IoU值
        """
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target) - intersection
        return intersection / (union + 1e-6)
    
    def export_quantized_model(
        self,
        quantized_model: nn.Module,
        save_path: str
    ) -> None:
        """
        导出量化模型
        Args:
            quantized_model: 量化后的模型
            save_path: 保存路径
        """
        # 创建示例输入
        dummy_input = torch.randn(
            self.batch_size,
            3,
            self.input_size[0],
            self.input_size[1]
        ).to(self.device)
        
        # 导出模型
        torch.onnx.export(
            quantized_model,
            dummy_input,
            save_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=11
        )
        
        self.logger.info(f'量化模型已导出到: {save_path}')
    
    def get_model_size(self, model: nn.Module) -> float:
        """
        获取模型大小
        Args:
            model: 模型
        Returns:
            size: 模型大小（MB）
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb 