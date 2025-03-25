import torch
import torch.nn as nn
import torch.onnx
import onnx
import onnxruntime
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from tqdm import tqdm

from models.segmentation.segnet import SegNet
from models.segmentation.augmentation import SegmentationAugmentation

class SegmentationDeployer:
    """分割模型部署器"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        conf_threshold: float = 0.5,
        input_size: Tuple[int, int] = (512, 512),
        batch_size: int = 1
    ):
        """
        初始化部署器
        Args:
            model_path: 模型路径
            device: 设备
            conf_threshold: 置信度阈值
            input_size: 输入图像大小
            batch_size: 批次大小
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
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
    
    def preprocess_image(
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
    
    def postprocess_mask(
        self,
        mask: torch.Tensor,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        后处理掩码
        Args:
            mask: 预测掩码
            original_size: 原始图像大小
        Returns:
            处理后的掩码
        """
        # 调整大小
        mask = mask.cpu().numpy()
        mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def predict(
        self,
        image: np.ndarray,
        return_prob: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        预测单张图像
        Args:
            image: 输入图像
            return_prob: 是否返回概率图
        Returns:
            预测掩码和概率图（可选）
        """
        # 保存原始大小
        original_size = (image.shape[1], image.shape[0])
        
        # 预处理
        image_tensor = self.preprocess_image(image)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            pred_prob = torch.softmax(outputs, dim=1)
            pred_mask = torch.argmax(outputs, dim=1)
            
            # 应用置信度阈值
            max_prob = torch.max(pred_prob, dim=1)[0]
            pred_mask[max_prob < self.conf_threshold] = 0
            
            # 后处理
            pred_mask = self.postprocess_mask(pred_mask[0], original_size)
            pred_prob = self.postprocess_mask(pred_prob[0], original_size)
            
            if return_prob:
                return pred_mask, pred_prob
            return pred_mask
    
    def predict_batch(
        self,
        images: List[np.ndarray],
        return_prob: bool = False
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        批量预测
        Args:
            images: 输入图像列表
            return_prob: 是否返回概率图
        Returns:
            预测掩码列表和概率图列表（可选）
        """
        # 保存原始大小
        original_sizes = [(img.shape[1], img.shape[0]) for img in images]
        
        # 预处理
        image_tensors = []
        for image in images:
            image_tensor = self.preprocess_image(image)
            image_tensors.append(image_tensor)
        
        # 批处理
        image_batch = torch.cat(image_tensors, dim=0)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_batch)
            pred_prob = torch.softmax(outputs, dim=1)
            pred_masks = torch.argmax(outputs, dim=1)
            
            # 应用置信度阈值
            max_prob = torch.max(pred_prob, dim=1)[0]
            pred_masks[max_prob < self.conf_threshold] = 0
            
            # 后处理
            pred_masks = [
                self.postprocess_mask(mask, size)
                for mask, size in zip(pred_masks, original_sizes)
            ]
            pred_probs = [
                self.postprocess_mask(prob, size)
                for prob, size in zip(pred_prob, original_sizes)
            ]
            
            if return_prob:
                return pred_masks, pred_probs
            return pred_masks
    
    def export_onnx(
        self,
        save_path: str,
        dynamic_axes: bool = True
    ) -> None:
        """
        导出ONNX模型
        Args:
            save_path: 保存路径
            dynamic_axes: 是否使用动态轴
        """
        # 创建示例输入
        dummy_input = torch.randn(
            self.batch_size,
            3,
            self.input_size[0],
            self.input_size[1]
        ).to(self.device)
        
        # 设置动态轴
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        } if dynamic_axes else None
        
        # 导出模型
        torch.onnx.export(
            self.model,
            dummy_input,
            save_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=11
        )
        
        # 验证模型
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        
        self.logger.info(f'模型已导出到: {save_path}')
    
    def optimize_onnx(
        self,
        model_path: str,
        save_path: str
    ) -> None:
        """
        优化ONNX模型
        Args:
            model_path: 模型路径
            save_path: 保存路径
        """
        # 加载模型
        model = onnx.load(model_path)
        
        # 优化模型
        optimized_model = onnx.optimizer.optimize(model)
        
        # 保存优化后的模型
        onnx.save(optimized_model, save_path)
        
        self.logger.info(f'模型已优化并保存到: {save_path}')
    
    def create_onnx_session(
        self,
        model_path: str,
        providers: Optional[List[str]] = None
    ) -> onnxruntime.InferenceSession:
        """
        创建ONNX推理会话
        Args:
            model_path: 模型路径
            providers: 推理提供程序列表
        Returns:
            session: 推理会话
        """
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        session = onnxruntime.InferenceSession(
            model_path,
            providers=providers
        )
        
        return session
    
    def predict_onnx(
        self,
        session: onnxruntime.InferenceSession,
        image: np.ndarray,
        return_prob: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        使用ONNX模型预测
        Args:
            session: ONNX会话
            image: 输入图像
            return_prob: 是否返回概率图
        Returns:
            预测掩码和概率图（可选）
        """
        # 保存原始大小
        original_size = (image.shape[1], image.shape[0])
        
        # 预处理
        processed_image = self.preprocess_image(image, return_tensor=False)
        
        # 准备输入
        input_name = session.get_inputs()[0].name
        input_data = processed_image.astype(np.float32)
        
        # 推理
        outputs = session.run(None, {input_name: input_data})
        pred_prob = outputs[0]
        
        # 后处理
        pred_mask = np.argmax(pred_prob, axis=0)
        pred_prob = np.max(pred_prob, axis=0)
        
        # 应用置信度阈值
        pred_mask[pred_prob < self.conf_threshold] = 0
        
        # 调整大小
        pred_mask = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)
        pred_prob = cv2.resize(pred_prob, original_size, interpolation=cv2.INTER_LINEAR)
        
        if return_prob:
            return pred_mask, pred_prob
        return pred_mask
    
    def benchmark(
        self,
        images: List[np.ndarray],
        num_runs: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """
        模型性能基准测试
        Args:
            images: 测试图像列表
            num_runs: 运行次数
            warmup: 预热次数
        Returns:
            性能指标
        """
        # 预热
        for _ in range(warmup):
            self.predict_batch(images[:self.batch_size])
        
        # 性能测试
        times = []
        for _ in tqdm(range(num_runs), desc='Benchmarking'):
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            self.predict_batch(images[:self.batch_size])
            end_time.record()
            
            torch.cuda.synchronize()
            times.append(start_time.elapsed_time(end_time))
        
        # 计算统计信息
        times = np.array(times)
        stats = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'fps': 1000 / np.mean(times) * self.batch_size
        }
        
        return stats 