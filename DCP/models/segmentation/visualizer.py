import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import networkx as nx
from torchviz import make_dot
import cv2
from PIL import Image
import io

class SegmentationVisualizer:
    """分割模型可视化器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化可视化器
        Args:
            model: 待可视化的模型
            device: 运行设备
        """
        self.model = model.to(device)
        self.device = device
        self.logger = logging.getLogger(__name__)
        
    def visualize_model_structure(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        可视化模型结构
        Args:
            save_path: 保存路径
            show: 是否显示
        """
        # 创建计算图
        dummy_input = torch.randn(1, 3, 512, 512, device=self.device)
        output = self.model(dummy_input)
        dot = make_dot(output, params=dict(self.model.named_parameters()))
        
        # 保存或显示
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            dot.render(str(save_path), format='png', cleanup=True)
            self.logger.info(f"模型结构图已保存至: {save_path}")
            
        if show:
            plt.figure(figsize=(20, 20))
            plt.imshow(Image.open(io.BytesIO(dot.pipe())))
            plt.axis('off')
            plt.show()
            
    def visualize_weight_distribution(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        可视化权重分布
        Args:
            save_path: 保存路径
            show: 是否显示
        """
        weights = []
        names = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weights.append(module.weight.data.cpu().numpy().flatten())
                names.append(name)
                
        # 创建子图
        n_layers = len(weights)
        n_cols = 3
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, (weight, name) in enumerate(zip(weights, names)):
            plt.subplot(n_rows, n_cols, i + 1)
            sns.histplot(weight, bins=50, kde=True)
            plt.title(f'{name}\nMean: {np.mean(weight):.4f}\nStd: {np.std(weight):.4f}')
            plt.xlabel('Weight Value')
            plt.ylabel('Count')
            
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            self.logger.info(f"权重分布图已保存至: {save_path}")
            
        if show:
            plt.show()
            
    def visualize_pruning_mask(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        可视化剪枝掩码
        Args:
            save_path: 保存路径
            show: 是否显示
        """
        masks = []
        names = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight_mask'):
                    mask = module.weight_mask.data.cpu().numpy()
                    masks.append(mask)
                    names.append(name)
                    
        if not masks:
            self.logger.warning("未找到剪枝掩码")
            return
            
        # 创建子图
        n_layers = len(masks)
        n_cols = 3
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, (mask, name) in enumerate(zip(masks, names)):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(mask, cmap='binary')
            plt.title(f'{name}\nSparsity: {1 - np.mean(mask):.4f}')
            plt.colorbar()
            
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            self.logger.info(f"剪枝掩码图已保存至: {save_path}")
            
        if show:
            plt.show()
            
    def visualize_prediction(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        可视化预测结果
        Args:
            image: 输入图像
            mask: 真实掩码
            save_path: 保存路径
            show: 是否显示
        """
        # 预处理图像
        image = cv2.resize(image, (512, 512))
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # 获取预测结果
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
        # 创建可视化图像
        plt.figure(figsize=(15, 5))
        
        # 显示原始图像
        plt.subplot(131)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')
        
        # 显示预测掩码
        plt.subplot(132)
        plt.imshow(pred_mask, cmap='tab20')
        plt.title('Predicted Mask')
        plt.axis('off')
        
        # 显示真实掩码（如果有）
        if mask is not None:
            plt.subplot(133)
            plt.imshow(mask, cmap='tab20')
            plt.title('Ground Truth')
            plt.axis('off')
            
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            self.logger.info(f"预测结果图已保存至: {save_path}")
            
        if show:
            plt.show()
            
    def visualize_sensitivity_analysis(
        self,
        sensitivity_results: Dict[str, Dict[float, float]],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        可视化敏感度分析结果
        Args:
            sensitivity_results: 敏感度分析结果
            save_path: 保存路径
            show: 是否显示
        """
        plt.figure(figsize=(10, 6))
        
        for method, results in sensitivity_results.items():
            amounts = list(results.keys())
            losses = list(results.values())
            plt.plot(amounts, losses, marker='o', label=method)
            
        plt.xlabel('Pruning Amount')
        plt.ylabel('Loss')
        plt.title('Sensitivity Analysis')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            self.logger.info(f"敏感度分析图已保存至: {save_path}")
            
        if show:
            plt.show()
            
    def visualize_performance_metrics(
        self,
        metrics: Dict[str, float],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        可视化性能指标
        Args:
            metrics: 性能指标
            save_path: 保存路径
            show: 是否显示
        """
        # 创建条形图
        plt.figure(figsize=(12, 6))
        
        names = list(metrics.keys())
        values = list(metrics.values())
        
        plt.bar(names, values)
        plt.xticks(rotation=45)
        plt.ylabel('Value')
        plt.title('Performance Metrics')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            self.logger.info(f"性能指标图已保存至: {save_path}")
            
        if show:
            plt.show()
            
    def create_visualization_report(
        self,
        output_dir: str,
        image: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        sensitivity_results: Optional[Dict] = None,
        performance_metrics: Optional[Dict] = None
    ) -> None:
        """
        创建完整的可视化报告
        Args:
            output_dir: 输出目录
            image: 输入图像
            mask: 真实掩码
            sensitivity_results: 敏感度分析结果
            performance_metrics: 性能指标
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 可视化模型结构
        self.visualize_model_structure(
            save_path=output_dir / 'model_structure.png',
            show=False
        )
        
        # 可视化权重分布
        self.visualize_weight_distribution(
            save_path=output_dir / 'weight_distribution.png',
            show=False
        )
        
        # 可视化剪枝掩码
        self.visualize_pruning_mask(
            save_path=output_dir / 'pruning_mask.png',
            show=False
        )
        
        # 可视化预测结果
        if image is not None:
            self.visualize_prediction(
                image=image,
                mask=mask,
                save_path=output_dir / 'prediction.png',
                show=False
            )
            
        # 可视化敏感度分析结果
        if sensitivity_results is not None:
            self.visualize_sensitivity_analysis(
                sensitivity_results=sensitivity_results,
                save_path=output_dir / 'sensitivity_analysis.png',
                show=False
            )
            
        # 可视化性能指标
        if performance_metrics is not None:
            self.visualize_performance_metrics(
                metrics=performance_metrics,
                save_path=output_dir / 'performance_metrics.png',
                show=False
            )
            
        self.logger.info(f"可视化报告已保存至: {output_dir}") 