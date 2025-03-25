import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class SegmentationMetrics:
    """分割评估指标"""
    def __init__(self, num_classes: int):
        """
        初始化
        Args:
            num_classes: 类别数量
        """
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        """重置指标"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.total_pixels = 0
        self.correct_pixels = 0
        self.total_dice = 0
        self.total_iou = 0
        self.total_samples = 0
        
    def update(self, 
               pred: Union[torch.Tensor, np.ndarray],
               target: Union[torch.Tensor, np.ndarray]):
        """
        更新指标
        Args:
            pred: 预测结果 [B, H, W] 或 [H, W]
            target: 真实标签 [B, H, W] 或 [H, W]
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
            
        if pred.ndim == 2:
            pred = pred[np.newaxis, ...]
            target = target[np.newaxis, ...]
            
        batch_size = pred.shape[0]
        
        for i in range(batch_size):
            # 更新混淆矩阵
            curr_confusion_matrix = confusion_matrix(
                target[i].flatten(),
                pred[i].flatten(),
                labels=range(self.num_classes)
            )
            self.confusion_matrix += curr_confusion_matrix
            
            # 更新像素级指标
            self.total_pixels += target[i].size
            self.correct_pixels += np.sum(pred[i] == target[i])
            
            # 更新Dice系数
            dice = self._compute_dice(pred[i], target[i])
            self.total_dice += dice
            
            # 更新IoU
            iou = self._compute_iou(pred[i], target[i])
            self.total_iou += iou
            
        self.total_samples += batch_size
        
    def get_metrics(self) -> Dict[str, float]:
        """
        获取评估指标
        Returns:
            metrics: 评估指标字典
        """
        metrics = {}
        
        # 像素准确率
        metrics['pixel_acc'] = self.correct_pixels / self.total_pixels
        
        # 类别准确率
        per_class_acc = np.diag(self.confusion_matrix) / \
                       (np.sum(self.confusion_matrix, axis=1) + 1e-5)
        metrics['mean_acc'] = np.mean(per_class_acc)
        
        # 类别IoU
        per_class_iou = np.diag(self.confusion_matrix) / \
                       (np.sum(self.confusion_matrix, axis=1) + 
                        np.sum(self.confusion_matrix, axis=0) - 
                        np.diag(self.confusion_matrix) + 1e-5)
        metrics['mean_iou'] = np.mean(per_class_iou)
        
        # 频率加权IoU
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        metrics['freq_weighted_iou'] = np.sum(freq * per_class_iou)
        
        # 平均Dice系数
        metrics['mean_dice'] = self.total_dice / self.total_samples
        
        # 平均IoU
        metrics['mean_batch_iou'] = self.total_iou / self.total_samples
        
        return metrics
        
    def _compute_dice(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        计算Dice系数
        Args:
            pred: 预测掩码 [H, W]
            target: 真实掩码 [H, W]
        Returns:
            dice: Dice系数
        """
        smooth = 1e-5
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice
        
    def _compute_iou(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        计算IoU
        Args:
            pred: 预测掩码 [H, W]
            target: 真实掩码 [H, W]
        Returns:
            iou: IoU
        """
        smooth = 1e-5
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou
        
    def plot_confusion_matrix(self, 
                            class_names: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (10, 8),
                            save_path: Optional[str] = None):
        """
        绘制混淆矩阵
        Args:
            class_names: 类别名称列表
            figsize: 图像大小
            save_path: 保存路径
        """
        if class_names is None:
            class_names = [str(i) for i in range(self.num_classes)]
            
        # 计算归一化的混淆矩阵
        confusion_matrix_norm = self.confusion_matrix / \
                              (np.sum(self.confusion_matrix, axis=1, keepdims=True) + 1e-5)
        
        # 创建热力图
        plt.figure(figsize=figsize)
        sns.heatmap(
            confusion_matrix_norm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
    def plot_metrics(self,
                    metrics_history: Dict[str, List[float]],
                    figsize: Tuple[int, int] = (12, 6),
                    save_path: Optional[str] = None):
        """
        绘制指标曲线
        Args:
            metrics_history: 指标历史记录
            figsize: 图像大小
            save_path: 保存路径
        """
        plt.figure(figsize=figsize)
        
        for metric_name, values in metrics_history.items():
            plt.plot(values, label=metric_name)
            
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Metrics')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close() 