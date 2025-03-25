import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import colorsys

class DetectionVisualizer:
    """目标检测可视化工具"""
    def __init__(self, 
                 num_classes: int,
                 class_names: Optional[List[str]] = None,
                 save_dir: Optional[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.save_dir = Path(save_dir) if save_dir else None
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 生成颜色映射
        self.colors = self._generate_colors(num_classes)
        
    def _generate_colors(self, num_classes: int) -> List[Tuple[float, float, float]]:
        """
        生成类别颜色
        Args:
            num_classes: 类别数量
        Returns:
            colors: 颜色列表
        """
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            saturation = 0.7
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        return colors
        
    def visualize_detection(self, 
                          image: np.ndarray,
                          boxes: np.ndarray,
                          labels: np.ndarray,
                          scores: Optional[np.ndarray] = None,
                          gt_boxes: Optional[np.ndarray] = None,
                          gt_labels: Optional[np.ndarray] = None,
                          save_name: Optional[str] = None) -> np.ndarray:
        """
        可视化检测结果
        Args:
            image: 输入图像
            boxes: 预测框
            labels: 预测标签
            scores: 预测得分
            gt_boxes: 真实框
            gt_labels: 真实标签
            save_name: 保存文件名
        Returns:
            vis_image: 可视化结果
        """
        # 创建图像副本
        vis_image = image.copy()
        
        # 绘制预测框
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box.astype(int)
            color = self.colors[label]
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), 
                         (int(color[0]*255), int(color[1]*255), int(color[2]*255)), 2)
            
            # 绘制标签
            label_text = self.class_names[label]
            if scores is not None:
                label_text += f' {scores[i]:.2f}'
                
            # 计算文本背景
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (x1, y1-text_height-baseline),
                         (x1+text_width, y1), 
                         (int(color[0]*255), int(color[1]*255), int(color[2]*255)), -1)
            
            # 绘制文本
            cv2.putText(vis_image, label_text, (x1, y1-baseline),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        # 绘制真实框
        if gt_boxes is not None and gt_labels is not None:
            for box, label in zip(gt_boxes, gt_labels):
                x1, y1, x2, y2 = box.astype(int)
                color = self.colors[label]
                
                # 绘制虚线边界框
                cv2.rectangle(vis_image, (x1, y1), (x2, y2),
                            (int(color[0]*255), int(color[1]*255), int(color[2]*255)), 1)
                # 绘制虚线
                for i in range(x1, x2, 10):
                    cv2.line(vis_image, (i, y1), (i+5, y1),
                            (int(color[0]*255), int(color[1]*255), int(color[2]*255)), 1)
                    cv2.line(vis_image, (i, y2), (i+5, y2),
                            (int(color[0]*255), int(color[1]*255), int(color[2]*255)), 1)
                for i in range(y1, y2, 10):
                    cv2.line(vis_image, (x1, i), (x1, i+5),
                            (int(color[0]*255), int(color[1]*255), int(color[2]*255)), 1)
                    cv2.line(vis_image, (x2, i), (x2, i+5),
                            (int(color[0]*255), int(color[1]*255), int(color[2]*255)), 1)
                    
        # 保存结果
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            cv2.imwrite(str(save_path), vis_image)
            self.logger.info(f'Saved visualization to {save_path}')
            
        return vis_image
        
    def visualize_batch(self, 
                       images: torch.Tensor,
                       boxes: List[torch.Tensor],
                       labels: List[torch.Tensor],
                       scores: Optional[List[torch.Tensor]] = None,
                       gt_boxes: Optional[List[torch.Tensor]] = None,
                       gt_labels: Optional[List[torch.Tensor]] = None,
                       save_name: Optional[str] = None) -> np.ndarray:
        """
        可视化批次结果
        Args:
            images: 输入图像批次
            boxes: 预测框批次
            labels: 预测标签批次
            scores: 预测得分批次
            gt_boxes: 真实框批次
            gt_labels: 真实标签批次
            save_name: 保存文件名
        Returns:
            vis_image: 可视化结果
        """
        # 将图像转换为numpy数组
        images = images.cpu().numpy()
        
        # 创建网格
        batch_size = len(images)
        grid_size = int(np.ceil(np.sqrt(batch_size)))
        grid_image = np.zeros((grid_size*images[0].shape[1],
                             grid_size*images[0].shape[2], 3), dtype=np.uint8)
        
        # 填充网格
        for i in range(batch_size):
            row = i // grid_size
            col = i % grid_size
            
            # 获取当前图像
            curr_image = images[i].transpose(1, 2, 0)
            curr_image = (curr_image * 255).astype(np.uint8)
            
            # 可视化检测结果
            curr_vis = self.visualize_detection(
                curr_image,
                boxes[i].cpu().numpy(),
                labels[i].cpu().numpy(),
                scores[i].cpu().numpy() if scores else None,
                gt_boxes[i].cpu().numpy() if gt_boxes else None,
                gt_labels[i].cpu().numpy() if gt_labels else None
            )
            
            # 填充到网格
            grid_image[row*curr_image.shape[0]:(row+1)*curr_image.shape[0],
                      col*curr_image.shape[1]:(col+1)*curr_image.shape[1]] = curr_vis
            
        # 保存结果
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            cv2.imwrite(str(save_path), grid_image)
            self.logger.info(f'Saved batch visualization to {save_path}')
            
        return grid_image
        
    def plot_metrics(self, 
                    metrics: Dict[str, List[float]],
                    save_name: Optional[str] = None):
        """
        绘制评估指标
        Args:
            metrics: 指标字典
            save_name: 保存文件名
        """
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 绘制每个指标
        for metric_name, values in metrics.items():
            plt.plot(values, label=metric_name)
            
        # 设置图形属性
        plt.title('Detection Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # 保存结果
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(str(save_path))
            plt.close()
            self.logger.info(f'Saved metrics plot to {save_path}')
        else:
            plt.show()
            
    def plot_confusion_matrix(self, 
                            confusion_matrix: np.ndarray,
                            save_name: Optional[str] = None):
        """
        绘制混淆矩阵
        Args:
            confusion_matrix: 混淆矩阵
            save_name: 保存文件名
        """
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 绘制混淆矩阵
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        # 设置图形属性
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # 保存结果
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(str(save_path))
            plt.close()
            self.logger.info(f'Saved confusion matrix to {save_path}')
        else:
            plt.show()
            
    def plot_pr_curve(self, 
                     recalls: List[float],
                     precisions: List[float],
                     save_name: Optional[str] = None):
        """
        绘制PR曲线
        Args:
            recalls: 召回率列表
            precisions: 精确率列表
            save_name: 保存文件名
        """
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 绘制PR曲线
        plt.plot(recalls, precisions)
        
        # 设置图形属性
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)
        
        # 保存结果
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(str(save_path))
            plt.close()
            self.logger.info(f'Saved PR curve to {save_path}')
        else:
            plt.show()
            
    def plot_roc_curve(self, 
                      fprs: List[float],
                      tprs: List[float],
                      save_name: Optional[str] = None):
        """
        绘制ROC曲线
        Args:
            fprs: 假阳性率列表
            tprs: 真阳性率列表
            save_name: 保存文件名
        """
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 绘制ROC曲线
        plt.plot(fprs, tprs)
        
        # 设置图形属性
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        
        # 保存结果
        if save_name and self.save_dir:
            save_path = self.save_dir / save_name
            plt.savefig(str(save_path))
            plt.close()
            self.logger.info(f'Saved ROC curve to {save_path}')
        else:
            plt.show() 