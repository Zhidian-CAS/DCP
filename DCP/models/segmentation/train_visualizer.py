import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

class TrainVisualizer:
    """训练过程可视化器"""
    
    def __init__(
        self,
        output_dir: str,
        save_frequency: int = 1,
        show: bool = True
    ):
        """
        初始化训练可视化器
        Args:
            output_dir: 输出目录
            save_frequency: 保存频率（每多少个epoch保存一次）
            show: 是否显示图表
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency
        self.show = show
        self.logger = logging.getLogger(__name__)
        
        # 初始化数据存储
        self.train_metrics = {
            'epoch': [],
            'loss': [],
            'dice': [],
            'iou': [],
            'learning_rate': []
        }
        self.val_metrics = {
            'epoch': [],
            'loss': [],
            'dice': [],
            'iou': []
        }
        
    def update_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        learning_rate: Optional[float] = None
    ) -> None:
        """
        更新训练指标
        Args:
            epoch: 当前epoch
            train_metrics: 训练指标
            val_metrics: 验证指标
            learning_rate: 学习率
        """
        # 更新训练指标
        self.train_metrics['epoch'].append(epoch)
        self.train_metrics['loss'].append(train_metrics['loss'])
        self.train_metrics['dice'].append(train_metrics['dice'])
        self.train_metrics['iou'].append(train_metrics['iou'])
        if learning_rate is not None:
            self.train_metrics['learning_rate'].append(learning_rate)
            
        # 更新验证指标
        if val_metrics is not None:
            self.val_metrics['epoch'].append(epoch)
            self.val_metrics['loss'].append(val_metrics['loss'])
            self.val_metrics['dice'].append(val_metrics['dice'])
            self.val_metrics['iou'].append(val_metrics['iou'])
            
        # 定期保存可视化结果
        if epoch % self.save_frequency == 0:
            self.save_visualizations()
            
    def plot_learning_curves(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        绘制学习曲线
        Args:
            save_path: 保存路径
            show: 是否显示
        """
        plt.figure(figsize=(15, 10))
        
        # 绘制损失曲线
        plt.subplot(2, 1, 1)
        plt.plot(self.train_metrics['epoch'], self.train_metrics['loss'], label='Train Loss')
        if self.val_metrics['epoch']:
            plt.plot(self.val_metrics['epoch'], self.val_metrics['loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curves - Loss')
        plt.legend()
        plt.grid(True)
        
        # 绘制Dice系数曲线
        plt.subplot(2, 1, 2)
        plt.plot(self.train_metrics['epoch'], self.train_metrics['dice'], label='Train Dice')
        if self.val_metrics['epoch']:
            plt.plot(self.val_metrics['epoch'], self.val_metrics['dice'], label='Val Dice')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Coefficient')
        plt.title('Learning Curves - Dice')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            self.logger.info(f"学习曲线已保存至: {save_path}")
            
        if show:
            plt.show()
            
    def plot_metrics_distribution(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        绘制指标分布
        Args:
            save_path: 保存路径
            show: 是否显示
        """
        plt.figure(figsize=(15, 10))
        
        # 绘制损失分布
        plt.subplot(2, 2, 1)
        sns.histplot(self.train_metrics['loss'], bins=50, kde=True)
        plt.title('Train Loss Distribution')
        plt.xlabel('Loss')
        plt.ylabel('Count')
        
        # 绘制Dice系数分布
        plt.subplot(2, 2, 2)
        sns.histplot(self.train_metrics['dice'], bins=50, kde=True)
        plt.title('Train Dice Distribution')
        plt.xlabel('Dice')
        plt.ylabel('Count')
        
        # 绘制IoU分布
        plt.subplot(2, 2, 3)
        sns.histplot(self.train_metrics['iou'], bins=50, kde=True)
        plt.title('Train IoU Distribution')
        plt.xlabel('IoU')
        plt.ylabel('Count')
        
        # 绘制学习率变化
        if self.train_metrics['learning_rate']:
            plt.subplot(2, 2, 4)
            plt.plot(self.train_metrics['epoch'], self.train_metrics['learning_rate'])
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            self.logger.info(f"指标分布图已保存至: {save_path}")
            
        if show:
            plt.show()
            
    def plot_metrics_correlation(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        绘制指标相关性
        Args:
            save_path: 保存路径
            show: 是否显示
        """
        # 创建数据框
        df = pd.DataFrame({
            'Loss': self.train_metrics['loss'],
            'Dice': self.train_metrics['dice'],
            'IoU': self.train_metrics['iou']
        })
        
        # 计算相关性矩阵
        corr_matrix = df.corr()
        
        # 绘制相关性热力图
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Metrics Correlation')
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            self.logger.info(f"相关性热力图已保存至: {save_path}")
            
        if show:
            plt.show()
            
    def save_metrics_to_csv(self) -> None:
        """保存指标到CSV文件"""
        # 保存训练指标
        train_df = pd.DataFrame(self.train_metrics)
        train_df.to_csv(self.output_dir / 'train_metrics.csv', index=False)
        
        # 保存验证指标
        val_df = pd.DataFrame(self.val_metrics)
        val_df.to_csv(self.output_dir / 'val_metrics.csv', index=False)
        
        self.logger.info(f"指标已保存至: {self.output_dir}")
        
    def save_visualizations(self) -> None:
        """保存所有可视化结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存学习曲线
        self.plot_learning_curves(
            save_path=self.output_dir / f'learning_curves_{timestamp}.png',
            show=False
        )
        
        # 保存指标分布
        self.plot_metrics_distribution(
            save_path=self.output_dir / f'metrics_distribution_{timestamp}.png',
            show=False
        )
        
        # 保存指标相关性
        self.plot_metrics_correlation(
            save_path=self.output_dir / f'metrics_correlation_{timestamp}.png',
            show=False
        )
        
        # 保存指标到CSV
        self.save_metrics_to_csv()
        
    def create_training_report(self) -> None:
        """创建训练报告"""
        report_path = self.output_dir / 'training_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("训练报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 训练信息
            f.write("训练信息\n")
            f.write("-" * 20 + "\n")
            f.write(f"总训练轮数: {len(self.train_metrics['epoch'])}\n")
            f.write(f"最佳训练损失: {min(self.train_metrics['loss']):.4f}\n")
            f.write(f"最佳训练Dice: {max(self.train_metrics['dice']):.4f}\n")
            f.write(f"最佳训练IoU: {max(self.train_metrics['iou']):.4f}\n\n")
            
            # 验证信息
            if self.val_metrics['epoch']:
                f.write("验证信息\n")
                f.write("-" * 20 + "\n")
                f.write(f"最佳验证损失: {min(self.val_metrics['loss']):.4f}\n")
                f.write(f"最佳验证Dice: {max(self.val_metrics['dice']):.4f}\n")
                f.write(f"最佳验证IoU: {max(self.val_metrics['iou']):.4f}\n\n")
                
            # 学习率信息
            if self.train_metrics['learning_rate']:
                f.write("学习率信息\n")
                f.write("-" * 20 + "\n")
                f.write(f"初始学习率: {self.train_metrics['learning_rate'][0]:.6f}\n")
                f.write(f"最终学习率: {self.train_metrics['learning_rate'][-1]:.6f}\n")
                
        self.logger.info(f"训练报告已保存至: {report_path}") 