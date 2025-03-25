import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class PruningTrainer:
    """剪枝后的模型重训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        metrics: Optional[Dict[str, Callable]] = None
    ):
        """
        初始化训练器
        Args:
            model: 待训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 运行设备
            metrics: 评估指标字典
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.metrics = metrics or {}
        self.logger = logging.getLogger(__name__)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': {},
            'val_metrics': {},
            'lr': []
        }
        
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch
        Returns:
            包含训练指标的字典
        """
        self.model.train()
        total_loss = 0
        num_samples = 0
        metrics_sum = {name: 0.0 for name in self.metrics}
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 更新指标
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                num_samples += batch_size
                
                # 计算其他指标
                with torch.no_grad():
                    for name, metric_fn in self.metrics.items():
                        metrics_sum[name] += metric_fn(outputs, masks) * batch_size
                        
                # 更新进度条
                pbar.set_postfix({
                    'loss': total_loss / num_samples,
                    **{name: metrics_sum[name] / num_samples for name in self.metrics}
                })
                
        # 计算平均指标
        avg_loss = total_loss / num_samples
        avg_metrics = {name: metrics_sum[name] / num_samples for name in self.metrics}
        
        return {'loss': avg_loss, **avg_metrics}
        
    def validate(self) -> Dict[str, float]:
        """
        验证模型
        Returns:
            包含验证指标的字典
        """
        self.model.eval()
        total_loss = 0
        num_samples = 0
        metrics_sum = {name: 0.0 for name in self.metrics}
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation') as pbar:
                for images, masks in pbar:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    # 前向传播
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    
                    # 更新指标
                    batch_size = images.size(0)
                    total_loss += loss.item() * batch_size
                    num_samples += batch_size
                    
                    # 计算其他指标
                    for name, metric_fn in self.metrics.items():
                        metrics_sum[name] += metric_fn(outputs, masks) * batch_size
                        
                    # 更新进度条
                    pbar.set_postfix({
                        'loss': total_loss / num_samples,
                        **{name: metrics_sum[name] / num_samples for name in self.metrics}
                    })
                    
        # 计算平均指标
        avg_loss = total_loss / num_samples
        avg_metrics = {name: metrics_sum[name] / num_samples for name in self.metrics}
        
        return {'loss': avg_loss, **avg_metrics}
        
    def train(
        self,
        epochs: int,
        output_dir: Path,
        save_best: bool = True,
        early_stopping: Optional[int] = None,
        save_interval: Optional[int] = None
    ) -> Dict:
        """
        训练模型
        Args:
            epochs: 训练轮数
            output_dir: 输出目录
            save_best: 是否保存最佳模型
            early_stopping: 早停轮数
            save_interval: 保存间隔
        Returns:
            训练历史
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        best_epoch = 0
        no_improve = 0
        
        for epoch in range(epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # 训练
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            for name in self.metrics:
                if name not in self.history['train_metrics']:
                    self.history['train_metrics'][name] = []
                self.history['train_metrics'][name].append(train_metrics[name])
                
            # 验证
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            for name in self.metrics:
                if name not in self.history['val_metrics']:
                    self.history['val_metrics'][name] = []
                self.history['val_metrics'][name].append(val_metrics[name])
                
            # 记录学习率
            if self.scheduler is not None:
                self.history['lr'].append(self.scheduler.get_last_lr()[0])
                self.scheduler.step()
                
            # 保存最佳模型
            if save_best and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch
                self.save_checkpoint(output_dir / 'best_model.pth')
                no_improve = 0
            else:
                no_improve += 1
                
            # 定期保存
            if save_interval and (epoch + 1) % save_interval == 0:
                self.save_checkpoint(output_dir / f'model_epoch_{epoch + 1}.pth')
                
            # 早停
            if early_stopping and no_improve >= early_stopping:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
                
            # 打印指标
            self.logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}"
            )
            for name in self.metrics:
                self.logger.info(
                    f"Train {name}: {train_metrics[name]:.4f}, "
                    f"Val {name}: {val_metrics[name]:.4f}"
                )
                
        # 保存训练历史
        self.save_history(output_dir)
        self.plot_history(output_dir)
        
        return self.history
        
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """
        保存检查点
        Args:
            path: 保存路径
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, path)
        self.logger.info(f"检查点已保存至: {path}")
        
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """
        加载检查点
        Args:
            path: 检查点路径
        """
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.logger.info(f"检查点已加载: {path}")
        
    def save_history(self, output_dir: Path) -> None:
        """
        保存训练历史
        Args:
            output_dir: 输出目录
        """
        history_file = output_dir / 'training_history.json'
        
        # 转换tensor为float
        history_dict = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                history_dict[key] = [float(v) if torch.is_tensor(v) else v for v in value]
            elif isinstance(value, dict):
                history_dict[key] = {
                    k: [float(v) if torch.is_tensor(v) else v for v in vals]
                    for k, vals in value.items()
                }
            else:
                history_dict[key] = value
                
        with open(history_file, 'w') as f:
            json.dump(history_dict, f, indent=4)
            
        self.logger.info(f"训练历史已保存至: {history_file}")
        
    def plot_history(self, output_dir: Path) -> None:
        """
        绘制训练历史
        Args:
            output_dir: 输出目录
        """
        # 设置风格
        plt.style.use('seaborn')
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(output_dir / 'loss_history.png')
        plt.close()
        
        # 绘制指标曲线
        for metric_name in self.metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.history['train_metrics'][metric_name],
                label=f'Train {metric_name}'
            )
            plt.plot(
                self.history['val_metrics'][metric_name],
                label=f'Val {metric_name}'
            )
            plt.title(f'Training and Validation {metric_name}')
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.legend()
            plt.savefig(output_dir / f'{metric_name}_history.png')
            plt.close()
            
        # 绘制学习率曲线
        if self.history['lr']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.history['lr'])
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.savefig(output_dir / 'lr_history.png')
            plt.close()
            
        self.logger.info(f"训练历史图表已保存至: {output_dir}")
        
    def compute_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算所有指标
        Args:
            outputs: 模型输出
            targets: 目标值
        Returns:
            指标字典
        """
        return {
            name: metric_fn(outputs, targets)
            for name, metric_fn in self.metrics.items()
        } 