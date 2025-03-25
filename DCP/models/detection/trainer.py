import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np
from .metrics import DetectionMetrics
from .visualization import DetectionVisualizer

class DetectionTrainer:
    """目标检测训练器"""
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 optimizer: Optional[optim.Optimizer] = None,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 num_classes: int,
                 device: str = 'cuda',
                 save_dir: Optional[str] = None,
                 class_names: Optional[List[str]] = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or optim.AdamW(model.parameters(), lr=0.0001)
        self.scheduler = scheduler or optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.001,
            epochs=100,
            steps_per_epoch=len(train_loader)
        )
        self.num_classes = num_classes
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else None
        self.class_names = class_names
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化指标和可视化工具
        self.metrics = DetectionMetrics(num_classes)
        self.visualizer = DetectionVisualizer(num_classes, class_names, save_dir)
        
        # 将模型移到指定设备
        self.model.to(device)
        
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch
        Returns:
            metrics: 训练指标
        """
        self.model.train()
        epoch_metrics = defaultdict(list)
        
        # 创建进度条
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # 将数据移到设备
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # 前向传播
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # 反向传播
            self.optimizer.zero_grad()
            losses.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            self.scheduler.step()
            
            # 更新指标
            for k, v in loss_dict.items():
                epoch_metrics[k].append(v.item())
            epoch_metrics['total_loss'].append(losses.item())
            
            # 更新进度条
            pbar.set_postfix({k: f'{np.mean(v):.4f}' for k, v in epoch_metrics.items()})
            
        # 计算平均指标
        metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        return metrics
        
    def validate(self) -> Dict[str, float]:
        """
        验证模型
        Returns:
            metrics: 验证指标
        """
        if not self.val_loader:
            return {}
            
        self.model.eval()
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validating'):
                # 将数据移到设备
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # 前向传播
                outputs = self.model(images)
                
                # 更新评估指标
                self.metrics.update(
                    outputs['boxes'],
                    outputs['labels'],
                    outputs['scores'],
                    targets['boxes'],
                    targets['labels']
                )
                
                # 计算损失
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # 更新指标
                for k, v in loss_dict.items():
                    val_metrics[k].append(v.item())
                val_metrics['total_loss'].append(losses.item())
                
        # 计算评估指标
        eval_metrics = self.metrics.compute()
        val_metrics.update(eval_metrics)
        
        # 计算平均指标
        metrics = {k: np.mean(v) for k, v in val_metrics.items()}
        return metrics
        
    def train(self,
              num_epochs: int,
              save_freq: int = 1,
              early_stopping_patience: int = 10,
              resume_from: Optional[str] = None):
        """
        训练模型
        Args:
            num_epochs: 训练轮数
            save_freq: 保存频率
            early_stopping_patience: 早停耐心值
            resume_from: 恢复训练的检查点路径
        """
        # 恢复训练
        start_epoch = 0
        best_metric = float('-inf')
        patience_counter = 0
        
        if resume_from:
            checkpoint = torch.load(resume_from)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_metric']
            self.logger.info(f'Resumed from epoch {start_epoch}')
            
        # 训练循环
        for epoch in range(start_epoch, num_epochs):
            self.logger.info(f'Epoch {epoch+1}/{num_epochs}')
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate()
            
            # 记录指标
            metrics = {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
            self.logger.info(f'Epoch {epoch+1} metrics: {metrics}')
            
            # 可视化训练过程
            self.visualizer.plot_metrics(metrics, f'metrics_epoch_{epoch+1}.png')
            
            # 保存检查点
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(epoch, metrics, best_metric)
                
            # 早停检查
            current_metric = metrics.get('val_mAP', metrics.get('mAP', float('-inf')))
            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0
                self.save_checkpoint(epoch, metrics, best_metric, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f'Early stopping triggered after {epoch+1} epochs')
                    break
                    
    def save_checkpoint(self,
                       epoch: int,
                       metrics: Dict[str, float],
                       best_metric: float,
                       is_best: bool = False):
        """
        保存检查点
        Args:
            epoch: 当前轮数
            metrics: 评估指标
            best_metric: 最佳指标
            is_best: 是否为最佳模型
        """
        if not self.save_dir:
            return
            
        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_metric': best_metric
        }
        
        # 保存最新检查点
        latest_path = self.save_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # 保存当前轮数检查点
        epoch_path = self.save_dir / f'epoch_{epoch+1}.pth'
        torch.save(checkpoint, epoch_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.save_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            
        self.logger.info(f'Saved checkpoint to {epoch_path}')
        
    def predict(self,
               images: torch.Tensor,
               score_threshold: float = 0.5,
               nms_threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        模型预测
        Args:
            images: 输入图像
            score_threshold: 置信度阈值
            nms_threshold: NMS阈值
        Returns:
            predictions: 预测结果
        """
        self.model.eval()
        
        with torch.no_grad():
            # 将数据移到设备
            images = images.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            
            # 后处理
            predictions = self.model.postprocess(outputs, score_threshold, nms_threshold)
            
        return predictions
        
    def export_model(self, save_path: str):
        """
        导出模型
        Args:
            save_path: 保存路径
        """
        # 创建保存目录
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 导出模型
        torch.onnx.export(
            self.model,
            torch.randn(1, 3, 224, 224).to(self.device),
            save_path,
            input_names=['input'],
            output_names=['boxes', 'labels', 'scores'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'boxes': {0: 'batch_size'},
                'labels': {0: 'batch_size'},
                'scores': {0: 'batch_size'}
            }
        )
        
        self.logger.info(f'Exported model to {save_path}') 