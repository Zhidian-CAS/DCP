import torch
import torch.nn as nn
from pathlib import Path
import argparse
from models.segmentation.training_manager import TrainingManager
from models.segmentation.unet import UNet
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Callable, Any, TypeVar, Generic
from dataclasses import dataclass
import functools
import threading
from queue import Queue
import concurrent.futures
from contextlib import contextmanager
import weakref
from itertools import chain
import operator
from collections import defaultdict, deque
import time

T = TypeVar('T')
ModelType = TypeVar('ModelType', bound=nn.Module)
DatasetType = TypeVar('DatasetType', bound=Dataset)

@dataclass
class ExperimentContext:
    """实验上下文数据类"""
    config_path: str
    train_dir: str
    train_mask_dir: str
    val_dir: str
    val_mask_dir: str
    experiment_name: str
    device: torch.device
    exp_dir: Path
    metrics_queue: Queue = Queue()
    resource_monitor: Optional['ResourceMonitor'] = None

class MetricsAggregator:
    """指标聚合器"""
    def __init__(self, window_size: int = 100):
        self.metrics_history = defaultdict(lambda: deque(maxlen=window_size))
        self.current_metrics = {}
        self._lock = threading.Lock()
        
    def update(self, metrics: Dict[str, float]) -> None:
        with self._lock:
            for key, value in metrics.items():
                self.metrics_history[key].append(value)
                self.current_metrics[key] = value
                
    def get_moving_average(self, metric_name: str) -> float:
        with self._lock:
            history = self.metrics_history[metric_name]
            return sum(history) / len(history) if history else 0.0
            
    def get_trend(self, metric_name: str) -> float:
        with self._lock:
            history = list(self.metrics_history[metric_name])
            if len(history) < 2:
                return 0.0
            return (history[-1] - history[0]) / len(history)

class ResourceMonitor:
    """资源监控器"""
    def __init__(self, context: ExperimentContext):
        self.context = context
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self._resource_history = defaultdict(list)
        
    def start(self):
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
    def stop(self):
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join()
            
    def _monitor_loop(self):
        while not self._stop_event.is_set():
            metrics = self.context.manager.monitor_resources()
            self._resource_history['timestamp'].append(time.time())
            for key, value in metrics.items():
                self._resource_history[key].append(value)
            time.sleep(1)

class ModelFactory(Generic[ModelType]):
    """模型工厂"""
    @staticmethod
    def create(config: Dict[str, Any]) -> ModelType:
        model_config = config['model']
        return UNet(
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            features=model_config['features']
        )

class TransformFactory:
    """转换工厂"""
    @staticmethod
    def create_transform_pipeline(config: Dict[str, Any]) -> Tuple[A.Compose, A.Compose]:
        def _create_augmentation(t: Dict[str, Any]) -> A.BasicTransform:
            name = t['name']
            if name == 'RandomHorizontalFlip':
                return A.HorizontalFlip(p=t['p'])
            elif name == 'RandomVerticalFlip':
                return A.VerticalFlip(p=t['p'])
            elif name == 'RandomRotation':
                return A.Rotate(limit=t['degrees'], p=0.5)
            elif name == 'ColorJitter':
                return A.ColorJitter(
                    brightness=t['brightness'],
                    contrast=t['contrast'],
                    saturation=t['saturation'],
                    p=0.5
                )
            raise ValueError(f"不支持的增强方法: {name}")
            
        train_transforms = [
            _create_augmentation(t)
            for t in config['data']['train_transforms']
        ]
        train_transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        val_transforms = [
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        return A.Compose(train_transforms), A.Compose(val_transforms)

class DatasetFactory(Generic[DatasetType]):
    """数据集工厂"""
    @staticmethod
    def create(
        image_dir: str,
        mask_dir: str,
        transform: Optional[A.Compose] = None
    ) -> DatasetType:
        return SegmentationDataset(image_dir, mask_dir, transform)

class LossFactory:
    """损失函数工厂"""
    @staticmethod
    def create(name: str = 'dice') -> Callable:
        if name == 'dice':
            return lambda pred, target: LossFactory._dice_loss(pred, target)
        raise ValueError(f"不支持的损失函数: {name}")
        
    @staticmethod
    def _dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1 - (2. * intersection + smooth) / (union + smooth)

class TrainingOrchestrator:
    """训练编排器"""
    def __init__(self, context: ExperimentContext):
        self.context = context
        self.metrics_aggregator = MetricsAggregator()
        self._setup_context()
        
    def _setup_context(self) -> None:
        self.context.resource_monitor = ResourceMonitor(self.context)
        
    @contextmanager
    def training_session(self):
        """训练会话上下文管理器"""
        self.context.resource_monitor.start()
        try:
            yield
        finally:
            self.context.resource_monitor.stop()
            
    def train(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_fn: Callable
    ) -> None:
        num_epochs = self.context.manager.config['training']['epochs']
        save_interval = self.context.manager.config['training']['save_interval']
        early_stopping = self.context.manager.config['training']['early_stopping']
        
        best_val_loss = float('inf')
        no_improve = 0
        
        with self.training_session():
            try:
                for epoch in range(num_epochs):
                    # 训练阶段
                    train_metrics = self._train_epoch(
                        model, train_loader, optimizer, loss_fn
                    )
                    
                    # 验证阶段
                    val_metrics = self._validate_epoch(
                        model, val_loader, loss_fn
                    )
                    
                    # 更新学习率
                    self._update_scheduler(scheduler, val_metrics['val_loss'])
                    
                    # 更新和记录指标
                    metrics = {**train_metrics, **val_metrics}
                    metrics['learning_rate'] = optimizer.param_groups[0]['lr']
                    self.metrics_aggregator.update(metrics)
                    self.context.manager.log_metrics(metrics, epoch)
                    
                    # 检查点保存
                    if (epoch + 1) % save_interval == 0:
                        self._save_checkpoint(
                            model, optimizer, scheduler, metrics,
                            f'model_epoch_{epoch + 1}.pth'
                        )
                    
                    # 最佳模型保存
                    if val_metrics['val_loss'] < best_val_loss:
                        best_val_loss = val_metrics['val_loss']
                        self._save_checkpoint(
                            model, optimizer, scheduler, metrics,
                            'best_model.pth'
                        )
                        no_improve = 0
                    else:
                        no_improve += 1
                        
                    # 早停检查
                    if no_improve >= early_stopping:
                        print(f"早停触发，{early_stopping}个epoch没有改善")
                        break
                        
            except KeyboardInterrupt:
                print("训练被用户中断")
                self._save_checkpoint(
                    model, optimizer, scheduler, metrics,
                    'interrupted_model.pth'
                )
                
    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable
    ) -> Dict[str, float]:
        model.train()
        metrics_computer = self._create_metrics_computer()
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(self.context.device)
            masks = masks.to(self.context.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            metrics_computer.update('train_loss', loss.item())
            metrics_computer.update('train_dice', 1 - loss.item())
            
            if batch_idx % 10 == 0:
                self.context.metrics_queue.put(metrics_computer.get_current())
                
        return metrics_computer.get_average()
        
    def _validate_epoch(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: Callable
    ) -> Dict[str, float]:
        model.eval()
        metrics_computer = self._create_metrics_computer()
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.context.device)
                masks = masks.to(self.context.device)
                
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                
                metrics_computer.update('val_loss', loss.item())
                metrics_computer.update('val_dice', 1 - loss.item())
                
        return metrics_computer.get_average()
        
    def _update_scheduler(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        val_loss: float
    ) -> None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
            
    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        metrics: Dict[str, float],
        filename: str
    ) -> None:
        self.context.manager.save_checkpoint(
            self.context.exp_dir / 'checkpoints' / filename,
            model,
            optimizer,
            scheduler,
            metrics
        )
        
    @staticmethod
    def _create_metrics_computer() -> 'MetricsComputer':
        return MetricsComputer()

class MetricsComputer:
    """指标计算器"""
    def __init__(self):
        self.metrics = defaultdict(lambda: {'sum': 0.0, 'count': 0})
        
    def update(self, name: str, value: float) -> None:
        self.metrics[name]['sum'] += value
        self.metrics[name]['count'] += 1
        
    def get_average(self) -> Dict[str, float]:
        return {
            name: stats['sum'] / stats['count']
            for name, stats in self.metrics.items()
        }
        
    def get_current(self) -> Dict[str, float]:
        return {
            name: stats['sum'] / stats['count']
            for name, stats in self.metrics.items()
        }

class SegmentationDataset(Dataset):
    """分割数据集"""
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_files = list(self.image_dir.glob('*.png'))
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_dir / image_path.name
        
        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
        return image, mask.unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description='使用TrainingManager训练分割模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--train-dir', type=str, required=True, help='训练图像目录')
    parser.add_argument('--train-mask-dir', type=str, required=True, help='训练掩码目录')
    parser.add_argument('--val-dir', type=str, required=True, help='验证图像目录')
    parser.add_argument('--val-mask-dir', type=str, required=True, help='验证掩码目录')
    parser.add_argument('--experiment-name', type=str, required=True, help='实验名称')
    args = parser.parse_args()
    
    # 创建实验上下文
    context = ExperimentContext(
        config_path=args.config,
        train_dir=args.train_dir,
        train_mask_dir=args.train_mask_dir,
        val_dir=args.val_dir,
        val_mask_dir=args.val_mask_dir,
        experiment_name=args.experiment_name,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        exp_dir=None
    )
    
    # 创建训练管理器
    context.manager = TrainingManager(
        config_path=args.config,
        experiment_name=args.experiment_name,
        use_wandb=True,
        debug_mode=False
    )
    
    # 创建实验目录
    context.exp_dir = context.manager.create_experiment_dir()
    
    # 创建转换
    train_transform, val_transform = TransformFactory.create_transform_pipeline(
        context.manager.config
    )
    
    # 创建数据集
    train_dataset = DatasetFactory.create(
        args.train_dir,
        args.train_mask_dir,
        train_transform
    )
    
    val_dataset = DatasetFactory.create(
        args.val_dir,
        args.val_mask_dir,
        val_transform
    )
    
    # 准备数据加载器
    train_loader, val_loader = context.manager.prepare_data(
        train_dataset,
        val_ratio=0.0,
        num_folds=None
    )
    
    # 创建模型
    model = ModelFactory.create(context.manager.config)
    model = model.to(context.device)
    
    # 创建优化器和调度器
    optimizer, scheduler = context.manager.create_optimizer(model)
    
    # 创建损失函数
    loss_fn = LossFactory.create('dice')
    
    # 创建训练编排器
    orchestrator = TrainingOrchestrator(context)
    
    try:
        # 开始训练
        orchestrator.train(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_fn
        )
        
        # 保存资源使用情况图表
        context.manager.plot_resource_usage(context.exp_dir / 'plots')
        
    finally:
        # 清理资源
        context.manager.cleanup()

if __name__ == '__main__':
    main() 