import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import logging
from pathlib import Path
import json
import yaml
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import psutil
import GPUtil
from collections import defaultdict
import time
import signal
import sys
import os

class TrainingManager:
    """高级训练管理器"""
    
    def __init__(
        self,
        config_path: Union[str, Path],
        experiment_name: str,
        use_wandb: bool = False,
        debug_mode: bool = False
    ):
        """
        初始化训练管理器
        Args:
            config_path: 配置文件路径
            experiment_name: 实验名称
            use_wandb: 是否使用wandb
            debug_mode: 是否为调试模式
        """
        self.config = self._load_config(config_path)
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.debug_mode = debug_mode
        self.logger = self._setup_logging()
        self.device = self._setup_device()
        self.distributed = False
        self.world_size = 1
        self.rank = 0
        
        # 资源监控
        self.resource_usage = defaultdict(list)
        self.start_time = time.time()
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        
        if use_wandb:
            self._setup_wandb()
            
    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """加载配置文件"""
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        # 验证配置
        required_keys = ['training', 'model', 'data', 'optimization']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置文件缺少必要的{key}部分")
                
        return config
        
    def _setup_logging(self) -> logging.Logger:
        """配置日志系统"""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)
        
        # 文件处理器
        log_dir = Path('logs') / self.experiment_name
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{datetime.now():%Y%m%d_%H%M%S}.log")
        fh.setLevel(logging.DEBUG)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
        
    def _setup_device(self) -> torch.device:
        """配置运行设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # 设置CUDA设备属性
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            # 清理GPU缓存
            torch.cuda.empty_cache()
        else:
            device = torch.device('cpu')
            
        self.logger.info(f"使用设备: {device}")
        return device
        
    def _setup_wandb(self) -> None:
        """配置Weights & Biases"""
        wandb.init(
            project=self.experiment_name,
            config=self.config,
            name=f"{self.experiment_name}_{datetime.now():%Y%m%d_%H%M%S}"
        )
        
    def _handle_interrupt(self, signum, frame):
        """处理中断信号"""
        self.logger.warning("接收到中断信号，正在保存状态...")
        self.save_checkpoint("interrupted_checkpoint.pth")
        sys.exit(0)
        
    def setup_distributed(self, world_size: int) -> None:
        """
        配置分布式训练
        Args:
            world_size: 进程数量
        """
        self.distributed = True
        self.world_size = world_size
        
        def setup(rank):
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            self.rank = rank
            
        mp.spawn(setup, nprocs=world_size)
        
    def prepare_data(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_ratio: float = 0.2,
        num_folds: Optional[int] = None
    ) -> Union[Tuple[DataLoader, DataLoader], List[Tuple[DataLoader, DataLoader]]]:
        """
        准备数据加载器
        Args:
            train_dataset: 训练数据集
            val_ratio: 验证集比例
            num_folds: K折交叉验证的折数
        Returns:
            数据加载器或数据加载器列表
        """
        if num_folds is not None:
            # K折交叉验证
            kfold = KFold(n_splits=num_folds, shuffle=True)
            dataloaders = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
                val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
                
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config['data']['batch_size'],
                    sampler=train_subsampler,
                    num_workers=self.config['data']['num_workers'],
                    pin_memory=True
                )
                
                val_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config['data']['batch_size'],
                    sampler=val_subsampler,
                    num_workers=self.config['data']['num_workers'],
                    pin_memory=True
                )
                
                dataloaders.append((train_loader, val_loader))
                
            return dataloaders
        else:
            # 简单分割
            val_size = int(len(train_dataset) * val_ratio)
            train_size = len(train_dataset) - val_size
            
            train_subset, val_subset = random_split(
                train_dataset,
                [train_size, val_size]
            )
            
            train_loader = DataLoader(
                train_subset,
                batch_size=self.config['data']['batch_size'],
                shuffle=True,
                num_workers=self.config['data']['num_workers'],
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=self.config['data']['batch_size'],
                shuffle=False,
                num_workers=self.config['data']['num_workers'],
                pin_memory=True
            )
            
            return train_loader, val_loader
            
    def create_optimizer(
        self,
        model: nn.Module,
        optimizer_name: str = 'adamw'
    ) -> Tuple[optim.Optimizer, _LRScheduler]:
        """
        创建优化器和学习率调度器
        Args:
            model: 模型
            optimizer_name: 优化器名称
        Returns:
            优化器和调度器
        """
        # 获取优化器参数
        opt_config = self.config['optimization']
        
        # 创建优化器
        if optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config['weight_decay']
            )
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=opt_config['learning_rate'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
            
        # 创建调度器
        scheduler_config = opt_config.get('scheduler', {})
        scheduler_name = scheduler_config.get('name', 'cosine')
        
        if scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=opt_config['epochs'],
                eta_min=opt_config['learning_rate'] * 0.01
            )
        elif scheduler_name == 'onecycle':
            scheduler = OneCycleLR(
                optimizer,
                max_lr=opt_config['learning_rate'],
                epochs=opt_config['epochs'],
                steps_per_epoch=len(train_loader)
            )
        elif scheduler_name == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.1),
                patience=scheduler_config.get('patience', 10)
            )
        else:
            raise ValueError(f"不支持的调度器: {scheduler_name}")
            
        return optimizer, scheduler
        
    def monitor_resources(self) -> Dict[str, float]:
        """监控系统资源使用情况"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent()
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU使用率
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            gpu_load = gpu.load
            gpu_memory = gpu.memoryUtil
        else:
            gpu_load = 0
            gpu_memory = 0
            
        metrics = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'gpu_load': gpu_load,
            'gpu_memory': gpu_memory
        }
        
        # 更新历史记录
        for key, value in metrics.items():
            self.resource_usage[key].append(value)
            
        return metrics
        
    def plot_resource_usage(self, output_dir: Path) -> None:
        """
        绘制资源使用情况
        Args:
            output_dir: 输出目录
        """
        plt.style.use('seaborn')
        
        # 创建时间轴
        times = np.linspace(0, (time.time() - self.start_time) / 3600, len(next(iter(self.resource_usage.values()))))
        
        # 绘制资源使用曲线
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # CPU和内存使用率
        ax1.plot(times, self.resource_usage['cpu_percent'], label='CPU')
        ax1.plot(times, self.resource_usage['memory_percent'], label='Memory')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Usage (%)')
        ax1.set_title('CPU and Memory Usage')
        ax1.legend()
        
        # GPU使用率
        ax2.plot(times, self.resource_usage['gpu_load'], label='GPU Load')
        ax2.plot(times, self.resource_usage['gpu_memory'], label='GPU Memory')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Usage (%)')
        ax2.set_title('GPU Usage')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'resource_usage.png')
        plt.close()
        
    def save_checkpoint(
        self,
        path: Union[str, Path],
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        metrics: Optional[Dict[str, float]] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        保存检查点
        Args:
            path: 保存路径
            model: 模型
            optimizer: 优化器
            scheduler: 调度器
            metrics: 指标
            extra_data: 额外数据
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config,
            'resource_usage': dict(self.resource_usage),
            'timestamp': datetime.now().isoformat()
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        if metrics is not None:
            checkpoint['metrics'] = metrics
            
        if extra_data is not None:
            checkpoint['extra_data'] = extra_data
            
        torch.save(checkpoint, path)
        self.logger.info(f"检查点已保存至: {path}")
        
    def load_checkpoint(
        self,
        path: Union[str, Path],
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[_LRScheduler] = None
    ) -> Dict[str, Any]:
        """
        加载检查点
        Args:
            path: 检查点路径
            model: 模型
            optimizer: 优化器
            scheduler: 调度器
        Returns:
            检查点数据
        """
        checkpoint = torch.load(path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.resource_usage = defaultdict(list, checkpoint['resource_usage'])
        
        self.logger.info(f"检查点已加载: {path}")
        return checkpoint
        
    def create_experiment_dir(self) -> Path:
        """
        创建实验目录
        Returns:
            实验目录路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = Path('experiments') / f"{self.experiment_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置文件
        with open(exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
            
        # 创建子目录
        (exp_dir / 'checkpoints').mkdir()
        (exp_dir / 'logs').mkdir()
        (exp_dir / 'plots').mkdir()
        (exp_dir / 'results').mkdir()
        
        return exp_dir
        
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        记录指标
        Args:
            metrics: 指标字典
            step: 步数
        """
        # 记录到日志
        metrics_str = ', '.join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info(f"Step {step}: {metrics_str}")
        
        # 记录到wandb
        if self.use_wandb:
            wandb.log(metrics, step=step)
            
    def cleanup(self) -> None:
        """清理资源"""
        if self.use_wandb:
            wandb.finish()
            
        if self.distributed:
            dist.destroy_process_group()
            
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info("资源清理完成") 