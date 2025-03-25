from typing import Dict, List, Optional, Union
import yaml
from pathlib import Path
import logging
from dataclasses import dataclass
from omegaconf import OmegaConf

@dataclass
class ModelConfig:
    """模型配置"""
    # 基础配置
    backbone: str = 'resnet50'
    num_classes: int = 2
    pretrained: bool = True
    
    # RPN配置
    rpn_anchor_sizes: List[int] = (32, 64, 128, 256, 512)
    rpn_aspect_ratios: List[float] = (0.5, 1.0, 2.0)
    rpn_fg_iou_thresh: float = 0.7
    rpn_bg_iou_thresh: float = 0.3
    rpn_batch_size_per_image: int = 256
    rpn_positive_fraction: float = 0.5
    rpn_pre_nms_top_n: Dict[str, int] = None
    rpn_post_nms_top_n: Dict[str, int] = None
    rpn_nms_thresh: float = 0.7
    rpn_score_thresh: float = 0.0
    
    # RoI配置
    box_roi_pool: str = 'roi_align'
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5
    box_detections_per_img: int = 100
    
    # 损失配置
    loss_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.rpn_pre_nms_top_n is None:
            self.rpn_pre_nms_top_n = {'training': 2000, 'testing': 1000}
        if self.rpn_post_nms_top_n is None:
            self.rpn_post_nms_top_n = {'training': 1000, 'testing': 1000}
        if self.loss_weights is None:
            self.loss_weights = {
                'loss_classifier': 1.0,
                'loss_box_reg': 1.0,
                'loss_objectness': 1.0,
                'loss_rpn_box_reg': 1.0
            }

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础配置
    num_epochs: int = 100
    batch_size: int = 8
    num_workers: int = 4
    device: str = 'cuda'
    
    # 优化器配置
    optimizer: str = 'adamw'
    learning_rate: float = 0.0001
    weight_decay: float = 0.0005
    momentum: float = 0.9
    
    # 学习率调度器配置
    scheduler: str = 'one_cycle'
    max_lr: float = 0.001
    min_lr: float = 0.00001
    warmup_epochs: int = 5
    
    # 训练策略配置
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10
    save_freq: int = 1
    
    # 数据增强配置
    train_augmentation: Dict[str, bool] = None
    val_augmentation: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.train_augmentation is None:
            self.train_augmentation = {
                'horizontal_flip': True,
                'vertical_flip': True,
                'rotation': True,
                'color_jitter': True,
                'random_scale': True
            }
        if self.val_augmentation is None:
            self.val_augmentation = {
                'horizontal_flip': False,
                'vertical_flip': False,
                'rotation': False,
                'color_jitter': False,
                'random_scale': False
            }

@dataclass
class DataConfig:
    """数据配置"""
    # 数据集配置
    train_data_dir: str
    val_data_dir: Optional[str] = None
    test_data_dir: Optional[str] = None
    
    # 图像配置
    image_size: List[int] = (224, 224)
    mean: List[float] = (0.485, 0.456, 0.406)
    std: List[float] = (0.229, 0.224, 0.225)
    
    # 数据加载配置
    shuffle: bool = True
    pin_memory: bool = True
    drop_last: bool = True
    
    # 数据格式配置
    image_format: str = 'jpg'
    annotation_format: str = 'json'
    
    def __post_init__(self):
        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]

@dataclass
class LoggingConfig:
    """日志配置"""
    # 日志级别
    level: str = 'INFO'
    
    # 日志文件配置
    log_dir: Optional[str] = None
    log_file: Optional[str] = None
    
    # 日志格式
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 日志轮转配置
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

@dataclass
class Config:
    """总配置"""
    # 基础配置
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    logging: LoggingConfig
    
    # 保存和加载配置
    save_dir: Optional[str] = None
    resume_from: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        从YAML文件加载配置
        Args:
            yaml_path: YAML文件路径
        Returns:
            config: 配置对象
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            
        # 创建配置对象
        config = cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            save_dir=config_dict.get('save_dir'),
            resume_from=config_dict.get('resume_from')
        )
        
        return config
        
    def to_yaml(self, yaml_path: str):
        """
        保存配置到YAML文件
        Args:
            yaml_path: YAML文件路径
        """
        # 创建保存目录
        save_dir = Path(yaml_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 转换为字典
        config_dict = OmegaConf.to_container(self)
        
        # 保存到文件
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            
    def setup_logging(self):
        """设置日志"""
        # 创建日志目录
        if self.logging.log_dir:
            log_dir = Path(self.logging.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
        # 配置日志
        logging.basicConfig(
            level=getattr(logging, self.logging.level),
            format=self.logging.format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    Path(self.logging.log_dir) / self.logging.log_file
                    if self.logging.log_dir and self.logging.log_file
                    else None
                )
            ]
        ) 