import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import cv2
import logging
from .augmentation import SegmentationAugmentation

class SegmentationDataset(Dataset):
    """分割数据集"""
    def __init__(self,
                 image_dir: Union[str, Path],
                 mask_dir: Union[str, Path],
                 transform: Optional[SegmentationAugmentation] = None,
                 class_names: Optional[List[str]] = None,
                 class_colors: Optional[List[Tuple[int, int, int]]] = None):
        """
        初始化
        Args:
            image_dir: 图像目录
            mask_dir: 掩码目录
            transform: 数据增强器
            class_names: 类别名称列表
            class_colors: 类别颜色列表
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.class_names = class_names or ['background', 'foreground']
        self.class_colors = class_colors or [(0, 0, 0), (255, 0, 0)]
        
        # 日志配置
        self.logger = logging.getLogger(__name__)
        
        # 验证目录
        if not self.image_dir.exists():
            raise ValueError(f"图像目录不存在: {self.image_dir}")
        if not self.mask_dir.exists():
            raise ValueError(f"掩码目录不存在: {self.mask_dir}")
            
        # 获取图像和掩码文件列表
        self.image_files = sorted([
            f for f in self.image_dir.glob("*")
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ])
        self.mask_files = sorted([
            f for f in self.mask_dir.glob("*")
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
        ])
        
        # 验证文件数量
        if len(self.image_files) != len(self.mask_files):
            raise ValueError(
                f"图像数量({len(self.image_files)})与掩码数量({len(self.mask_files)})不匹配"
            )
            
        # 验证文件对应关系
        self._validate_file_pairs()
        
        self.logger.info(f"加载数据集: {len(self.image_files)}个样本")
        
    def _validate_file_pairs(self):
        """验证图像和掩码文件的对应关系"""
        for img_file, mask_file in zip(self.image_files, self.mask_files):
            if img_file.stem != mask_file.stem:
                raise ValueError(
                    f"图像和掩码文件名不匹配: {img_file.name} vs {mask_file.name}"
                )
                
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.image_files)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据样本
        Args:
            idx: 样本索引
        Returns:
            sample: 数据样本字典
        """
        # 加载图像和掩码
        image = cv2.imread(str(self.image_files[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_files[idx]), cv2.IMREAD_GRAYSCALE)
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
        return {
            'image': image,
            'mask': mask,
            'image_path': str(self.image_files[idx]),
            'mask_path': str(self.mask_files[idx])
        }
        
    def get_class_weights(self) -> torch.Tensor:
        """
        计算类别权重
        Returns:
            weights: 类别权重张量
        """
        # 统计每个类别的像素数量
        class_counts = np.zeros(len(self.class_names))
        total_pixels = 0
        
        for mask_file in self.mask_files:
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            for i in range(len(self.class_names)):
                class_counts[i] += np.sum(mask == i)
            total_pixels += mask.size
            
        # 计算权重
        weights = total_pixels / (len(self.class_names) * class_counts + 1e-5)
        weights = weights / np.sum(weights)
        
        return torch.FloatTensor(weights)
        
    def visualize_sample(self,
                        idx: int,
                        save_path: Optional[str] = None) -> np.ndarray:
        """
        可视化样本
        Args:
            idx: 样本索引
            save_path: 保存路径
        Returns:
            vis_image: 可视化图像
        """
        sample = self.__getitem__(idx)
        image = sample['image']
        mask = sample['mask']
        
        # 反归一化图像
        if self.transform:
            image = self.transform.denormalize(image)
            
        # 创建彩色掩码
        mask_color = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for i in range(len(self.class_names)):
            mask_color[mask == i] = self.class_colors[i]
            
        # 叠加显示
        vis_image = cv2.addWeighted(image, 0.7, mask_color, 0.3, 0)
        
        # 添加类别标签
        for i, (name, color) in enumerate(zip(self.class_names, self.class_colors)):
            cv2.putText(
                vis_image,
                name,
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )
            
        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
        return vis_image
        
def create_data_loaders(
    train_dataset: SegmentationDataset,
    valid_dataset: Optional[SegmentationDataset] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    创建数据加载器
    Args:
        train_dataset: 训练数据集
        valid_dataset: 验证数据集
        batch_size: 批次大小
        num_workers: 工作进程数
        pin_memory: 是否使用固定内存
        shuffle: 是否打乱数据
    Returns:
        train_loader: 训练数据加载器
        valid_loader: 验证数据加载器
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    valid_loader = None
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
    return train_loader, valid_loader 