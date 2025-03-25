import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional, Union
import pandas as pd
from datetime import datetime
import logging

class DataSplitter:
    """数据划分工具类"""
    
    def __init__(self, dataset: Dataset):
        """
        初始化数据划分器
        Args:
            dataset: 待划分的数据集
        """
        self.dataset = dataset
        self.logger = logging.getLogger(__name__)
        
    def random_split(
        self,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: Optional[int] = None
    ) -> Tuple[Subset, Subset, Subset]:
        """
        随机划分数据集
        Args:
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
        Returns:
            训练集、验证集、测试集
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        total_size = len(self.dataset)
        test_size = int(total_size * test_ratio)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size - test_size
        
        train_subset, val_subset, test_subset = random_split(
            self.dataset,
            [train_size, val_size, test_size]
        )
        
        return train_subset, val_subset, test_subset
        
    def cluster_based_split(
        self,
        n_clusters: int = 3,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: Optional[int] = None
    ) -> Tuple[Subset, Subset, Subset]:
        """
        基于聚类的划分
        Args:
            n_clusters: 聚类数量
            val_ratio: 验证集比例
            test_ratio: 验证集比例
            seed: 随机种子
        Returns:
            训练集、验证集、测试集
        """
        if seed is not None:
            np.random.seed(seed)
            
        # 提取特征用于聚类
        features = []
        for i in range(len(self.dataset)):
            # 假设数据集返回字典，包含'image'和'mask'
            data = self.dataset[i]
            # 使用图像特征进行聚类
            if isinstance(data, dict):
                image = data['image']
            else:
                image = data[0]
                
            # 将图像展平并归一化
            feature = image.flatten() / 255.0
            features.append(feature)
            
        features = np.array(features)
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        cluster_labels = kmeans.fit_predict(features)
        
        # 基于聚类结果划分数据集
        indices = np.arange(len(self.dataset))
        train_indices = []
        val_indices = []
        test_indices = []
        
        for cluster in range(n_clusters):
            cluster_indices = indices[cluster_labels == cluster]
            np.random.shuffle(cluster_indices)
            
            # 计算每个集合的大小
            cluster_size = len(cluster_indices)
            val_size = int(cluster_size * val_ratio)
            test_size = int(cluster_size * test_ratio)
            
            # 划分索引
            val_indices.extend(cluster_indices[:val_size])
            test_indices.extend(cluster_indices[val_size:val_size+test_size])
            train_indices.extend(cluster_indices[val_size+test_size:])
            
        return (
            Subset(self.dataset, train_indices),
            Subset(self.dataset, val_indices),
            Subset(self.dataset, test_indices)
        )
        
    def time_based_split(
        self,
        time_column: str,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2
    ) -> Tuple[Subset, Subset, Subset]:
        """
        基于时间的划分
        Args:
            time_column: 时间列名
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        Returns:
            训练集、验证集、测试集
        """
        # 假设数据集包含时间信息
        if not hasattr(self.dataset, 'df'):
            raise ValueError("数据集必须包含DataFrame属性")
            
        df = self.dataset.df
        df = df.sort_values(time_column)
        
        total_size = len(df)
        test_size = int(total_size * test_ratio)
        val_size = int(total_size * val_ratio)
        
        # 按时间顺序划分
        train_indices = df.index[:-val_size-test_size].tolist()
        val_indices = df.index[-val_size-test_size:-test_size].tolist()
        test_indices = df.index[-test_size:].tolist()
        
        return (
            Subset(self.dataset, train_indices),
            Subset(self.dataset, val_indices),
            Subset(self.dataset, test_indices)
        )
        
    def forward_cross_validation(
        self,
        n_splits: int = 5,
        val_ratio: float = 0.2
    ) -> List[Tuple[Subset, Subset, Subset]]:
        """
        前向交叉验证
        Args:
            n_splits: 划分数量
            val_ratio: 验证集比例
        Returns:
            训练集、验证集、测试集列表
        """
        total_size = len(self.dataset)
        split_size = total_size // n_splits
        
        splits = []
        for i in range(n_splits):
            # 计算当前划分的索引范围
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < n_splits - 1 else total_size
            
            # 测试集为当前划分
            test_indices = list(range(start_idx, end_idx))
            
            # 训练集为之前的所有数据
            train_indices = list(range(0, start_idx))
            
            # 验证集为训练集的一部分
            val_size = int(len(train_indices) * val_ratio)
            val_indices = train_indices[-val_size:]
            train_indices = train_indices[:-val_size]
            
            splits.append((
                Subset(self.dataset, train_indices),
                Subset(self.dataset, val_indices),
                Subset(self.dataset, test_indices)
            ))
            
        return splits 