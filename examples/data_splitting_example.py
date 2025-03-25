import torch
from torch.utils.data import Dataset
import numpy as np
from src.utils.data_splitter import DataSplitter
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DummyDataset(Dataset):
    """示例数据集"""
    def __init__(self, size=1000):
        self.size = size
        self.data = torch.randn(size, 3, 224, 224)  # 模拟图像数据
        self.labels = torch.randint(0, 2, (size,))  # 模拟标签
        self.timestamps = np.arange(size)  # 模拟时间戳
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return {
            'image': self.data[idx],
            'label': self.labels[idx],
            'timestamp': self.timestamps[idx]
        }

def main():
    # 创建示例数据集
    dataset = DummyDataset(size=1000)
    splitter = DataSplitter(dataset)
    
    # 1. 随机划分
    logger.info("执行随机划分...")
    train_set, val_set, test_set = splitter.random_split(
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42
    )
    logger.info(f"随机划分结果: 训练集 {len(train_set)}, 验证集 {len(val_set)}, 测试集 {len(test_set)}")
    
    # 2. 基于聚类的划分
    logger.info("\n执行基于聚类的划分...")
    train_set, val_set, test_set = splitter.cluster_based_split(
        n_clusters=3,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42
    )
    logger.info(f"聚类划分结果: 训练集 {len(train_set)}, 验证集 {len(val_set)}, 测试集 {len(test_set)}")
    
    # 3. 基于时间的划分
    logger.info("\n执行基于时间的划分...")
    train_set, val_set, test_set = splitter.time_based_split(
        time_column='timestamp',
        val_ratio=0.2,
        test_ratio=0.2
    )
    logger.info(f"时间划分结果: 训练集 {len(train_set)}, 验证集 {len(val_set)}, 测试集 {len(test_set)}")
    
    # 4. 前向交叉验证
    logger.info("\n执行前向交叉验证...")
    splits = splitter.forward_cross_validation(
        n_splits=5,
        val_ratio=0.2
    )
    for i, (train_set, val_set, test_set) in enumerate(splits):
        logger.info(f"第{i+1}折: 训练集 {len(train_set)}, 验证集 {len(val_set)}, 测试集 {len(test_set)}")

if __name__ == "__main__":
    main() 