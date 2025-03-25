import logging
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import torch
import torch.nn as nn
from src.models.baseline import BaselineComparator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleNN(nn.Module):
    """简单的神经网络模型"""
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def generate_dummy_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 2
) -> tuple:
    """
    生成示例数据
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        n_classes: 类别数量
    Returns:
        训练数据和标签
    """
    # 生成随机特征
    X = np.random.randn(n_samples, n_features)
    
    # 生成随机标签
    y = np.random.randint(0, n_classes, n_samples)
    
    # 添加一些噪声使数据更真实
    X += np.random.randn(n_samples, n_features) * 0.1
    
    return X, y

def main():
    # 生成示例数据
    logger.info("生成示例数据...")
    X, y = generate_dummy_data()
    
    # 划分训练集和测试集
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 创建基线模型比较器
    comparator = BaselineComparator()
    
    # 添加基线模型
    
    # 1. 1-最近邻
    logger.info("添加1-最近邻模型...")
    comparator.add_baseline(
        name="1-Nearest Neighbor",
        model=KNeighborsClassifier(n_neighbors=1),
        params={"n_neighbors": 1}
    )
    
    # 2. 随机森林
    logger.info("添加随机森林模型...")
    comparator.add_baseline(
        name="Random Forest",
        model=RandomForestClassifier(n_estimators=100),
        params={"n_estimators": 100}
    )
    
    # 3. 最频繁类别
    logger.info("添加最频繁类别模型...")
    comparator.add_baseline(
        name="Most Frequent Class",
        model=DummyClassifier(strategy="most_frequent"),
        params={"strategy": "most_frequent"}
    )
    
    # 4. 简单神经网络
    logger.info("添加简单神经网络模型...")
    input_size = X.shape[1]
    num_classes = len(np.unique(y))
    comparator.add_baseline(
        name="Simple Neural Network",
        model=SimpleNN(input_size, num_classes),
        params={
            "input_size": input_size,
            "num_classes": num_classes,
            "architecture": "3-layer MLP"
        }
    )
    
    # 训练和评估所有基线模型
    logger.info("开始训练和评估基线模型...")
    results = comparator.train_and_evaluate(
        X_train, y_train,
        X_test, y_test,
        batch_size=32
    )
    
    # 输出比较结果
    logger.info("\n" + comparator.get_comparison_summary())
    
    # 保存结果到文件
    with open("baseline_comparison_results.txt", "w") as f:
        f.write(comparator.get_comparison_summary())
    logger.info("结果已保存到 baseline_comparison_results.txt")

if __name__ == "__main__":
    main() 