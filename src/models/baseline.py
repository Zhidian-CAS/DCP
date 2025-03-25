import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class BaselineResults:
    """基线模型结果"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    training_time: float
    inference_time: float
    params: Dict[str, Union[int, float, str]]

class BaselineComparator:
    """基线模型比较器"""
    
    def __init__(self):
        """初始化基线模型比较器"""
        self.logger = logging.getLogger(__name__)
        self.baselines = {}
        self.results = {}
        
    def add_baseline(
        self,
        name: str,
        model: Union[nn.Module, object],
        params: Dict[str, Union[int, float, str]]
    ):
        """
        添加基线模型
        Args:
            name: 模型名称
            model: 模型实例
            params: 模型参数
        """
        self.baselines[name] = {
            'model': model,
            'params': params
        }
        
    def train_and_evaluate(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        batch_size: int = 32
    ) -> Dict[str, BaselineResults]:
        """
        训练和评估所有基线模型
        Args:
            train_data: 训练数据
            train_labels: 训练标签
            test_data: 测试数据
            test_labels: 测试标签
            batch_size: 批处理大小
        Returns:
            评估结果字典
        """
        results = {}
        
        for name, baseline in self.baselines.items():
            self.logger.info(f"训练和评估基线模型: {name}")
            
            # 训练模型
            import time
            start_time = time.time()
            
            if isinstance(baseline['model'], nn.Module):
                # PyTorch模型
                self._train_pytorch_model(
                    baseline['model'],
                    train_data,
                    train_labels,
                    batch_size
                )
            else:
                # scikit-learn模型
                baseline['model'].fit(train_data, train_labels)
                
            training_time = time.time() - start_time
            
            # 评估模型
            start_time = time.time()
            
            if isinstance(baseline['model'], nn.Module):
                # PyTorch模型
                predictions = self._predict_pytorch_model(
                    baseline['model'],
                    test_data,
                    batch_size
                )
            else:
                # scikit-learn模型
                predictions = baseline['model'].predict(test_data)
                
            inference_time = time.time() - start_time
            
            # 计算指标
            metrics = self._compute_metrics(test_labels, predictions)
            
            # 保存结果
            results[name] = BaselineResults(
                model_name=name,
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1=metrics['f1'],
                training_time=training_time,
                inference_time=inference_time,
                params=baseline['params']
            )
            
        self.results = results
        return results
        
    def _train_pytorch_model(
        self,
        model: nn.Module,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        batch_size: int
    ):
        """
        训练PyTorch模型
        Args:
            model: PyTorch模型
            train_data: 训练数据
            train_labels: 训练标签
            batch_size: 批处理大小
        """
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # 转换为PyTorch张量
        train_data = torch.FloatTensor(train_data)
        train_labels = torch.LongTensor(train_labels)
        
        # 批处理训练
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
    def _predict_pytorch_model(
        self,
        model: nn.Module,
        test_data: np.ndarray,
        batch_size: int
    ) -> np.ndarray:
        """
        使用PyTorch模型进行预测
        Args:
            model: PyTorch模型
            test_data: 测试数据
            batch_size: 批处理大小
        Returns:
            预测结果
        """
        model.eval()
        predictions = []
        
        # 转换为PyTorch张量
        test_data = torch.FloatTensor(test_data)
        
        # 批处理预测
        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch_data = test_data[i:i+batch_size]
                outputs = model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.numpy())
                
        return np.array(predictions)
        
    def _compute_metrics(
        self,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        计算评估指标
        Args:
            true_labels: 真实标签
            predicted_labels: 预测标签
        Returns:
            指标字典
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            'accuracy': accuracy_score(true_labels, predicted_labels),
            'precision': precision_score(true_labels, predicted_labels, average='weighted'),
            'recall': recall_score(true_labels, predicted_labels, average='weighted'),
            'f1': f1_score(true_labels, predicted_labels, average='weighted')
        }
        
    def get_comparison_summary(self) -> str:
        """
        获取比较结果摘要
        Returns:
            摘要文本
        """
        if not self.results:
            return "没有可用的比较结果"
            
        summary = "基线模型比较结果:\n\n"
        
        # 按准确率排序
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].accuracy,
            reverse=True
        )
        
        for name, result in sorted_results:
            summary += f"模型: {name}\n"
            summary += f"准确率: {result.accuracy:.4f}\n"
            summary += f"精确率: {result.precision:.4f}\n"
            summary += f"召回率: {result.recall:.4f}\n"
            summary += f"F1分数: {result.f1:.4f}\n"
            summary += f"训练时间: {result.training_time:.2f}秒\n"
            summary += f"推理时间: {result.inference_time:.2f}秒\n"
            summary += f"参数: {result.params}\n\n"
            
        return summary 