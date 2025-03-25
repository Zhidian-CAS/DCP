import logging
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
import json
import yaml
from datetime import datetime

@dataclass
class SOTAModelInfo:
    """SOTA模型信息"""
    name: str
    paper_title: str
    authors: List[str]
    year: int
    conference: str
    metrics: Dict[str, float]
    implementation_url: Optional[str] = None
    pretrained_url: Optional[str] = None
    architecture: Optional[str] = None
    training_details: Optional[Dict] = None

@dataclass
class ComparisonResult:
    """比较结果"""
    model_name: str
    sota_name: str
    metrics: Dict[str, float]
    relative_performance: Dict[str, float]
    hardware_requirements: Dict[str, Union[float, str]]
    training_time: float
    inference_time: float
    memory_usage: float
    comparison_date: str

class SOTAComparator:
    """SOTA模型比较器"""
    
    def __init__(self, sota_config_path: Union[str, Path]):
        """
        初始化SOTA比较器
        Args:
            sota_config_path: SOTA模型配置文件路径
        """
        self.logger = logging.getLogger(__name__)
        self.sota_config_path = Path(sota_config_path)
        self.sota_models = self._load_sota_config()
        
    def _load_sota_config(self) -> Dict[str, SOTAModelInfo]:
        """
        加载SOTA模型配置
        Returns:
            SOTA模型信息字典
        """
        if not self.sota_config_path.exists():
            raise FileNotFoundError(f"SOTA配置文件不存在: {self.sota_config_path}")
            
        with open(self.sota_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        sota_models = {}
        for name, info in config.items():
            sota_models[name] = SOTAModelInfo(**info)
            
        return sota_models
        
    def compare_with_sota(
        self,
        model: nn.Module,
        model_name: str,
        sota_name: str,
        test_dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda',
        num_runs: int = 100,
        warmup: int = 10
    ) -> ComparisonResult:
        """
        与SOTA模型比较
        Args:
            model: 待评估模型
            model_name: 模型名称
            sota_name: SOTA模型名称
            test_dataloader: 测试数据加载器
            device: 设备
            num_runs: 运行次数
            warmup: 预热次数
        Returns:
            比较结果
        """
        if sota_name not in self.sota_models:
            raise ValueError(f"未知的SOTA模型: {sota_name}")
            
        sota_info = self.sota_models[sota_name]
        model = model.to(device)
        
        # 评估模型性能
        metrics = self._evaluate_model(
            model,
            test_dataloader,
            device,
            num_runs,
            warmup
        )
        
        # 计算相对性能
        relative_performance = self._compute_relative_performance(
            metrics,
            sota_info.metrics
        )
        
        # 获取硬件需求
        hardware_requirements = self._get_hardware_requirements(model)
        
        # 创建比较结果
        result = ComparisonResult(
            model_name=model_name,
            sota_name=sota_name,
            metrics=metrics,
            relative_performance=relative_performance,
            hardware_requirements=hardware_requirements,
            training_time=metrics['training_time'],
            inference_time=metrics['inference_time'],
            memory_usage=metrics['memory_usage'],
            comparison_date=datetime.now().isoformat()
        )
        
        return result
        
    def _evaluate_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str,
        num_runs: int,
        warmup: int
    ) -> Dict[str, float]:
        """
        评估模型性能
        Args:
            model: 模型
            dataloader: 数据加载器
            device: 设备
            num_runs: 运行次数
            warmup: 预热次数
        Returns:
            性能指标字典
        """
        model.eval()
        metrics = {}
        
        # 预热
        for _ in range(warmup):
            self._run_inference(model, dataloader, device)
            
        # 性能测试
        times = []
        memory_usage = []
        
        for _ in range(num_runs):
            # 记录开始时间
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            # 记录开始内存
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                
            start_time.record()
            self._run_inference(model, dataloader, device)
            end_time.record()
            
            torch.cuda.synchronize()
            times.append(start_time.elapsed_time(end_time))
            
            # 记录内存使用
            if device == 'cuda':
                memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
                
        # 计算统计信息
        times = np.array(times)
        metrics['inference_time'] = float(np.mean(times))
        metrics['inference_time_std'] = float(np.std(times))
        metrics['memory_usage'] = float(np.mean(memory_usage)) if memory_usage else 0
        
        # 计算准确率等指标
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        metrics['accuracy'] = correct / total
        
        return metrics
        
    def _run_inference(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str
    ):
        """运行推理"""
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                _ = model(images)
                
    def _compute_relative_performance(
        self,
        metrics: Dict[str, float],
        sota_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        计算相对性能
        Args:
            metrics: 当前模型指标
            sota_metrics: SOTA模型指标
        Returns:
            相对性能字典
        """
        relative = {}
        for metric in metrics:
            if metric in sota_metrics:
                if metric in ['inference_time', 'memory_usage']:
                    # 越小越好
                    relative[metric] = sota_metrics[metric] / metrics[metric]
                else:
                    # 越大越好
                    relative[metric] = metrics[metric] / sota_metrics[metric]
        return relative
        
    def _get_hardware_requirements(
        self,
        model: nn.Module
    ) -> Dict[str, Union[float, str]]:
        """
        获取硬件需求
        Args:
            model: 模型
        Returns:
            硬件需求字典
        """
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 计算模型大小
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2  # MB
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size,
            'gpu_memory_required': f"{model_size * 2:.1f}MB",  # 估计值
            'recommended_gpu': 'NVIDIA GPU with 8GB+ VRAM'  # 估计值
        }
        
    def save_comparison_results(
        self,
        result: ComparisonResult,
        output_dir: Union[str, Path]
    ):
        """
        保存比较结果
        Args:
            result: 比较结果
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为JSON
        result_dict = {
            'model_name': result.model_name,
            'sota_name': result.sota_name,
            'metrics': result.metrics,
            'relative_performance': result.relative_performance,
            'hardware_requirements': result.hardware_requirements,
            'training_time': result.training_time,
            'inference_time': result.inference_time,
            'memory_usage': result.memory_usage,
            'comparison_date': result.comparison_date
        }
        
        with open(output_dir / 'comparison_results.json', 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4)
            
        # 生成比较报告
        self._generate_comparison_report(result, output_dir)
        
    def _generate_comparison_report(
        self,
        result: ComparisonResult,
        output_dir: Path
    ):
        """
        生成比较报告
        Args:
            result: 比较结果
            output_dir: 输出目录
        """
        sota_info = self.sota_models[result.sota_name]
        
        report = f"""# 模型性能比较报告

## 基本信息
- 比较日期: {result.comparison_date}
- 当前模型: {result.model_name}
- SOTA模型: {result.sota_name}
- SOTA论文: {sota_info.paper_title}
- SOTA作者: {', '.join(sota_info.authors)}
- SOTA会议: {sota_info.conference} ({sota_info.year})

## 性能指标比较

### 准确率
- 当前模型: {result.metrics['accuracy']:.4f}
- SOTA模型: {sota_info.metrics['accuracy']:.4f}
- 相对性能: {result.relative_performance['accuracy']:.2f}x

### 推理时间
- 当前模型: {result.inference_time:.2f}ms
- SOTA模型: {sota_info.metrics.get('inference_time', 'N/A')}ms
- 相对性能: {result.relative_performance.get('inference_time', 'N/A')}x

### 内存使用
- 当前模型: {result.memory_usage:.1f}MB
- SOTA模型: {sota_info.metrics.get('memory_usage', 'N/A')}MB
- 相对性能: {result.relative_performance.get('memory_usage', 'N/A')}x

## 硬件需求
- 总参数量: {result.hardware_requirements['total_params']:,}
- 可训练参数量: {result.hardware_requirements['trainable_params']:,}
- 模型大小: {result.hardware_requirements['model_size_mb']:.1f}MB
- GPU内存需求: {result.hardware_requirements['gpu_memory_required']}
- 推荐GPU: {result.hardware_requirements['recommended_gpu']}

## 其他指标
"""
        # 添加其他指标
        for metric, value in result.metrics.items():
            if metric not in ['accuracy', 'inference_time', 'memory_usage']:
                report += f"- {metric}: {value:.4f}\n"
                
        # 保存报告
        with open(output_dir / 'comparison_report.md', 'w', encoding='utf-8') as f:
            f.write(report) 