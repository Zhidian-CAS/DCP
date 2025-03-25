import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
import psutil
import GPUtil
import time

class SegmentationOptimizer:
    """分割模型性能优化建议器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化优化建议器
        Args:
            model: 待优化的模型
            device: 运行设备
        """
        self.model = model.to(device)
        self.device = device
        self.logger = logging.getLogger(__name__)
        
    def analyze_model_size(self) -> Dict[str, float]:
        """
        分析模型大小
        Returns:
            包含模型大小信息的字典
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_size = total_params * 4 / (1024 * 1024)  # 转换为MB
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size
        }
        
    def analyze_inference_time(
        self,
        input_size: Tuple[int, int] = (512, 512),
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """
        分析推理时间
        Args:
            input_size: 输入图像大小
            num_iterations: 迭代次数
            warmup: 预热次数
        Returns:
            包含推理时间统计信息的字典
        """
        # 创建输入数据
        dummy_input = torch.randn(1, 3, *input_size, device=self.device)
        
        # 预热
        self.model.eval()
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)
                
        # 测量推理时间
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                _ = self.model(dummy_input)
                torch.cuda.synchronize()  # 确保GPU操作完成
                end_time = time.time()
                times.append(end_time - start_time)
                
        # 计算统计信息
        times = np.array(times)
        return {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'p95_time': float(np.percentile(times, 95)),
            'p99_time': float(np.percentile(times, 99))
        }
        
    def analyze_memory_usage(self) -> Dict[str, float]:
        """
        分析内存使用情况
        Returns:
            包含内存使用统计信息的字典
        """
        # CPU内存使用
        cpu_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # 转换为MB
        
        # GPU内存使用
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # 转换为MB
            
        return {
            'cpu_memory_mb': cpu_memory,
            'gpu_memory_mb': gpu_memory
        }
        
    def analyze_gpu_utilization(self) -> Dict[str, float]:
        """
        分析GPU利用率
        Returns:
            包含GPU利用率统计信息的字典
        """
        if not torch.cuda.is_available():
            return {}
            
        gpus = GPUtil.getGPUs()
        if not gpus:
            return {}
            
        gpu = gpus[0]
        return {
            'gpu_load': gpu.load,
            'gpu_memory_util': gpu.memoryUtil,
            'gpu_temperature': gpu.temperature
        }
        
    def analyze_layer_time(
        self,
        input_size: Tuple[int, int] = (512, 512),
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        分析每层推理时间
        Args:
            input_size: 输入图像大小
            num_iterations: 迭代次数
        Returns:
            包含每层推理时间的字典
        """
        # 创建输入数据
        dummy_input = torch.randn(1, 3, *input_size, device=self.device)
        
        # 使用PyTorch Profiler
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True
        ) as prof:
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = self.model(dummy_input)
                    
        # 获取每层时间
        layer_times = {}
        for event in prof.key_averages().table(sort_by="cuda_time_total", row_limit=10):
            if event.key.startswith("aten::"):
                layer_times[event.key] = event.cuda_time_total / num_iterations
                
        return layer_times
        
    def generate_optimization_suggestions(
        self,
        metrics: Dict[str, Union[float, Dict[str, float]]]
    ) -> List[str]:
        """
        生成优化建议
        Args:
            metrics: 性能指标
        Returns:
            优化建议列表
        """
        suggestions = []
        
        # 模型大小优化建议
        model_size = metrics.get('model_size', {})
        if model_size.get('model_size_mb', 0) > 100:
            suggestions.append("模型大小超过100MB，建议进行模型压缩或量化")
            
        # 推理时间优化建议
        inference_time = metrics.get('inference_time', {})
        if inference_time.get('mean_time', 0) > 0.1:  # 100ms
            suggestions.append("推理时间过长，建议进行模型优化或使用更轻量级的架构")
            
        # 内存使用优化建议
        memory_usage = metrics.get('memory_usage', {})
        if memory_usage.get('gpu_memory_mb', 0) > 2000:  # 2GB
            suggestions.append("GPU内存使用过高，建议进行内存优化或使用梯度检查点")
            
        # GPU利用率优化建议
        gpu_util = metrics.get('gpu_utilization', {})
        if gpu_util.get('gpu_load', 0) < 0.5:
            suggestions.append("GPU利用率较低，建议增加批量大小或使用混合精度训练")
            
        # 层时间分析建议
        layer_times = metrics.get('layer_time', {})
        slow_layers = [layer for layer, time in layer_times.items() if time > 10]  # 10ms
        if slow_layers:
            suggestions.append(f"以下层推理时间过长，建议优化：{', '.join(slow_layers)}")
            
        return suggestions
        
    def run_complete_analysis(
        self,
        input_size: Tuple[int, int] = (512, 512),
        num_iterations: int = 100,
        warmup: int = 10,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        运行完整的性能分析
        Args:
            input_size: 输入图像大小
            num_iterations: 迭代次数
            warmup: 预热次数
            output_dir: 输出目录
        Returns:
            包含所有分析结果的字典
        """
        # 收集所有指标
        metrics = {
            'model_size': self.analyze_model_size(),
            'inference_time': self.analyze_inference_time(
                input_size=input_size,
                num_iterations=num_iterations,
                warmup=warmup
            ),
            'memory_usage': self.analyze_memory_usage(),
            'gpu_utilization': self.analyze_gpu_utilization(),
            'layer_time': self.analyze_layer_time(
                input_size=input_size,
                num_iterations=num_iterations
            )
        }
        
        # 生成优化建议
        metrics['suggestions'] = self.generate_optimization_suggestions(metrics)
        
        # 保存结果
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存指标到JSON
            import json
            with open(output_dir / 'optimization_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
                
            # 创建优化报告
            self.create_optimization_report(metrics, output_dir)
            
        return metrics
        
    def create_optimization_report(
        self,
        metrics: Dict,
        output_dir: Path
    ) -> None:
        """
        创建优化报告
        Args:
            metrics: 性能指标
            output_dir: 输出目录
        """
        report_path = output_dir / 'optimization_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("性能优化报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 模型大小信息
            f.write("模型大小信息\n")
            f.write("-" * 20 + "\n")
            f.write(f"总参数量: {metrics['model_size']['total_params']:,}\n")
            f.write(f"可训练参数量: {metrics['model_size']['trainable_params']:,}\n")
            f.write(f"模型大小: {metrics['model_size']['model_size_mb']:.2f} MB\n\n")
            
            # 推理时间信息
            f.write("推理时间信息\n")
            f.write("-" * 20 + "\n")
            f.write(f"平均推理时间: {metrics['inference_time']['mean_time']*1000:.2f} ms\n")
            f.write(f"标准差: {metrics['inference_time']['std_time']*1000:.2f} ms\n")
            f.write(f"最小推理时间: {metrics['inference_time']['min_time']*1000:.2f} ms\n")
            f.write(f"最大推理时间: {metrics['inference_time']['max_time']*1000:.2f} ms\n")
            f.write(f"95分位推理时间: {metrics['inference_time']['p95_time']*1000:.2f} ms\n")
            f.write(f"99分位推理时间: {metrics['inference_time']['p99_time']*1000:.2f} ms\n\n")
            
            # 内存使用信息
            f.write("内存使用信息\n")
            f.write("-" * 20 + "\n")
            f.write(f"CPU内存使用: {metrics['memory_usage']['cpu_memory_mb']:.2f} MB\n")
            f.write(f"GPU内存使用: {metrics['memory_usage']['gpu_memory_mb']:.2f} MB\n\n")
            
            # GPU利用率信息
            if metrics['gpu_utilization']:
                f.write("GPU利用率信息\n")
                f.write("-" * 20 + "\n")
                f.write(f"GPU负载: {metrics['gpu_utilization']['gpu_load']*100:.2f}%\n")
                f.write(f"GPU内存利用率: {metrics['gpu_utilization']['gpu_memory_util']*100:.2f}%\n")
                f.write(f"GPU温度: {metrics['gpu_utilization']['gpu_temperature']:.2f}°C\n\n")
                
            # 层时间分析
            f.write("层时间分析\n")
            f.write("-" * 20 + "\n")
            for layer, time in metrics['layer_time'].items():
                f.write(f"{layer}: {time:.2f} ms\n")
            f.write("\n")
            
            # 优化建议
            f.write("优化建议\n")
            f.write("-" * 20 + "\n")
            for suggestion in metrics['suggestions']:
                f.write(f"- {suggestion}\n")
                
        self.logger.info(f"优化报告已保存至: {report_path}") 