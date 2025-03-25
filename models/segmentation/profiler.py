import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import psutil
import GPUtil
from pathlib import Path
import json
import logging
from tqdm import tqdm

class SegmentationProfiler:
    """分割模型性能分析器"""
    
    def __init__(
        self,
        model: nn.Module,
        input_size: Tuple[int, int] = (512, 512),
        batch_size: int = 1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化性能分析器
        Args:
            model: 待分析的模型
            input_size: 输入大小
            batch_size: 批次大小
            device: 运行设备
        """
        self.model = model.to(device)
        self.input_size = input_size
        self.batch_size = batch_size
        self.device = device
        self.logger = logging.getLogger(__name__)
        
    def profile_model_size(self) -> Dict[str, float]:
        """
        分析模型大小
        Returns:
            模型大小信息（MB）
        """
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # 计算模型大小
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': size_all_mb
        }
        
    def profile_inference_time(
        self,
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """
        分析推理时间
        Args:
            num_iterations: 迭代次数
            warmup: 预热次数
        Returns:
            推理时间信息（ms）
        """
        self.model.eval()
        times = []
        
        # 创建随机输入
        dummy_input = torch.randn(
            self.batch_size, 3, *self.input_size,
            device=self.device
        )
        
        # 预热
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)
        
        # 测量推理时间
        with torch.no_grad():
            for _ in tqdm(range(num_iterations), desc='Profiling inference time'):
                start_time = time.perf_counter()
                _ = self.model(dummy_input)
                torch.cuda.synchronize() if self.device == 'cuda' else None
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        times = np.array(times)
        return {
            'mean_inference_time': float(np.mean(times)),
            'std_inference_time': float(np.std(times)),
            'min_inference_time': float(np.min(times)),
            'max_inference_time': float(np.max(times)),
            'p90_inference_time': float(np.percentile(times, 90)),
            'p95_inference_time': float(np.percentile(times, 95)),
            'p99_inference_time': float(np.percentile(times, 99))
        }
        
    def profile_memory_usage(
        self,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        分析内存使用
        Args:
            num_iterations: 迭代次数
        Returns:
            内存使用信息（MB）
        """
        self.model.eval()
        memory_stats = []
        
        # 创建随机输入
        dummy_input = torch.randn(
            self.batch_size, 3, *self.input_size,
            device=self.device
        )
        
        # 测量内存使用
        with torch.no_grad():
            for _ in tqdm(range(num_iterations), desc='Profiling memory usage'):
                if self.device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                    _ = self.model(dummy_input)
                    torch.cuda.synchronize()
                    
                    memory_stats.append({
                        'allocated': torch.cuda.memory_allocated() / 1024**2,
                        'reserved': torch.cuda.memory_reserved() / 1024**2,
                        'peak_allocated': torch.cuda.max_memory_allocated() / 1024**2,
                        'peak_reserved': torch.cuda.max_memory_reserved() / 1024**2
                    })
                else:
                    process = psutil.Process()
                    _ = self.model(dummy_input)
                    memory_stats.append({
                        'ram_usage': process.memory_info().rss / 1024**2
                    })
        
        # 计算平均值
        avg_stats = {}
        for key in memory_stats[0].keys():
            values = [stats[key] for stats in memory_stats]
            avg_stats[key] = float(np.mean(values))
            
        return avg_stats
        
    def profile_gpu_utilization(
        self,
        num_iterations: int = 100
    ) -> Optional[Dict[str, float]]:
        """
        分析GPU使用率
        Args:
            num_iterations: 迭代次数
        Returns:
            GPU使用信息（%）
        """
        if self.device != 'cuda':
            return None
            
        self.model.eval()
        gpu_stats = []
        
        # 创建随机输入
        dummy_input = torch.randn(
            self.batch_size, 3, *self.input_size,
            device=self.device
        )
        
        # 测量GPU使用率
        with torch.no_grad():
            for _ in tqdm(range(num_iterations), desc='Profiling GPU utilization'):
                _ = self.model(dummy_input)
                torch.cuda.synchronize()
                
                gpu = GPUtil.getGPUs()[0]
                gpu_stats.append({
                    'gpu_utilization': gpu.load * 100,
                    'gpu_memory_utilization': gpu.memoryUtil * 100
                })
                
        # 计算平均值
        avg_stats = {}
        for key in gpu_stats[0].keys():
            values = [stats[key] for stats in gpu_stats]
            avg_stats[key] = float(np.mean(values))
            
        return avg_stats
        
    def profile_layer_time(self) -> Dict[str, float]:
        """
        分析各层推理时间
        Returns:
            各层推理时间信息（ms）
        """
        self.model.eval()
        
        # 创建随机输入
        dummy_input = torch.randn(
            self.batch_size, 3, *self.input_size,
            device=self.device
        )
        
        # 使用PyTorch Profiler分析各层时间
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        ) as prof:
            with record_function("model_inference"):
                _ = self.model(dummy_input)
                
        # 提取各层时间信息
        layer_stats = {}
        for event in prof.key_averages():
            layer_stats[event.key] = {
                'cpu_time_ms': event.cpu_time_total / 1000,  # 转换为毫秒
                'cuda_time_ms': event.cuda_time_total / 1000 if event.cuda_time_total else 0,
                'input_shape': event.input_shapes,
                'output_shape': event.output_shapes
            }
            
        return layer_stats
        
    def run_complete_profile(
        self,
        num_iterations: int = 100,
        warmup: int = 10,
        save_path: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        运行完整的性能分析
        Args:
            num_iterations: 迭代次数
            warmup: 预热次数
            save_path: 结果保存路径
        Returns:
            完整的性能分析结果
        """
        self.logger.info("开始性能分析...")
        
        # 收集所有性能指标
        profile_results = {
            'model_info': {
                'batch_size': self.batch_size,
                'input_size': self.input_size,
                'device': self.device
            },
            'model_size': self.profile_model_size(),
            'inference_time': self.profile_inference_time(num_iterations, warmup),
            'memory_usage': self.profile_memory_usage(num_iterations),
            'layer_time': self.profile_layer_time()
        }
        
        # 如果使用GPU，添加GPU使用率信息
        if self.device == 'cuda':
            profile_results['gpu_utilization'] = self.profile_gpu_utilization(num_iterations)
            
        # 保存结果
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(profile_results, f, indent=4)
            self.logger.info(f"性能分析结果已保存至: {save_path}")
            
        return profile_results 