import logging
from typing import Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
import json
import yaml
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

@dataclass
class AblationResult:
    """消融实验结果"""
    experiment_name: str
    component_name: str
    metrics: Dict[str, float]
    relative_performance: Dict[str, float]
    experiment_date: str
    configuration: Dict

class AblationStudy:
    """消融实验管理器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化消融实验管理器
        Args:
            model: 待测试模型
            device: 运行设备
        """
        self.model = model.to(device)
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.results = []
        
    def run_component_ablation(
        self,
        component_name: str,
        disable_fn: callable,
        enable_fn: callable,
        test_dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        num_runs: int = 5,
        warmup: int = 2
    ) -> AblationResult:
        """
        运行组件消融实验
        Args:
            component_name: 组件名称
            disable_fn: 禁用组件的函数
            enable_fn: 启用组件的函数
            test_dataloader: 测试数据加载器
            criterion: 损失函数
            num_runs: 运行次数
            warmup: 预热次数
        Returns:
            消融实验结果
        """
        # 保存原始模型状态
        original_state = self._save_model_state()
        
        # 禁用组件
        disable_fn()
        disabled_metrics = self._evaluate_model(
            test_dataloader,
            criterion,
            num_runs,
            warmup
        )
        
        # 恢复原始状态
        self._load_model_state(original_state)
        
        # 启用组件
        enable_fn()
        enabled_metrics = self._evaluate_model(
            test_dataloader,
            criterion,
            num_runs,
            warmup
        )
        
        # 计算相对性能
        relative_performance = self._compute_relative_performance(
            enabled_metrics,
            disabled_metrics
        )
        
        # 创建结果
        result = AblationResult(
            experiment_name="component_ablation",
            component_name=component_name,
            metrics={
                'enabled': enabled_metrics,
                'disabled': disabled_metrics
            },
            relative_performance=relative_performance,
            experiment_date=datetime.now().isoformat(),
            configuration={
                'num_runs': num_runs,
                'warmup': warmup
            }
        )
        
        self.results.append(result)
        return result
        
    def run_feature_ablation(
        self,
        feature_names: List[str],
        test_dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        num_runs: int = 5,
        warmup: int = 2
    ) -> List[AblationResult]:
        """
        运行特征消融实验
        Args:
            feature_names: 特征名称列表
            test_dataloader: 测试数据加载器
            criterion: 损失函数
            num_runs: 运行次数
            warmup: 预热次数
        Returns:
            消融实验结果列表
        """
        results = []
        
        # 保存原始模型状态
        original_state = self._save_model_state()
        
        # 基准性能
        baseline_metrics = self._evaluate_model(
            test_dataloader,
            criterion,
            num_runs,
            warmup
        )
        
        # 对每个特征进行消融
        for feature in feature_names:
            # 禁用特征
            self._disable_feature(feature)
            
            # 评估性能
            ablated_metrics = self._evaluate_model(
                test_dataloader,
                criterion,
                num_runs,
                warmup
            )
            
            # 计算相对性能
            relative_performance = self._compute_relative_performance(
                baseline_metrics,
                ablated_metrics
            )
            
            # 创建结果
            result = AblationResult(
                experiment_name="feature_ablation",
                component_name=feature,
                metrics={
                    'baseline': baseline_metrics,
                    'ablated': ablated_metrics
                },
                relative_performance=relative_performance,
                experiment_date=datetime.now().isoformat(),
                configuration={
                    'num_runs': num_runs,
                    'warmup': warmup
                }
            )
            
            results.append(result)
            self.results.append(result)
            
            # 恢复原始状态
            self._load_model_state(original_state)
            
        return results
        
    def run_architecture_ablation(
        self,
        architecture_variants: List[Dict],
        test_dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        num_runs: int = 5,
        warmup: int = 2
    ) -> List[AblationResult]:
        """
        运行架构消融实验
        Args:
            architecture_variants: 架构变体列表
            test_dataloader: 测试数据加载器
            criterion: 损失函数
            num_runs: 运行次数
            warmup: 预热次数
        Returns:
            消融实验结果列表
        """
        results = []
        
        # 保存原始模型状态
        original_state = self._save_model_state()
        
        # 基准性能
        baseline_metrics = self._evaluate_model(
            test_dataloader,
            criterion,
            num_runs,
            warmup
        )
        
        # 测试每个架构变体
        for variant in architecture_variants:
            # 应用架构变体
            self._apply_architecture_variant(variant)
            
            # 评估性能
            variant_metrics = self._evaluate_model(
                test_dataloader,
                criterion,
                num_runs,
                warmup
            )
            
            # 计算相对性能
            relative_performance = self._compute_relative_performance(
                baseline_metrics,
                variant_metrics
            )
            
            # 创建结果
            result = AblationResult(
                experiment_name="architecture_ablation",
                component_name=variant['name'],
                metrics={
                    'baseline': baseline_metrics,
                    'variant': variant_metrics
                },
                relative_performance=relative_performance,
                experiment_date=datetime.now().isoformat(),
                configuration={
                    'num_runs': num_runs,
                    'warmup': warmup,
                    'variant_config': variant
                }
            )
            
            results.append(result)
            self.results.append(result)
            
            # 恢复原始状态
            self._load_model_state(original_state)
            
        return results
        
    def _evaluate_model(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        num_runs: int,
        warmup: int
    ) -> Dict[str, float]:
        """
        评估模型性能
        Args:
            dataloader: 数据加载器
            criterion: 损失函数
            num_runs: 运行次数
            warmup: 预热次数
        Returns:
            性能指标字典
        """
        self.model.eval()
        metrics = {}
        
        # 预热
        for _ in range(warmup):
            self._run_inference(dataloader)
            
        # 性能测试
        times = []
        memory_usage = []
        losses = []
        
        for _ in range(num_runs):
            # 记录开始时间
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            # 记录开始内存
            if self.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                
            start_time.record()
            loss = self._run_inference(dataloader, criterion)
            end_time.record()
            
            torch.cuda.synchronize()
            times.append(start_time.elapsed_time(end_time))
            losses.append(loss)
            
            # 记录内存使用
            if self.device == 'cuda':
                memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
                
        # 计算统计信息
        times = np.array(times)
        losses = np.array(losses)
        
        metrics['inference_time'] = float(np.mean(times))
        metrics['inference_time_std'] = float(np.std(times))
        metrics['memory_usage'] = float(np.mean(memory_usage)) if memory_usage else 0
        metrics['loss'] = float(np.mean(losses))
        metrics['loss_std'] = float(np.std(losses))
        
        return metrics
        
    def _run_inference(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: Optional[nn.Module] = None
    ) -> Optional[float]:
        """
        运行推理
        Args:
            dataloader: 数据加载器
            criterion: 损失函数
        Returns:
            损失值（如果提供了损失函数）
        """
        total_loss = 0
        num_samples = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                
                if criterion is not None:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item() * images.size(0)
                    num_samples += images.size(0)
                    
        return total_loss / num_samples if criterion is not None else None
        
    def _compute_relative_performance(
        self,
        baseline_metrics: Dict[str, float],
        ablated_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        计算相对性能
        Args:
            baseline_metrics: 基准性能指标
            ablated_metrics: 消融后性能指标
        Returns:
            相对性能字典
        """
        relative = {}
        for metric in baseline_metrics:
            if metric in ablated_metrics:
                if metric in ['inference_time', 'memory_usage']:
                    # 越小越好
                    relative[metric] = baseline_metrics[metric] / ablated_metrics[metric]
                else:
                    # 越大越好
                    relative[metric] = ablated_metrics[metric] / baseline_metrics[metric]
        return relative
        
    def _save_model_state(self) -> Dict:
        """保存模型状态"""
        return {
            name: param.clone() for name, param in self.model.named_parameters()
        }
        
    def _load_model_state(self, state: Dict):
        """加载模型状态"""
        for name, param in self.model.named_parameters():
            param.data.copy_(state[name])
            
    def _disable_feature(self, feature_name: str):
        """禁用特征"""
        # 实现特征禁用逻辑
        pass
        
    def _apply_architecture_variant(self, variant: Dict):
        """应用架构变体"""
        # 实现架构变体应用逻辑
        pass
        
    def save_results(self, output_dir: Union[str, Path]):
        """
        保存实验结果
        Args:
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存所有结果到JSON
        results_dict = [
            {
                'experiment_name': result.experiment_name,
                'component_name': result.component_name,
                'metrics': result.metrics,
                'relative_performance': result.relative_performance,
                'experiment_date': result.experiment_date,
                'configuration': result.configuration
            }
            for result in self.results
        ]
        
        with open(output_dir / 'ablation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=4)
            
        # 生成可视化报告
        self._generate_visualization_report(output_dir)
        
    def _generate_visualization_report(self, output_dir: Path):
        """
        生成可视化报告
        Args:
            output_dir: 输出目录
        """
        # 创建性能对比图
        plt.figure(figsize=(12, 6))
        
        # 按实验类型分组
        experiment_groups = {}
        for result in self.results:
            if result.experiment_name not in experiment_groups:
                experiment_groups[result.experiment_name] = []
            experiment_groups[result.experiment_name].append(result)
            
        # 绘制每种实验的结果
        for exp_name, results in experiment_groups.items():
            components = [r.component_name for r in results]
            performances = [r.relative_performance.get('loss', 0) for r in results]
            
            plt.bar(components, performances)
            plt.title(f"{exp_name} Results")
            plt.xlabel("Component")
            plt.ylabel("Relative Performance")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存图表
            plt.savefig(output_dir / f"{exp_name}_results.png")
            plt.close()
            
        # 生成HTML报告
        html_report = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ablation Study Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .experiment {{ margin-bottom: 30px; }}
        .metric {{ margin: 10px 0; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <h1>Ablation Study Results</h1>
    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
        
        for exp_name, results in experiment_groups.items():
            html_report += f"""
    <div class="experiment">
        <h2>{exp_name}</h2>
        <img src="{exp_name}_results.png" alt="{exp_name} Results">
        <div class="details">
"""
            
            for result in results:
                html_report += f"""
            <h3>{result.component_name}</h3>
            <div class="metric">
                <h4>Metrics:</h4>
                <ul>
"""
                for metric, value in result.metrics.items():
                    html_report += f"                    <li>{metric}: {value:.4f}</li>\n"
                html_report += """
                </ul>
            </div>
            <div class="metric">
                <h4>Relative Performance:</h4>
                <ul>
"""
                for metric, value in result.relative_performance.items():
                    html_report += f"                    <li>{metric}: {value:.4f}x</li>\n"
                html_report += """
                </ul>
            </div>
"""
            
            html_report += """
        </div>
    </div>
"""
        
        html_report += """
</body>
</html>
"""
        
        with open(output_dir / 'ablation_report.html', 'w', encoding='utf-8') as f:
            f.write(html_report) 