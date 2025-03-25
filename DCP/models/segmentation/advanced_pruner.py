import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class AdvancedPruner:
    """高级剪枝器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.logger = logging.getLogger(__name__)
        
    def random_pruning(
        self,
        amount: float,
        seed: Optional[int] = None
    ) -> None:
        """
        随机剪枝
        Args:
            amount: 剪枝比例
            seed: 随机种子
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                mask = torch.rand_like(module.weight) > amount
                module.weight.data *= mask
                
    def gradient_based_pruning(
        self,
        amount: float,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        num_batches: int = 10
    ) -> None:
        """
        基于梯度的剪枝
        Args:
            amount: 剪枝比例
            dataloader: 数据加载器
            criterion: 损失函数
            num_batches: 批次数
        """
        # 收集梯度
        gradients = {}
        self.model.train()
        
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_batches:
                break
                
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # 累积梯度
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if name not in gradients:
                        gradients[name] = torch.zeros_like(module.weight)
                    gradients[name] += torch.abs(module.weight.grad)
                    
        # 应用剪枝
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if name in gradients:
                    threshold = torch.quantile(gradients[name], amount)
                    mask = gradients[name] > threshold
                    module.weight.data *= mask
                    
    def weight_norm_pruning(
        self,
        amount: float,
        norm_type: int = 2
    ) -> None:
        """
        基于权重范数的剪枝
        Args:
            amount: 剪枝比例
            norm_type: 范数类型
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight_norm = torch.norm(module.weight.data.view(module.weight.size(0), -1), p=norm_type, dim=1)
                num_channels = weight_norm.size(0)
                num_to_keep = int(num_channels * (1 - amount))
                _, indices = torch.topk(weight_norm, num_to_keep)
                mask = torch.zeros_like(module.weight.data)
                mask[indices] = 1
                module.weight.data *= mask
                
    def taylor_pruning(
        self,
        amount: float,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        num_batches: int = 10
    ) -> None:
        """
        基于Taylor展开的剪枝
        Args:
            amount: 剪枝比例
            dataloader: 数据加载器
            criterion: 损失函数
            num_batches: 批次数
        """
        # 计算一阶Taylor系数
        taylor_scores = {}
        self.model.train()
        
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_batches:
                break
                
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # 计算Taylor分数
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if name not in taylor_scores:
                        taylor_scores[name] = torch.zeros_like(module.weight)
                    taylor_scores[name] += module.weight.data * module.weight.grad
                    
        # 应用剪枝
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if name in taylor_scores:
                    threshold = torch.quantile(torch.abs(taylor_scores[name]), amount)
                    mask = torch.abs(taylor_scores[name]) > threshold
                    module.weight.data *= mask
                    
    def iterative_pruning(
        self,
        target_amount: float,
        steps: int,
        method: str,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        **kwargs
    ) -> List[Dict]:
        """
        迭代式剪枝
        Args:
            target_amount: 目标剪枝比例
            steps: 迭代步数
            method: 剪枝方法
            dataloader: 数据加载器
            criterion: 损失函数
            **kwargs: 其他参数
        Returns:
            每步的剪枝结果
        """
        results = []
        step_amount = target_amount / steps
        
        for step in range(steps):
            # 保存当前状态
            current_state = self._save_model_state()
            
            # 应用剪枝
            if method == 'random':
                self.random_pruning(step_amount, **kwargs)
            elif method == 'gradient':
                self.gradient_based_pruning(step_amount, dataloader, criterion, **kwargs)
            elif method == 'weight_norm':
                self.weight_norm_pruning(step_amount, **kwargs)
            elif method == 'taylor':
                self.taylor_pruning(step_amount, dataloader, criterion, **kwargs)
            else:
                raise ValueError(f"不支持的剪枝方法: {method}")
                
            # 评估性能
            if dataloader is not None and criterion is not None:
                performance = self._evaluate_model(dataloader, criterion, num_batches=5)
            else:
                performance = None
                
            # 记录结果
            sparsity = self.analyze_model_sparsity()
            results.append({
                'step': step + 1,
                'amount': (step + 1) * step_amount,
                'sparsity': sparsity,
                'performance': performance
            })
            
        return results
        
    def mixed_pruning(
        self,
        methods: List[Dict[str, Union[str, float]]],
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        criterion: Optional[nn.Module] = None
    ) -> Dict:
        """
        混合剪枝
        Args:
            methods: 剪枝方法列表，每个方法包含名称和比例
            dataloader: 数据加载器
            criterion: 损失函数
        Returns:
            剪枝结果
        """
        original_state = self._save_model_state()
        results = {}
        
        for method_config in methods:
            method = method_config['name']
            amount = method_config['amount']
            
            # 应用剪枝
            if method == 'random':
                self.random_pruning(amount)
            elif method == 'gradient':
                self.gradient_based_pruning(amount, dataloader, criterion)
            elif method == 'weight_norm':
                self.weight_norm_pruning(amount)
            elif method == 'taylor':
                self.taylor_pruning(amount, dataloader, criterion)
            else:
                raise ValueError(f"不支持的剪枝方法: {method}")
                
            # 记录结果
            results[method] = self.analyze_model_sparsity()
            
        return results
        
    def analyze_model_sparsity(self) -> Dict[str, float]:
        """分析模型稀疏度"""
        total_params = 0
        zero_params = 0
        layer_sparsity = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight'):
                    params = module.weight.numel()
                    zeros = torch.sum(module.weight == 0).item()
                    total_params += params
                    zero_params += zeros
                    layer_sparsity[name] = zeros / params
                    
        overall_sparsity = zero_params / total_params if total_params > 0 else 0
        
        return {
            'total_params': total_params,
            'zero_params': zero_params,
            'overall_sparsity': overall_sparsity,
            'layer_sparsity': layer_sparsity
        }
        
    def _evaluate_model(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        num_batches: int = 10
    ) -> float:
        """评估模型性能"""
        self.model.eval()
        total_loss = 0
        num_samples = 0
        
        with torch.no_grad():
            for i, (images, masks) in enumerate(dataloader):
                if i >= num_batches:
                    break
                    
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                
                total_loss += loss.item() * images.size(0)
                num_samples += images.size(0)
                
        return total_loss / num_samples
        
    def _save_model_state(self) -> Dict[str, torch.Tensor]:
        """保存模型状态"""
        return {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }
        
    def _load_model_state(self, state: Dict[str, torch.Tensor]) -> None:
        """加载模型状态"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in state:
                    param.copy_(state[name])
                    
    def save_pruned_model(
        self,
        save_path: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        保存剪枝后的模型
        Args:
            save_path: 保存路径
            metadata: 元数据
        """
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'sparsity': self.analyze_model_sparsity()
        }
        
        if metadata:
            save_dict['metadata'] = metadata
            
        torch.save(save_dict, save_path)
        self.logger.info(f"剪枝后的模型已保存至: {save_path}")
        
    def create_pruning_report(
        self,
        results: Dict,
        output_dir: Path,
        include_layer_details: bool = True
    ) -> None:
        """
        创建详细的剪枝报告
        Args:
            results: 剪枝结果
            output_dir: 输出目录
            include_layer_details: 是否包含层级详细信息
        """
        report_path = output_dir / 'pruning_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("高级模型剪枝报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 总体信息
            f.write("总体信息\n")
            f.write("-" * 20 + "\n")
            f.write(f"总参数量: {results['total_params']:,}\n")
            f.write(f"零参数量: {results['zero_params']:,}\n")
            f.write(f"整体稀疏度: {results['overall_sparsity']:.4f}\n\n")
            
            # 层级详细信息
            if include_layer_details and 'layer_sparsity' in results:
                f.write("层级稀疏度详情\n")
                f.write("-" * 20 + "\n")
                for layer_name, sparsity in results['layer_sparsity'].items():
                    f.write(f"{layer_name}: {sparsity:.4f}\n")
                f.write("\n")
                
            # 性能指标
            if 'performance' in results:
                f.write("性能指标\n")
                f.write("-" * 20 + "\n")
                f.write(f"评估损失: {results['performance']:.4f}\n\n")
                
            # 迭代信息
            if 'steps' in results:
                f.write("迭代剪枝信息\n")
                f.write("-" * 20 + "\n")
                for step in results['steps']:
                    f.write(f"步骤 {step['step']}/{results['total_steps']}:\n")
                    f.write(f"  剪枝比例: {step['amount']:.4f}\n")
                    f.write(f"  稀疏度: {step['sparsity']['overall_sparsity']:.4f}\n")
                    if 'performance' in step:
                        f.write(f"  性能: {step['performance']:.4f}\n")
                    f.write("\n")
                    
        self.logger.info(f"剪枝报告已保存至: {report_path}") 