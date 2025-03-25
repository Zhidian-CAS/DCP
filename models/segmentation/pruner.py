import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime

class SegmentationPruner:
    """分割模型剪枝器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化剪枝器
        Args:
            model: 待剪枝的模型
            device: 运行设备
        """
        self.model = model.to(device)
        self.device = device
        self.logger = logging.getLogger(__name__)
        
    def analyze_model_sparsity(self) -> Dict[str, float]:
        """
        分析模型稀疏度
        Returns:
            包含稀疏度信息的字典
        """
        total_params = 0
        zero_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight'):
                    total_params += module.weight.numel()
                    zero_params += torch.sum(module.weight == 0).item()
                    
        sparsity = zero_params / total_params if total_params > 0 else 0
        
        return {
            'total_params': total_params,
            'zero_params': zero_params,
            'sparsity': sparsity
        }
        
    def l1_structured_pruning(
        self,
        amount: float,
        dim: int = 0
    ) -> None:
        """
        L1结构化剪枝
        Args:
            amount: 剪枝比例
            dim: 剪枝维度
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')
                
    def magnitude_pruning(
        self,
        amount: float,
        global_pruning: bool = False
    ) -> None:
        """
        基于幅度的剪枝
        Args:
            amount: 剪枝比例
            global_pruning: 是否使用全局剪枝
        """
        if global_pruning:
            # 收集所有权重
            weights = []
            for module in self.model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    weights.append(module.weight.data.cpu().numpy().flatten())
            weights = np.concatenate(weights)
            
            # 计算全局阈值
            threshold = np.percentile(np.abs(weights), amount * 100)
            
            # 应用剪枝
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    mask = torch.abs(module.weight.data) > threshold
                    module.weight.data *= mask
        else:
            # 逐层剪枝
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    prune.l1_unstructured(module, name='weight', amount=amount)
                    prune.remove(module, 'weight')
                    
    def channel_pruning(
        self,
        amount: float,
        criterion: str = 'l1'
    ) -> None:
        """
        通道剪枝
        Args:
            amount: 剪枝比例
            criterion: 剪枝准则（l1/l2）
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # 计算通道重要性
                if criterion == 'l1':
                    importance = torch.sum(torch.abs(module.weight.data), dim=(2, 3))
                else:  # l2
                    importance = torch.sum(module.weight.data ** 2, dim=(2, 3))
                    
                # 选择要保留的通道
                num_channels = module.out_channels
                num_to_keep = int(num_channels * (1 - amount))
                _, indices = torch.topk(importance, num_to_keep)
                
                # 应用剪枝
                module.weight.data = module.weight.data[indices]
                if module.bias is not None:
                    module.bias.data = module.bias.data[indices]
                    
    def sensitivity_analysis(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        num_batches: int = 10
    ) -> Dict[str, Dict[float, float]]:
        """
        敏感度分析
        Args:
            dataloader: 数据加载器
            criterion: 损失函数
            num_batches: 批次数
        Returns:
            包含敏感度分析结果的字典
        """
        results = {}
        
        # 记录原始性能
        original_loss = self._evaluate_model(dataloader, criterion, num_batches)
        
        # 对每种剪枝方法进行分析
        methods = ['magnitude', 'l1_structured', 'channel']
        amounts = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for method in methods:
            results[method] = {}
            for amount in amounts:
                # 保存原始权重
                original_state = self._save_model_state()
                
                # 应用剪枝
                if method == 'magnitude':
                    self.magnitude_pruning(amount)
                elif method == 'l1_structured':
                    self.l1_structured_pruning(amount)
                else:  # channel
                    self.channel_pruning(amount)
                    
                # 评估性能
                pruned_loss = self._evaluate_model(dataloader, criterion, num_batches)
                
                # 计算性能损失
                loss_increase = (pruned_loss - original_loss) / original_loss
                results[method][amount] = loss_increase
                
                # 恢复原始权重
                self._load_model_state(original_state)
                
        return results
        
    def _evaluate_model(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        num_batches: int
    ) -> float:
        """
        评估模型性能
        Args:
            dataloader: 数据加载器
            criterion: 损失函数
            num_batches: 批次数
        Returns:
            平均损失
        """
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
        """
        保存模型状态
        Returns:
            模型状态字典
        """
        state = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                state[name] = {
                    'weight': module.weight.data.clone(),
                    'bias': module.bias.data.clone() if module.bias is not None else None
                }
        return state
        
    def _load_model_state(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """
        加载模型状态
        Args:
            state: 模型状态字典
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if name in state:
                    module.weight.data = state[name]['weight']
                    if module.bias is not None and state[name]['bias'] is not None:
                        module.bias.data = state[name]['bias']
                        
    def remove_pruning(self) -> None:
        """移除所有剪枝掩码"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight_mask'):
                    prune.remove(module, 'weight')
                    
    def save_pruned_model(
        self,
        save_path: str,
        remove_masks: bool = True
    ) -> None:
        """
        保存剪枝后的模型
        Args:
            save_path: 保存路径
            remove_masks: 是否移除剪枝掩码
        """
        if remove_masks:
            self.remove_pruning()
            
        torch.save(self.model, save_path)
        self.logger.info(f"剪枝后的模型已保存至: {save_path}")
        
    def run_complete_pruning(
        self,
        method: str,
        amount: float,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        analyze_sensitivity: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        运行完整的剪枝过程
        Args:
            method: 剪枝方法
            amount: 剪枝比例
            dataloader: 数据加载器
            criterion: 损失函数
            analyze_sensitivity: 是否进行敏感度分析
            output_dir: 输出目录
        Returns:
            包含剪枝结果的字典
        """
        # 记录原始模型信息
        original_sparsity = self.analyze_model_sparsity()
        
        # 应用剪枝
        if method == 'magnitude':
            self.magnitude_pruning(amount)
        elif method == 'l1_structured':
            self.l1_structured_pruning(amount)
        elif method == 'channel':
            self.channel_pruning(amount)
        else:
            raise ValueError(f"不支持的剪枝方法: {method}")
            
        # 记录剪枝后的模型信息
        pruned_sparsity = self.analyze_model_sparsity()
        
        # 进行敏感度分析
        sensitivity_results = None
        if analyze_sensitivity and dataloader is not None and criterion is not None:
            sensitivity_results = self.sensitivity_analysis(dataloader, criterion)
            
        # 收集结果
        results = {
            'method': method,
            'amount': amount,
            'original_sparsity': original_sparsity,
            'pruned_sparsity': pruned_sparsity,
            'sensitivity_results': sensitivity_results
        }
        
        # 保存结果
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存指标到JSON
            with open(output_dir / 'pruning_results.json', 'w') as f:
                json.dump(results, f, indent=4)
                
            # 创建剪枝报告
            self.create_pruning_report(results, output_dir)
            
        return results
        
    def create_pruning_report(
        self,
        results: Dict,
        output_dir: Path
    ) -> None:
        """
        创建剪枝报告
        Args:
            results: 剪枝结果
            output_dir: 输出目录
        """
        report_path = output_dir / 'pruning_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("模型剪枝报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 剪枝方法信息
            f.write("剪枝方法信息\n")
            f.write("-" * 20 + "\n")
            f.write(f"剪枝方法: {results['method']}\n")
            f.write(f"剪枝比例: {results['amount']:.2f}\n\n")
            
            # 原始模型信息
            f.write("原始模型信息\n")
            f.write("-" * 20 + "\n")
            f.write(f"总参数量: {results['original_sparsity']['total_params']:,}\n")
            f.write(f"零参数量: {results['original_sparsity']['zero_params']:,}\n")
            f.write(f"稀疏度: {results['original_sparsity']['sparsity']:.4f}\n\n")
            
            # 剪枝后模型信息
            f.write("剪枝后模型信息\n")
            f.write("-" * 20 + "\n")
            f.write(f"总参数量: {results['pruned_sparsity']['total_params']:,}\n")
            f.write(f"零参数量: {results['pruned_sparsity']['zero_params']:,}\n")
            f.write(f"稀疏度: {results['pruned_sparsity']['sparsity']:.4f}\n")
            f.write(f"压缩率: {(1 - results['pruned_sparsity']['sparsity']) / (1 - results['original_sparsity']['sparsity']):.2f}\n\n")
            
            # 敏感度分析结果
            if results['sensitivity_results']:
                f.write("敏感度分析结果\n")
                f.write("-" * 20 + "\n")
                for method, amounts in results['sensitivity_results'].items():
                    f.write(f"\n{method}方法:\n")
                    for amount, loss_increase in amounts.items():
                        f.write(f"剪枝比例 {amount:.2f}: 性能损失 {loss_increase:.4f}\n")
                        
        self.logger.info(f"剪枝报告已保存至: {report_path}") 