import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import Config
from typing import Dict, List, Optional, Tuple, Union
import math
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

class AttentionModule(nn.Module):
    """注意力模块"""
    def __init__(self, in_channels: int, ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out

class EncoderBlock(nn.Module):
    """编码器块"""
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 use_attention: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)
        
        self.use_attention = use_attention
        if use_attention:
            self.channel_attention = AttentionModule(out_channels)
            self.spatial_attention = SpatialAttention()
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # 注意力机制
        if self.use_attention:
            x = self.channel_attention(x)
            x = self.spatial_attention(x)
            
        # 最大池化
        size = x.size()
        x, indices = self.maxpool(x)
        
        return x, indices, size

class DecoderBlock(nn.Module):
    """解码器块"""
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 use_attention: bool = True):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.use_attention = use_attention
        if use_attention:
            self.channel_attention = AttentionModule(out_channels)
            self.spatial_attention = SpatialAttention()
            
    def forward(self, x: torch.Tensor, indices: torch.Tensor, size: torch.Size) -> torch.Tensor:
        # 反池化
        x = self.unpool(x, indices, size)
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # 注意力机制
        if self.use_attention:
            x = self.channel_attention(x)
            x = self.spatial_attention(x)
            
        return x

class SegNet(nn.Module):
    """SegNet语义分割网络"""
    def __init__(self,
                 num_classes: int,
                 in_channels: int = 3,
                 base_channels: int = 64,
                 num_encoders: int = 5,
                 use_attention: bool = True):
        super().__init__()
        
        # 网络参数
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_encoders = num_encoders
        
        # 编码器
        self.encoders = nn.ModuleList()
        curr_channels = in_channels
        for i in range(num_encoders):
            out_channels = base_channels * (2 ** i)
            self.encoders.append(
                EncoderBlock(curr_channels, out_channels, use_attention)
            )
            curr_channels = out_channels
            
        # 解码器
        self.decoders = nn.ModuleList()
        for i in range(num_encoders-1, -1, -1):
            in_channels = base_channels * (2 ** i)
            out_channels = base_channels * (2 ** max(0, i-1))
            if i == 0:
                out_channels = base_channels
            self.decoders.append(
                DecoderBlock(in_channels, out_channels, use_attention)
            )
            
        # 输出层
        self.final_conv = nn.Conv2d(base_channels, num_classes, 1)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保存编码器的输出
        encoder_features = []
        indices_list = []
        sizes_list = []
        
        # 编码器前向传播
        for encoder in self.encoders:
            x, indices, size = encoder(x)
            encoder_features.append(x)
            indices_list.append(indices)
            sizes_list.append(size)
            
        # 解码器前向传播
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, indices_list[-i-1], sizes_list[-i-1])
            
        # 输出层
        x = self.final_conv(x)
        
        return x
        
    def get_loss(self, 
                 pred: torch.Tensor, 
                 target: torch.Tensor,
                 weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算损失
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 真实标签 [B, H, W]
            weights: 类别权重 [C]
        Returns:
            loss: 损失值
        """
        # 交叉熵损失
        ce_loss = F.cross_entropy(pred, target, weight=weights, reduction='mean')
        
        # Dice损失
        pred_softmax = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2)
        dice_loss = 1 - (2 * (pred_softmax * target_onehot).sum() + 1e-5) / \
                   (pred_softmax.sum() + target_onehot.sum() + 1e-5)
        
        # 边界损失
        pred_softmax_log = F.log_softmax(pred, dim=1)
        boundary_loss = F.kl_div(
            pred_softmax_log,
            target_onehot.float(),
            reduction='batchmean'
        )
        
        # 总损失
        loss = ce_loss + 0.5 * dice_loss + 0.1 * boundary_loss
        
        return loss
        
    @torch.no_grad()
    def predict(self, 
                x: torch.Tensor,
                threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        模型预测
        Args:
            x: 输入图像 [B, C, H, W]
            threshold: 置信度阈值
        Returns:
            pred_mask: 预测掩码 [B, H, W]
            pred_prob: 预测概率 [B, C, H, W]
        """
        self.eval()
        
        # 前向传播
        pred = self(x)
        
        # 计算预测概率
        pred_prob = F.softmax(pred, dim=1)
        
        # 获取预测掩码
        pred_mask = torch.argmax(pred_prob, dim=1)
        
        # 过滤低置信度预测
        max_prob = torch.max(pred_prob, dim=1)[0]
        pred_mask[max_prob < threshold] = 0
        
        return pred_mask, pred_prob
        
    def export_model(self, save_path: str):
        """
        导出模型
        Args:
            save_path: 保存路径
        """
        # 创建示例输入
        dummy_input = torch.randn(1, self.in_channels, 224, 224)
        
        # 导出模型
        torch.onnx.export(
            self,
            dummy_input,
            save_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

class ColonySegmenter:
    """菌落分割器"""
    def __init__(self,
                 num_classes: int = 2,
                 in_channels: int = 3,
                 base_channels: int = 64,
                 num_encoders: int = 5,
                 use_attention: bool = True,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0001):
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型配置
        self.model = SegNet(
            num_classes=num_classes,
            in_channels=in_channels,
            base_channels=base_channels,
            num_encoders=num_encoders,
            use_attention=use_attention
        ).to(self.device)
        
        # 优化器配置
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate * 10,
            epochs=100,
            steps_per_epoch=100,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # 训练状态
        self.epoch = 0
        self.best_loss = float('inf')
        self.best_metric = 0.0
        self.early_stopping_counter = 0
        
        # 日志配置
        self.logger = logging.getLogger(__name__)
        
    def train(self, 
              train_loader: torch.utils.data.DataLoader,
              valid_loader: torch.utils.data.DataLoader,
              num_epochs: int,
              save_dir: Optional[str] = None,
              early_stopping_patience: int = 10,
              save_freq: int = 1,
              class_weights: Optional[torch.Tensor] = None,
              resume_from: Optional[str] = None):
        """
        训练模型
        Args:
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器
            num_epochs: 训练轮数
            save_dir: 保存目录
            early_stopping_patience: 早停耐心值
            save_freq: 保存频率
            class_weights: 类别权重
            resume_from: 恢复训练的检查点路径
        """
        # 创建保存目录
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
        # 恢复训练
        if resume_from:
            self._load_checkpoint(resume_from)
            self.logger.info(f'Resumed from epoch {self.epoch}')
            
        # 训练循环
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            self.logger.info(f'Epoch {epoch+1}/{num_epochs}')
            
            # 训练一个epoch
            train_metrics = self._train_epoch(train_loader, class_weights)
            
            # 验证
            val_metrics = self._validate(valid_loader, class_weights)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录指标
            metrics = {
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }
            self.logger.info(f'Metrics: {metrics}')
            
            # 保存检查点
            if save_dir and (epoch + 1) % save_freq == 0:
                self._save_checkpoint(save_dir / f'epoch_{epoch+1}.pth', metrics)
                
            # 保存最佳模型
            current_metric = val_metrics.get('dice', -val_metrics['loss'])
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.early_stopping_counter = 0
                if save_dir:
                    self._save_checkpoint(save_dir / 'best.pth', metrics, is_best=True)
            else:
                self.early_stopping_counter += 1
                
            # 早停检查
            if self.early_stopping_counter >= early_stopping_patience:
                self.logger.info(f'Early stopping triggered after {epoch+1} epochs')
                break
                
    def _train_epoch(self, 
                     train_loader: torch.utils.data.DataLoader,
                     class_weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        训练一个epoch
        Args:
            train_loader: 训练数据加载器
            class_weights: 类别权重
        Returns:
            metrics: 训练指标
        """
        self.model.train()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        
        # 创建进度条
        pbar = tqdm(train_loader, desc='Training')
        
        for images, masks in pbar:
            # 将数据移到设备
            images = images.to(self.device)
            masks = masks.to(self.device)
            if class_weights is not None:
                class_weights = class_weights.to(self.device)
                
            # 前向传播
            outputs = self.model(images)
            loss = self.model.get_loss(outputs, masks, class_weights)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 计算指标
            total_loss += loss.item()
            pred_mask = torch.argmax(outputs, dim=1)
            total_dice += self._compute_dice(pred_mask, masks)
            total_iou += self._compute_iou(pred_mask, masks)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
        # 计算平均指标
        metrics = {
            'loss': total_loss / len(train_loader),
            'dice': total_dice / len(train_loader),
            'iou': total_iou / len(train_loader)
        }
        
        return metrics
        
    @torch.no_grad()
    def _validate(self, 
                  valid_loader: torch.utils.data.DataLoader,
                  class_weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        验证模型
        Args:
            valid_loader: 验证数据加载器
            class_weights: 类别权重
        Returns:
            metrics: 验证指标
        """
        self.model.eval()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        
        for images, masks in tqdm(valid_loader, desc='Validating'):
            # 将数据移到设备
            images = images.to(self.device)
            masks = masks.to(self.device)
            if class_weights is not None:
                class_weights = class_weights.to(self.device)
                
            # 前向传播
            outputs = self.model(images)
            loss = self.model.get_loss(outputs, masks, class_weights)
            
            # 计算指标
            total_loss += loss.item()
            pred_mask = torch.argmax(outputs, dim=1)
            total_dice += self._compute_dice(pred_mask, masks)
            total_iou += self._compute_iou(pred_mask, masks)
            
        # 计算平均指标
        metrics = {
            'loss': total_loss / len(valid_loader),
            'dice': total_dice / len(valid_loader),
            'iou': total_iou / len(valid_loader)
        }
        
        return metrics
        
    def predict(self, 
                image: Union[torch.Tensor, np.ndarray],
                threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测图像中的菌落区域
        Args:
            image: 输入图像 [H, W, C] 或 [C, H, W]
            threshold: 置信度阈值
        Returns:
            pred_mask: 预测掩码 [H, W]
            pred_prob: 预测概率 [C, H, W]
        """
        self.model.eval()
        
        with torch.no_grad():
            # 预处理图像
            if isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[-1] == 3:
                    image = torch.from_numpy(image).permute(2, 0, 1)
                elif image.ndim == 3 and image.shape[0] == 3:
                    image = torch.from_numpy(image)
                else:
                    raise ValueError("Unsupported image format")
                    
            if image.dim() == 3:
                image = image.unsqueeze(0)
                
            # 将图像移到设备
            image = image.float().to(self.device)
            if image.max() > 1:
                image = image / 255.0
                
            # 预测
            pred_mask, pred_prob = self.model.predict(image, threshold)
            
            # 转换为numpy数组
            pred_mask = pred_mask[0].cpu().numpy()
            pred_prob = pred_prob[0].cpu().numpy()
            
            return pred_mask, pred_prob
            
    def _compute_dice(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算Dice系数
        Args:
            pred: 预测掩码 [B, H, W]
            target: 真实掩码 [B, H, W]
        Returns:
            dice: Dice系数
        """
        smooth = 1e-5
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.item()
        
    def _compute_iou(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算IoU
        Args:
            pred: 预测掩码 [B, H, W]
            target: 真实掩码 [B, H, W]
        Returns:
            iou: IoU
        """
        smooth = 1e-5
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()
        
    def _save_checkpoint(self,
                        save_path: Union[str, Path],
                        metrics: Dict[str, float],
                        is_best: bool = False):
        """
        保存检查点
        Args:
            save_path: 保存路径
            metrics: 评估指标
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'metrics': metrics
        }
        
        torch.save(checkpoint, save_path)
        self.logger.info(f'Saved checkpoint to {save_path}')
        
        if is_best:
            self.logger.info(f'New best model with metric: {self.best_metric:.4f}')
            
    def _load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """
        加载检查点
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        
        self.logger.info(f'Loaded checkpoint from {checkpoint_path}') 