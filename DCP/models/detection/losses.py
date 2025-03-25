import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import math

class FocalLoss(nn.Module):
    """Focal Loss"""
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class SmoothL1Loss(nn.Module):
    """平滑L1损失"""
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(inputs, targets, beta=self.beta)

class IoULoss(nn.Module):
    """IoU损失"""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 计算IoU
        x1 = inputs[:, 0]
        y1 = inputs[:, 1]
        x2 = inputs[:, 2]
        y2 = inputs[:, 3]
        
        x1g = targets[:, 0]
        y1g = targets[:, 1]
        x2g = targets[:, 2]
        y2g = targets[:, 3]
        
        # 计算交集
        x2i = torch.min(x2, x2g)
        y2i = torch.min(y2, y2g)
        x1i = torch.max(x1, x1g)
        y1i = torch.max(y1, y1g)
        
        # 计算交集面积
        inter_area = torch.clamp(x2i - x1i, min=0) * torch.clamp(y2i - y1i, min=0)
        
        # 计算并集面积
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2g - x1g) * (y2g - y1g)
        union_area = box1_area + box2_area - inter_area
        
        # 计算IoU
        iou = inter_area / (union_area + 1e-6)
        
        # 计算IoU损失
        loss = 1 - iou
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
            
        return loss

class DetectionLoss(nn.Module):
    """目标检测损失函数"""
    def __init__(self, 
                 num_classes: int,
                 focal_loss: bool = False,
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0,
                 iou_loss: bool = False,
                 loss_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.num_classes = num_classes
        self.focal_loss = focal_loss
        self.iou_loss = iou_loss
        
        # 设置损失权重
        self.loss_weights = loss_weights or {
            "loss_classifier": 1.0,
            "loss_box_reg": 1.0,
            "loss_objectness": 1.0,
            "loss_rpn_box_reg": 1.0,
            "loss_iou": 0.5
        }
        
        # 初始化损失函数
        if focal_loss:
            self.classifier_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.classifier_loss = nn.CrossEntropyLoss()
            
        self.box_reg_loss = SmoothL1Loss()
        self.objectness_loss = nn.BCEWithLogitsLoss()
        self.rpn_box_reg_loss = SmoothL1Loss()
        
        if iou_loss:
            self.iou_loss = IoULoss()
            
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算检测损失
        Args:
            predictions: 模型预测结果
            targets: 目标标注
        Returns:
            losses: 损失字典
        """
        losses = {}
        
        # 分类损失
        if "class_logits" in predictions and "labels" in targets:
            losses["loss_classifier"] = self.classifier_loss(
                predictions["class_logits"],
                targets["labels"]
            ) * self.loss_weights["loss_classifier"]
            
        # 边界框回归损失
        if "box_regression" in predictions and "regression_targets" in targets:
            losses["loss_box_reg"] = self.box_reg_loss(
                predictions["box_regression"],
                targets["regression_targets"]
            ) * self.loss_weights["loss_box_reg"]
            
        # 目标性损失
        if "objectness" in predictions and "objectness_targets" in targets:
            losses["loss_objectness"] = self.objectness_loss(
                predictions["objectness"],
                targets["objectness_targets"]
            ) * self.loss_weights["loss_objectness"]
            
        # RPN边界框回归损失
        if "rpn_box_regression" in predictions and "rpn_regression_targets" in targets:
            losses["loss_rpn_box_reg"] = self.rpn_box_reg_loss(
                predictions["rpn_box_regression"],
                targets["rpn_regression_targets"]
            ) * self.loss_weights["loss_rpn_box_reg"]
            
        # IoU损失
        if self.iou_loss and "boxes" in predictions and "boxes" in targets:
            losses["loss_iou"] = self.iou_loss(
                predictions["boxes"],
                targets["boxes"]
            ) * self.loss_weights["loss_iou"]
            
        return losses

class DetectionLossWithAuxiliary(nn.Module):
    """带辅助任务的目标检测损失函数"""
    def __init__(self, 
                 num_classes: int,
                 focal_loss: bool = False,
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0,
                 iou_loss: bool = False,
                 loss_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.base_loss = DetectionLoss(
            num_classes=num_classes,
            focal_loss=focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            iou_loss=iou_loss,
            loss_weights=loss_weights
        )
        
        # 辅助任务损失
        self.auxiliary_losses = nn.ModuleDict({
            "segmentation": nn.CrossEntropyLoss(),
            "keypoints": nn.MSELoss(),
            "pose": nn.MSELoss()
        })
        
        # 辅助任务权重
        self.auxiliary_weights = {
            "segmentation": 0.5,
            "keypoints": 0.3,
            "pose": 0.2
        }
        
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算检测损失和辅助任务损失
        Args:
            predictions: 模型预测结果
            targets: 目标标注
        Returns:
            losses: 损失字典
        """
        # 基础检测损失
        losses = self.base_loss(predictions, targets)
        
        # 辅助任务损失
        for task, loss_fn in self.auxiliary_losses.items():
            if f"{task}_pred" in predictions and f"{task}_target" in targets:
                losses[f"loss_{task}"] = loss_fn(
                    predictions[f"{task}_pred"],
                    targets[f"{task}_target"]
                ) * self.auxiliary_weights[task]
                
        return losses 