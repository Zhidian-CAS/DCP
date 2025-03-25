import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from typing import Dict, List, Tuple, Optional
import logging

class RoIAlign(nn.Module):
    """RoI对齐层"""
    def __init__(self, output_size: Tuple[int, int], sampling_ratio: int = -1):
        super().__init__()
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        
    def forward(self, input: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        return F.roi_align(input, rois, self.output_size, self.sampling_ratio)

class RoIHead(nn.Module):
    """RoI头部网络"""
    def __init__(self, in_channels: int, num_classes: int, representation_size: int = 1024):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        
        # 分类头
        self.cls_score = nn.Linear(representation_size, num_classes)
        # 边界框回归头
        self.bbox_pred = nn.Linear(representation_size, num_classes * 4)
        
        # 初始化权重
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        
        return cls_score, bbox_pred

class RoIHeads(nn.Module):
    """RoI处理模块"""
    def __init__(self, 
                 box_roi_pool: MultiScaleRoIAlign,
                 head: RoIHead,
                 fg_iou_thresh: float = 0.5,
                 bg_iou_thresh: float = 0.5,
                 batch_size_per_image: int = 512,
                 positive_fraction: float = 0.25,
                 score_thresh: float = 0.05,
                 nms_thresh: float = 0.5,
                 detections_per_img: int = 100):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.head = head
        
        # 训练参数
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        
        # 推理参数
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        
        self.logger = logging.getLogger(__name__)
        
    def forward(self, features: Dict[str, torch.Tensor], proposals: List[torch.Tensor],
                image_shapes: List[Tuple[int, int]], targets: Optional[List[Dict[str, torch.Tensor]]] = None) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """
        前向传播
        Args:
            features: 特征图字典
            proposals: 建议框列表
            image_shapes: 图像尺寸列表
            targets: 目标标注（训练时使用）
        Returns:
            result: 检测结果列表
            losses: 损失字典（训练时使用）
        """
        if self.training and targets is not None:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(
                proposals, targets
            )
            
        # 提取RoI特征
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.head(box_features)
        
        class_logits, box_regression = box_features
        
        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        
        if self.training and targets is not None:
            loss_classifier, loss_box_reg = self.compute_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
            }
        else:
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
                
        return result, losses
        
    def select_training_samples(self, proposals: List[torch.Tensor],
                              targets: List[Dict[str, torch.Tensor]]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """选择训练样本"""
        matched_idxs = []
        labels = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            if targets_per_image["boxes"].numel() == 0:
                # 背景图像
                device = proposals_per_image.device
                matched_idxs_per_image = torch.full(
                    (proposals_per_image.shape[0],), -1, dtype=torch.int64, device=device
                )
                labels_per_image = torch.zeros(
                    (proposals_per_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                # 计算IoU
                match_quality_matrix = box_iou(targets_per_image["boxes"], proposals_per_image)
                matched_idxs_per_image = self.proposal_matcher(match_quality_matrix)
                
                # 获取匹配的标签
                labels_per_image = targets_per_image["labels"][matched_idxs_per_image]
                labels_per_image = labels_per_image.to(dtype=torch.int64)
                
                # 背景标签
                labels_per_image[matched_idxs_per_image == self.proposal_matcher.BELOW_LOW_THRESHOLD] = 0
                # 忽略标签
                labels_per_image[matched_idxs_per_image == self.proposal_matcher.BETWEEN_THRESHOLDS] = -1
                
            matched_idxs.append(matched_idxs_per_image)
            labels.append(labels_per_image)
            
        # 采样正负样本
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        
        # 获取采样索引
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
            
        # 获取回归目标
        regression_targets = self.box_coder.encode(
            [targets_per_image["boxes"][matched_idxs_per_image] for targets_per_image, matched_idxs_per_image in zip(targets, matched_idxs)],
            proposals
        )
        
        return proposals, matched_idxs, labels, regression_targets
        
    def compute_loss(self, class_logits: torch.Tensor, box_regression: torch.Tensor,
                    labels: List[torch.Tensor], regression_targets: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算RoI损失"""
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        
        # 分类损失
        classification_loss = F.cross_entropy(class_logits, labels)
        
        # 回归损失
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=class_logits.device)
        box_regression_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            beta=1.0,
            reduction="sum",
        )
        box_regression_loss = box_regression_loss / labels.numel()
        
        return classification_loss, box_regression_loss
        
    def postprocess_detections(self, class_logits: torch.Tensor, box_regression: torch.Tensor,
                             proposals: List[torch.Tensor], image_shapes: List[Tuple[int, int]]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """后处理检测结果"""
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            # 裁剪到图像边界
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, image_shape[1])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, image_shape[0])
            
            # 创建结果列表
            boxes_list = []
            scores_list = []
            labels_list = []
            
            # 对每个类别进行NMS
            for class_id in range(1, num_classes):
                # 获取当前类别的得分
                score = scores[:, class_id]
                
                # 应用阈值
                keep_idxs = score > self.score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]
                
                if score.shape[0] == 0:
                    continue
                    
                # 应用NMS
                keep = nms(box, score, self.nms_thresh)
                keep = keep[:self.detections_per_img]
                
                boxes_list.append(box[keep])
                scores_list.append(score[keep])
                labels_list.append(torch.full_like(score[keep], class_id, dtype=torch.int64))
                
            # 合并所有类别的结果
            if boxes_list:
                boxes_list = torch.cat(boxes_list, dim=0)
                scores_list = torch.cat(scores_list, dim=0)
                labels_list = torch.cat(labels_list, dim=0)
                
                # 按得分排序
                _, idx = scores_list.sort(0, descending=True)
                idx = idx[:self.detections_per_img]
                
                all_boxes.append(boxes_list[idx])
                all_scores.append(scores_list[idx])
                all_labels.append(labels_list[idx])
            else:
                all_boxes.append(torch.zeros((0, 4), dtype=torch.float32, device=device))
                all_scores.append(torch.zeros((0,), dtype=torch.float32, device=device))
                all_labels.append(torch.zeros((0,), dtype=torch.int64, device=device))
                
        return all_boxes, all_scores, all_labels 