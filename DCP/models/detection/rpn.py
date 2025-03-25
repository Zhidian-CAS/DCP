import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from typing import Dict, List, Tuple, Optional
import logging

class RPNHead(nn.Module):
    """RPN头部网络"""
    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
        
        # 初始化权重
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = F.relu(self.conv(x))
        cls_logits = self.cls_logits(t)
        bbox_pred = self.bbox_pred(t)
        
        return cls_logits, bbox_pred

class RegionProposalNetwork(nn.Module):
    """增强版区域建议网络"""
    def __init__(self, anchor_generator, head, 
                 fg_iou_thresh: float = 0.7,
                 bg_iou_thresh: float = 0.3,
                 batch_size_per_image: int = 256,
                 positive_fraction: float = 0.5,
                 pre_nms_top_n: Dict[str, int] = None,
                 post_nms_top_n: Dict[str, int] = None,
                 nms_thresh: float = 0.7,
                 score_thresh: float = 0.0):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        
        # 训练参数
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        
        # 推理参数
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        
        self.logger = logging.getLogger(__name__)
        
    def forward(self, images: List[torch.Tensor], features: Dict[str, torch.Tensor],
                targets: Optional[List[Dict[str, torch.Tensor]]] = None) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        前向传播
        Args:
            images: 输入图像列表
            features: 特征图字典
            targets: 目标标注（训练时使用）
        Returns:
            proposals: 建议框列表
            losses: 损失字典（训练时使用）
        """
        # 生成锚框
        anchors = self.anchor_generator(images, features)
        
        # RPN头部预测
        objectness, pred_bbox_deltas = self.head(features)
        
        # 获取图像尺寸
        num_images = len(images)
        image_sizes = [img.shape[-2:] for img in images]
        
        # 解码预测框
        proposals = self.box_coder.decode(pred_bbox_deltas, anchors)
        proposals = proposals.view(num_images, -1, 4)
        
        # 后处理
        boxes = []
        scores = []
        for proposals_per_image, scores_per_image in zip(proposals, objectness):
            boxes_per_image, scores_per_image = self.filter_proposals(
                proposals_per_image, scores_per_image,
                image_sizes, num_images
            )
            boxes.append(boxes_per_image)
            scores.append(scores_per_image)
            
        losses = {}
        if self.training and targets is not None:
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
            
        return boxes, losses
        
    def filter_proposals(self, proposals: torch.Tensor, scores: torch.Tensor,
                        image_sizes: List[Tuple[int, int]], num_images: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """过滤和处理建议框"""
        device = proposals.device
        batch_size = proposals.shape[0]
        
        # 获取每张图像的建议框数量
        proposals_per_image = [proposals[i] for i in range(batch_size)]
        scores_per_image = [scores[i] for i in range(batch_size)]
        
        # 应用NMS
        boxes = []
        scores = []
        for proposals_per_image, scores_per_image, image_size in zip(
            proposals_per_image, scores_per_image, image_sizes
        ):
            boxes_per_image = proposals_per_image
            scores_per_image = scores_per_image
            
            # 裁剪到图像边界
            boxes_per_image[:, [0, 2]] = boxes_per_image[:, [0, 2]].clamp(0, image_size[1])
            boxes_per_image[:, [1, 3]] = boxes_per_image[:, [1, 3]].clamp(0, image_size[0])
            
            # 移除小框
            keep = box_area(boxes_per_image) > 1
            boxes_per_image = boxes_per_image[keep]
            scores_per_image = scores_per_image[keep]
            
            # 非极大值抑制
            keep = nms(boxes_per_image, scores_per_image, self.nms_thresh)
            keep = keep[:self.post_nms_top_n["testing"]]
            boxes_per_image = boxes_per_image[keep]
            scores_per_image = scores_per_image[keep]
            
            boxes.append(boxes_per_image)
            scores.append(scores_per_image)
            
        return boxes, scores
        
    def assign_targets_to_anchors(self, anchors: List[torch.Tensor],
                                targets: List[Dict[str, torch.Tensor]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """为锚框分配目标"""
        labels = []
        matched_gt_boxes = []
        
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            
            if gt_boxes.numel() == 0:
                # 背景图像
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                # 计算IoU
                match_quality_matrix = box_iou(gt_boxes, anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                
                # 获取匹配的GT框
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]
                
                # 设置标签
                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)
                
                # 背景标签
                labels_per_image[matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD] = 0.0
                # 忽略标签
                labels_per_image[matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS] = -1.0
                
            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
            
        return labels, matched_gt_boxes
        
    def compute_loss(self, objectness: List[torch.Tensor], pred_bbox_deltas: List[torch.Tensor],
                    labels: List[torch.Tensor], regression_targets: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算RPN损失"""
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
        
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        
        objectness_loss = F.binary_cross_entropy_with_logits(
            torch.cat(objectness, dim=0)[sampled_inds],
            torch.cat(labels, dim=0)[sampled_inds]
        )
        
        # 只对正样本计算回归损失
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        if sampled_pos_inds.numel() > 0:
            box_loss = F.smooth_l1_loss(
                torch.cat(pred_bbox_deltas, dim=0)[sampled_pos_inds],
                torch.cat(regression_targets, dim=0)[sampled_pos_inds],
                beta=1.0
            )
        else:
            box_loss = torch.tensor(0.0, device=objectness[0].device)
            
        return objectness_loss, box_loss

class BoxCoder:
    """边界框编码器"""
    def __init__(self, weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)):
        self.weights = weights
        
    def encode(self, reference_boxes: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
        """编码参考框"""
        wx, wy, ww, wh = self.weights
        proposals_x1 = proposals[:, 0].unsqueeze(1)
        proposals_y1 = proposals[:, 1].unsqueeze(1)
        proposals_x2 = proposals[:, 2].unsqueeze(1)
        proposals_y2 = proposals[:, 3].unsqueeze(1)
        
        reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
        reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
        reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
        reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)
        
        # 计算中心点和宽高
        ex_widths = proposals_x2 - proposals_x1
        ex_heights = proposals_y2 - proposals_y1
        ex_ctr_x = proposals_x1 + 0.5 * ex_widths
        ex_ctr_y = proposals_y1 + 0.5 * ex_heights
        
        gt_widths = reference_boxes_x2 - reference_boxes_x1
        gt_heights = reference_boxes_y2 - reference_boxes_y1
        gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
        gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights
        
        # 计算目标
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        
        targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets
        
    def decode(self, rel_codes: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """解码相对编码"""
        boxes = boxes.to(rel_codes.dtype)
        
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        
        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh
        
        # 防止指数爆炸
        dw = torch.clamp(dw, max=math.log(1000.0 / 16))
        dh = torch.clamp(dh, max=math.log(1000.0 / 16))
        
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        
        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h
        
        pred_boxes = torch.stack((pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2), dim=1)
        return pred_boxes.reshape(pred_boxes.shape[0], -1) 