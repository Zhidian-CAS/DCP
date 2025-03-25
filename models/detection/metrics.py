import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict

class DetectionMetrics:
    """目标检测评估指标"""
    def __init__(self, 
                 num_classes: int,
                 iou_threshold: float = 0.5,
                 score_threshold: float = 0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化指标
        self.reset()
        
    def reset(self):
        """重置指标"""
        self.gt_boxes = defaultdict(list)
        self.pred_boxes = defaultdict(list)
        self.gt_labels = defaultdict(list)
        self.pred_labels = defaultdict(list)
        self.pred_scores = defaultdict(list)
        
    def update(self, 
               pred_boxes: List[torch.Tensor],
               pred_labels: List[torch.Tensor],
               pred_scores: List[torch.Tensor],
               gt_boxes: List[torch.Tensor],
               gt_labels: List[torch.Tensor]):
        """
        更新指标
        Args:
            pred_boxes: 预测框列表
            pred_labels: 预测标签列表
            pred_scores: 预测得分列表
            gt_boxes: 真实框列表
            gt_labels: 真实标签列表
        """
        for i in range(len(pred_boxes)):
            # 获取当前图像的预测和真实值
            curr_pred_boxes = pred_boxes[i].cpu().numpy()
            curr_pred_labels = pred_labels[i].cpu().numpy()
            curr_pred_scores = pred_scores[i].cpu().numpy()
            curr_gt_boxes = gt_boxes[i].cpu().numpy()
            curr_gt_labels = gt_labels[i].cpu().numpy()
            
            # 按类别存储
            for class_id in range(self.num_classes):
                # 真实框
                gt_mask = curr_gt_labels == class_id
                if gt_mask.any():
                    self.gt_boxes[class_id].extend(curr_gt_boxes[gt_mask])
                    self.gt_labels[class_id].extend(curr_gt_labels[gt_mask])
                    
                # 预测框
                pred_mask = curr_pred_labels == class_id
                if pred_mask.any():
                    self.pred_boxes[class_id].extend(curr_pred_boxes[pred_mask])
                    self.pred_labels[class_id].extend(curr_pred_labels[pred_mask])
                    self.pred_scores[class_id].extend(curr_pred_scores[pred_mask])
                    
    def compute(self) -> Dict[str, float]:
        """
        计算评估指标
        Returns:
            metrics: 指标字典
        """
        metrics = {}
        
        # 计算每个类别的指标
        class_metrics = {}
        for class_id in range(self.num_classes):
            class_metrics[class_id] = self._compute_class_metrics(class_id)
            
        # 计算平均指标
        metrics['mAP'] = np.mean([m['AP'] for m in class_metrics.values()])
        metrics['mAP50'] = np.mean([m['AP50'] for m in class_metrics.values()])
        metrics['mAP75'] = np.mean([m['AP75'] for m in class_metrics.values()])
        metrics['mRecall'] = np.mean([m['Recall'] for m in class_metrics.values()])
        metrics['mPrecision'] = np.mean([m['Precision'] for m in class_metrics.values()])
        metrics['mF1'] = np.mean([m['F1'] for m in class_metrics.values()])
        
        # 添加每个类别的指标
        for class_id, m in class_metrics.items():
            for metric_name, value in m.items():
                metrics[f'{metric_name}_class_{class_id}'] = value
                
        return metrics
        
    def _compute_class_metrics(self, class_id: int) -> Dict[str, float]:
        """
        计算单个类别的指标
        Args:
            class_id: 类别ID
        Returns:
            metrics: 指标字典
        """
        # 获取当前类别的预测和真实值
        pred_boxes = np.array(self.pred_boxes[class_id])
        pred_scores = np.array(self.pred_scores[class_id])
        gt_boxes = np.array(self.gt_boxes[class_id])
        
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return {
                'AP': 0.0,
                'AP50': 0.0,
                'AP75': 0.0,
                'Recall': 0.0,
                'Precision': 0.0,
                'F1': 0.0
            }
            
        # 按置信度排序
        sorted_indices = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]
        
        # 计算IoU矩阵
        iou_matrix = self._compute_iou_matrix(pred_boxes, gt_boxes)
        
        # 计算TP、FP、FN
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        fn = np.ones(len(gt_boxes))
        
        for i in range(len(pred_boxes)):
            if iou_matrix[i].max() >= self.iou_threshold:
                matched_gt_idx = iou_matrix[i].argmax()
                if fn[matched_gt_idx]:
                    tp[i] = 1
                    fn[matched_gt_idx] = 0
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
                
        # 计算累积值
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        fn_cumsum = np.cumsum(fn)
        
        # 计算召回率和精确率
        recalls = tp_cumsum / len(gt_boxes)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)
        
        # 计算AP
        ap = self._compute_ap(recalls, precisions)
        
        # 计算不同IoU阈值下的AP
        ap50 = self._compute_ap(recalls, precisions, iou_threshold=0.5)
        ap75 = self._compute_ap(recalls, precisions, iou_threshold=0.75)
        
        # 计算最终指标
        final_recall = tp_cumsum[-1] / len(gt_boxes)
        final_precision = tp_cumsum[-1] / np.maximum(tp_cumsum[-1] + fp_cumsum[-1], np.finfo(np.float64).eps)
        f1 = 2 * final_precision * final_recall / np.maximum(final_precision + final_recall, np.finfo(np.float64).eps)
        
        return {
            'AP': ap,
            'AP50': ap50,
            'AP75': ap75,
            'Recall': final_recall,
            'Precision': final_precision,
            'F1': f1
        }
        
    def _compute_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        计算IoU矩阵
        Args:
            boxes1: 第一组边界框
            boxes2: 第二组边界框
        Returns:
            iou_matrix: IoU矩阵
        """
        iou_matrix = np.zeros((len(boxes1), len(boxes2)))
        
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                iou_matrix[i, j] = self._compute_iou(box1, box2)
                
        return iou_matrix
        
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        计算两个边界框的IoU
        Args:
            box1: 第一个边界框
            box2: 第二个边界框
        Returns:
            iou: IoU值
        """
        # 计算交集
        x1 = np.maximum(box1[0], box2[0])
        y1 = np.maximum(box1[1], box2[1])
        x2 = np.minimum(box1[2], box2[2])
        y2 = np.minimum(box1[3], box2[3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # 计算并集
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        # 计算IoU
        iou = intersection / np.maximum(union, np.finfo(np.float64).eps)
        
        return iou
        
    def _compute_ap(self, recalls: np.ndarray, precisions: np.ndarray, iou_threshold: float = 0.5) -> float:
        """
        计算AP
        Args:
            recalls: 召回率数组
            precisions: 精确率数组
            iou_threshold: IoU阈值
        Returns:
            ap: AP值
        """
        # 确保召回率是单调递增的
        recalls = np.concatenate(([0.], recalls, [1.]))
        precisions = np.concatenate(([1.], precisions, [0.]))
        
        # 计算PR曲线下的面积
        for i in range(precisions.size - 1, 0, -1):
            precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            
        i = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        
        return ap 