import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from config.config import Config
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import cv2

class FeaturePyramidNetwork(nn.Module):
    """自定义特征金字塔网络"""
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
            
    def forward(self, x: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        last_inner = self.inner_blocks[-1](x['feat3'])
        results = []
        results.append(self.layer_blocks[-1](last_inner))
        
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x[f'feat{idx}'])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))
            
        return results

class WellDetector:
    def __init__(self, num_classes=2, backbone='resnet50', pretrained=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.backbone = backbone
        self.model = self._create_model(pretrained)
        self.model.to(self.device)
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _create_model(self, pretrained: bool) -> nn.Module:
        """创建增强版Fast R-CNN模型"""
        # 基础模型
        if self.backbone == 'resnet50':
            model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
            
        # 获取特征维度
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # 自定义特征金字塔网络
        backbone = model.backbone
        fpn = FeaturePyramidNetwork(
            in_channels_list=backbone.out_channels,
            out_channels=256
        )
        backbone.fpn = fpn
        
        # 自定义RPN
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        # 自定义RoI对齐
        roi_align = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )
        
        # 自定义RoI头部
        roi_heads = RoIHeads(
            box_roi_pool=roi_align,
            box_head=model.roi_heads.box_head,
            box_predictor=FastRCNNPredictor(in_features, self.num_classes),
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100
        )
        
        # 自定义变换
        transform = GeneralizedRCNNTransform(
            min_size=800,
            max_size=1333,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )
        
        # 组装模型
        model.rpn = model.rpn
        model.roi_heads = roi_heads
        model.transform = transform
        
        return model
        
    def train(self, train_loader, valid_loader, num_epochs, 
              learning_rate_scheduler=None, warmup_epochs=2):
        """增强版训练函数"""
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # 使用AdamW优化器
        optimizer = torch.optim.AdamW(
            params,
            lr=Config.DETECTION_MODEL['learning_rate'],
            weight_decay=Config.DETECTION_MODEL['weight_decay']
        )
        
        # 学习率调度器
        if learning_rate_scheduler is None:
            learning_rate_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=Config.DETECTION_MODEL['learning_rate'],
                epochs=num_epochs,
                steps_per_epoch=len(train_loader)
            )
            
        # 训练循环
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 预热阶段
            if epoch < warmup_epochs:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = Config.DETECTION_MODEL['learning_rate'] * (epoch + 1) / warmup_epochs
                    
            self.model.train()
            total_loss = 0
            cls_loss = 0
            reg_loss = 0
            rpn_loss = 0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} 
                          for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # 记录各类损失
                cls_loss += loss_dict.get('loss_classifier', 0).item()
                reg_loss += loss_dict.get('loss_box_reg', 0).item()
                rpn_loss += loss_dict.get('loss_rpn_cls', 0).item() + loss_dict.get('loss_rpn_reg', 0).item()
                
                optimizer.zero_grad()
                losses.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                learning_rate_scheduler.step()
                
                total_loss += losses.item()
                
                if batch_idx % 10 == 0:
                    self.logger.info(
                        f'Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_loader)} - '
                        f'Loss: {losses.item():.4f}'
                    )
                    
            # 验证
            val_loss = self._validate(valid_loader)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f'Early stopping triggered after {epoch+1} epochs')
                    break
                    
            self.logger.info(
                f'Epoch {epoch+1}/{num_epochs}:\n'
                f'Training Loss: {total_loss/len(train_loader):.4f}\n'
                f'Classification Loss: {cls_loss/len(train_loader):.4f}\n'
                f'Regression Loss: {reg_loss/len(train_loader):.4f}\n'
                f'RPN Loss: {rpn_loss/len(train_loader):.4f}\n'
                f'Validation Loss: {val_loss:.4f}'
            )
            
    def _validate(self, valid_loader):
        """验证函数"""
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in valid_loader:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} 
                          for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                
        return val_loss / len(valid_loader)
        
    def predict(self, image: np.ndarray, confidence_threshold: float = 0.5,
                nms_threshold: float = 0.5) -> List[Dict]:
        """增强版预测函数"""
        self.model.eval()
        with torch.no_grad():
            # 图像预处理
            image = self._preprocess_image(image)
            
            # 模型推理
            predictions = self.model(image)
            
            # 后处理
            results = self._postprocess_predictions(
                predictions[0],
                confidence_threshold,
                nms_threshold
            )
            
            return results
            
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """图像预处理"""
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = image.unsqueeze(0).to(self.device)
        return image
        
    def _postprocess_predictions(self, prediction: Dict[str, torch.Tensor],
                               confidence_threshold: float,
                               nms_threshold: float) -> List[Dict]:
        """预测结果后处理"""
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        
        # 应用置信度阈值
        mask = scores > confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # 应用NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            confidence_threshold,
            nms_threshold
        )
        
        results = []
        for idx in indices:
            box = boxes[idx]
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            results.append({
                'bbox': (int(x1), int(y1), int(w), int(h)),
                'center': (int(x1 + w/2), int(y1 + h/2)),
                'score': float(scores[idx]),
                'label': int(labels[idx])
            })
            
        return results
        
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'backbone': self.backbone
        }, path)
        self.logger.info(f'Model saved to {path}')
        
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f'Model loaded from {path}')
        
    def export_to_onnx(self, path: str, input_shape: Tuple[int, int] = (800, 800)):
        """导出模型为ONNX格式"""
        dummy_input = torch.randn(1, 3, *input_shape).to(self.device)
        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        self.logger.info(f'Model exported to ONNX format: {path}') 