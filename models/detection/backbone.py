import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Optional

class BackboneBase(nn.Module):
    """基础骨干网络类"""
    def __init__(self, backbone_name: str, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.out_channels = None
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
        
    def get_out_channels(self) -> int:
        return self.out_channels

class ResNetBackbone(BackboneBase):
    """ResNet骨干网络"""
    def __init__(self, backbone_name: str, pretrained: bool = True):
        super().__init__(backbone_name, pretrained)
        
        if backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            self.out_channels = 2048
        elif backbone_name == 'resnet101':
            backbone = models.resnet101(pretrained=pretrained)
            self.out_channels = 2048
        elif backbone_name == 'resnet152':
            backbone = models.resnet152(pretrained=pretrained)
            self.out_channels = 2048
        else:
            raise ValueError(f"Unsupported ResNet backbone: {backbone_name}")
            
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return {'feat': x}

class EfficientNetBackbone(BackboneBase):
    """EfficientNet骨干网络"""
    def __init__(self, backbone_name: str, pretrained: bool = True):
        super().__init__(backbone_name, pretrained)
        
        if backbone_name == 'efficientnet_b0':
            backbone = models.efficientnet_b0(pretrained=pretrained)
            self.out_channels = 1280
        elif backbone_name == 'efficientnet_b1':
            backbone = models.efficientnet_b1(pretrained=pretrained)
            self.out_channels = 1280
        elif backbone_name == 'efficientnet_b2':
            backbone = models.efficientnet_b2(pretrained=pretrained)
            self.out_channels = 1408
        else:
            raise ValueError(f"Unsupported EfficientNet backbone: {backbone_name}")
            
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return {'feat': x}

class DenseNetBackbone(BackboneBase):
    """DenseNet骨干网络"""
    def __init__(self, backbone_name: str, pretrained: bool = True):
        super().__init__(backbone_name, pretrained)
        
        if backbone_name == 'densenet121':
            backbone = models.densenet121(pretrained=pretrained)
            self.out_channels = 1024
        elif backbone_name == 'densenet169':
            backbone = models.densenet169(pretrained=pretrained)
            self.out_channels = 1664
        elif backbone_name == 'densenet201':
            backbone = models.densenet201(pretrained=pretrained)
            self.out_channels = 1920
        else:
            raise ValueError(f"Unsupported DenseNet backbone: {backbone_name}")
            
        self.features = backbone.features
        self.norm5 = backbone.norm5
        self.relu = backbone.relu
        self.avgpool = backbone.avgpool
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.features(x)
        x = self.relu(x)
        x = self.norm5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return {'feat': x}

class BackboneFactory:
    """骨干网络工厂类"""
    @staticmethod
    def create_backbone(backbone_name: str, pretrained: bool = True) -> BackboneBase:
        if backbone_name.startswith('resnet'):
            return ResNetBackbone(backbone_name, pretrained)
        elif backbone_name.startswith('efficientnet'):
            return EfficientNetBackbone(backbone_name, pretrained)
        elif backbone_name.startswith('densenet'):
            return DenseNetBackbone(backbone_name, pretrained)
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_name}") 