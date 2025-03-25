import logging
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.models.ablation_study import AblationStudy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SegmentationModel(nn.Module):
    """分割模型示例"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.attention = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.Sigmoid()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 2, stride=2),
            nn.Sigmoid()
        )
        
        self.use_attention = True
        
    def forward(self, x):
        x = self.encoder(x)
        if self.use_attention:
            attention = self.attention(x)
            x = x * attention
        x = self.decoder(x)
        return x
        
    def disable_attention(self):
        """禁用注意力机制"""
        self.use_attention = False
        
    def enable_attention(self):
        """启用注意力机制"""
        self.use_attention = True

def create_dummy_data(num_samples=100, image_size=256):
    """创建示例数据"""
    # 创建随机图像和掩码
    images = torch.randn(num_samples, 3, image_size, image_size)
    masks = torch.randint(0, 2, (num_samples, 1, image_size, image_size), dtype=torch.float32)
    
    return TensorDataset(images, masks)

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建示例数据
    dataset = create_dummy_data()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 创建模型
    model = SegmentationModel().to(device)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 初始化消融实验管理器
    ablation_study = AblationStudy(model)
    
    # 1. 组件消融实验：测试注意力机制
    logger.info("\n运行注意力机制消融实验...")
    result = ablation_study.run_component_ablation(
        component_name="attention",
        disable_fn=model.disable_attention,
        enable_fn=model.enable_attention,
        test_dataloader=dataloader,
        criterion=nn.BCELoss(),
        num_runs=5,
        warmup=2
    )
    logger.info(f"注意力机制消融实验结果: {result.relative_performance}")
    
    # 2. 特征消融实验：测试不同层
    logger.info("\n运行特征消融实验...")
    feature_names = ['encoder', 'decoder']
    results = ablation_study.run_feature_ablation(
        feature_names=feature_names,
        test_dataloader=dataloader,
        criterion=nn.BCELoss(),
        num_runs=5,
        warmup=2
    )
    for result in results:
        logger.info(f"{result.component_name} 消融实验结果: {result.relative_performance}")
    
    # 3. 架构消融实验：测试不同的编码器配置
    logger.info("\n运行架构消融实验...")
    architecture_variants = [
        {
            'name': 'shallow_encoder',
            'config': {
                'encoder_channels': [64, 64, 64],
                'decoder_channels': [64, 64, 1]
            }
        },
        {
            'name': 'deep_encoder',
            'config': {
                'encoder_channels': [64, 128, 256, 512],
                'decoder_channels': [512, 256, 128, 1]
            }
        }
    ]
    results = ablation_study.run_architecture_ablation(
        architecture_variants=architecture_variants,
        test_dataloader=dataloader,
        criterion=nn.BCELoss(),
        num_runs=5,
        warmup=2
    )
    for result in results:
        logger.info(f"{result.component_name} 架构变体实验结果: {result.relative_performance}")
    
    # 保存所有实验结果
    output_dir = Path("results/ablation_study")
    ablation_study.save_results(output_dir)
    logger.info(f"\n实验结果已保存到: {output_dir}")

if __name__ == "__main__":
    main() 