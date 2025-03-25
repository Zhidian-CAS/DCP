import logging
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.models.sota_comparison import SOTAComparator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleSegmentationModel(nn.Module):
    """简单的分割模型示例"""
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
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 2, stride=2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
    model = SimpleSegmentationModel().to(device)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 初始化SOTA比较器
    sota_config_path = Path("configs/sota_models.yaml")
    comparator = SOTAComparator(sota_config_path)
    
    # 与SegNet比较
    logger.info("开始与SegNet比较...")
    result = comparator.compare_with_sota(
        model=model,
        model_name="SimpleSegmentationModel",
        sota_name="segnet",
        test_dataloader=dataloader,
        device=device,
        num_runs=10,  # 示例中减少运行次数
        warmup=2
    )
    
    # 保存比较结果
    output_dir = Path("results/sota_comparison")
    comparator.save_comparison_results(result, output_dir)
    logger.info(f"比较结果已保存到: {output_dir}")
    
    # 与U-Net比较
    logger.info("开始与U-Net比较...")
    result = comparator.compare_with_sota(
        model=model,
        model_name="SimpleSegmentationModel",
        sota_name="unet",
        test_dataloader=dataloader,
        device=device,
        num_runs=10,
        warmup=2
    )
    
    # 保存比较结果
    output_dir = Path("results/sota_comparison_unet")
    comparator.save_comparison_results(result, output_dir)
    logger.info(f"比较结果已保存到: {output_dir}")

if __name__ == "__main__":
    main() 