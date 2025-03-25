import torch
import torch.nn as nn
from pathlib import Path
import logging
from models.segmentation.pruner import SegmentationPruner
from models.segmentation.unet import UNet
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

def setup_logging(output_dir: Path):
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'prune_example.log'),
            logging.StreamHandler()
        ]
    )

class SegmentationDataset(torch.utils.data.Dataset):
    """简单的分割数据集"""
    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_files = list(self.image_dir.glob('*.png'))
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_dir / image_path.name
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return image, mask

def main():
    """主函数"""
    # 设置输出目录
    output_dir = Path('pruning_example')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    try:
        # 创建示例模型
        logger.info("创建示例模型...")
        model = UNet(n_channels=3, n_classes=1)
        
        # 创建数据加载器
        logger.info("创建数据加载器...")
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        
        dataset = SegmentationDataset(
            image_dir='path/to/images',
            mask_dir='path/to/masks',
            transform=transform
        )
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # 创建剪枝器
        logger.info("创建剪枝器...")
        pruner = SegmentationPruner(model=model)
        
        # 运行不同剪枝方法的对比
        methods = ['magnitude', 'l1_structured', 'channel']
        amounts = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for method in methods:
            logger.info(f"\n测试 {method} 剪枝方法...")
            
            for amount in amounts:
                logger.info(f"剪枝比例: {amount}")
                
                # 运行剪枝
                results = pruner.run_complete_pruning(
                    method=method,
                    amount=amount,
                    dataloader=dataloader,
                    criterion=nn.BCEWithLogitsLoss(),
                    analyze_sensitivity=True,
                    output_dir=output_dir / f"{method}_{amount}"
                )
                
                # 保存剪枝后的模型
                pruner.save_pruned_model(
                    output_dir / f"{method}_{amount}" / "model_pruned.pth"
                )
                
                logger.info(f"完成 {method} 方法 {amount} 比例的剪枝")
                
        logger.info("\n所有剪枝测试完成！")
        
    except Exception as e:
        logger.error(f"剪枝过程中出错: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 