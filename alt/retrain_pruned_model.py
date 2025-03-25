import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import logging
import argparse
from models.segmentation.advanced_pruner import AdvancedPruner
from models.segmentation.pruning_trainer import PruningTrainer
from models.segmentation.unet import UNet
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='剪枝模型重训练示例')
    
    # 基本参数
    parser.add_argument('--pruned-model-path', type=str, required=True, help='剪枝后的模型路径')
    parser.add_argument('--train-image-dir', type=str, required=True, help='训练图像目录')
    parser.add_argument('--train-mask-dir', type=str, required=True, help='训练掩码目录')
    parser.add_argument('--val-image-dir', type=str, required=True, help='验证图像目录')
    parser.add_argument('--val-mask-dir', type=str, required=True, help='验证掩码目录')
    parser.add_argument('--output-dir', type=str, default='retrain_results', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备')
    
    # 数据加载参数
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--image-size', type=int, default=512, help='图像大小')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--early-stopping', type=int, default=10, help='早停轮数')
    parser.add_argument('--save-interval', type=int, default=5, help='保存间隔')
    
    return parser.parse_args()

def setup_logging(output_dir: Path):
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'retrain.log'),
            logging.StreamHandler()
        ]
    )

class SegmentationDataset(torch.utils.data.Dataset):
    """分割数据集"""
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

def dice_coefficient(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """计算Dice系数"""
    smooth = 1e-5
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > 0.5).float()
    intersection = (outputs * targets).sum()
    union = outputs.sum() + targets.sum()
    return (2. * intersection + smooth) / (union + smooth)

def iou_score(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """计算IoU分数"""
    smooth = 1e-5
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > 0.5).float()
    intersection = (outputs * targets).sum()
    union = outputs.sum() + targets.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def main():
    """主函数"""
    args = parse_args()
    
    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    try:
        # 加载剪枝后的模型
        logger.info("加载剪枝后的模型...")
        checkpoint = torch.load(args.pruned_model_path)
        model = checkpoint['model_state_dict']
        sparsity = checkpoint.get('sparsity', None)
        
        if sparsity:
            logger.info(f"模型稀疏度: {sparsity['overall_sparsity']:.4f}")
            
        # 创建数据加载器
        logger.info("创建数据加载器...")
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ])
        
        train_dataset = SegmentationDataset(
            image_dir=args.train_image_dir,
            mask_dir=args.train_mask_dir,
            transform=transform
        )
        val_dataset = SegmentationDataset(
            image_dir=args.val_image_dir,
            mask_dir=args.val_mask_dir,
            transform=transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        # 设置优化器和调度器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
        
        # 设置损失函数和评估指标
        criterion = nn.BCEWithLogitsLoss()
        metrics = {
            'dice': dice_coefficient,
            'iou': iou_score
        }
        
        # 创建训练器
        logger.info("创建训练器...")
        trainer = PruningTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            metrics=metrics
        )
        
        # 开始训练
        logger.info("开始重训练...")
        history = trainer.train(
            epochs=args.epochs,
            output_dir=output_dir,
            save_best=True,
            early_stopping=args.early_stopping,
            save_interval=args.save_interval
        )
        
        logger.info("重训练完成！")
        
    except Exception as e:
        logger.error(f"重训练过程中出错: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 