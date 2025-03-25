import torch
import torch.nn as nn
from pathlib import Path
import logging
import argparse
from models.segmentation.advanced_pruner import AdvancedPruner
from models.segmentation.unet import UNet
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='高级模型剪枝示例')
    
    # 基本参数
    parser.add_argument('--model-path', type=str, help='模型路径')
    parser.add_argument('--image-dir', type=str, required=True, help='图像目录')
    parser.add_argument('--mask-dir', type=str, required=True, help='掩码目录')
    parser.add_argument('--output-dir', type=str, default='advanced_pruning_results', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备')
    
    # 数据加载参数
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--image-size', type=int, default=512, help='图像大小')
    
    # 剪枝参数
    parser.add_argument('--methods', nargs='+', default=['random', 'gradient', 'weight_norm', 'taylor'],
                      help='剪枝方法列表')
    parser.add_argument('--amounts', nargs='+', type=float, default=[0.1, 0.3, 0.5],
                      help='剪枝比例列表')
    parser.add_argument('--iterative', action='store_true', help='是否使用迭代式剪枝')
    parser.add_argument('--steps', type=int, default=5, help='迭代步数')
    parser.add_argument('--mixed', action='store_true', help='是否使用混合剪枝')
    
    return parser.parse_args()

def setup_logging(output_dir: Path):
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'advanced_pruning.log'),
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
        # 创建或加载模型
        logger.info("准备模型...")
        if args.model_path:
            model = torch.load(args.model_path)
        else:
            model = UNet(n_channels=3, n_classes=1)
            
        # 创建数据加载器
        logger.info("创建数据加载器...")
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ])
        
        dataset = SegmentationDataset(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            transform=transform
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        
        # 创建剪枝器
        logger.info("创建高级剪枝器...")
        pruner = AdvancedPruner(model=model, device=args.device)
        criterion = nn.BCEWithLogitsLoss()
        
        # 执行剪枝
        if args.mixed:
            # 混合剪枝
            logger.info("执行混合剪枝...")
            methods = [
                {'name': method, 'amount': amount}
                for method in args.methods
                for amount in args.amounts
            ]
            results = pruner.mixed_pruning(
                methods=methods,
                dataloader=dataloader,
                criterion=criterion
            )
            
            # 保存结果
            for method, result in results.items():
                method_dir = output_dir / f"mixed_{method}"
                method_dir.mkdir(parents=True, exist_ok=True)
                pruner.create_pruning_report(result, method_dir)
                
        else:
            # 单独测试每种方法
            for method in args.methods:
                logger.info(f"\n测试 {method} 剪枝方法...")
                
                for amount in args.amounts:
                    logger.info(f"剪枝比例: {amount}")
                    method_dir = output_dir / method / f"amount_{amount}"
                    method_dir.mkdir(parents=True, exist_ok=True)
                    
                    if args.iterative:
                        # 迭代式剪枝
                        results = pruner.iterative_pruning(
                            target_amount=amount,
                            steps=args.steps,
                            method=method,
                            dataloader=dataloader,
                            criterion=criterion
                        )
                        
                        # 保存每步的结果
                        for step_result in results:
                            step_dir = method_dir / f"step_{step_result['step']}"
                            step_dir.mkdir(parents=True, exist_ok=True)
                            pruner.create_pruning_report(step_result, step_dir)
                            
                    else:
                        # 直接剪枝
                        if method == 'random':
                            pruner.random_pruning(amount)
                        elif method == 'gradient':
                            pruner.gradient_based_pruning(amount, dataloader, criterion)
                        elif method == 'weight_norm':
                            pruner.weight_norm_pruning(amount)
                        elif method == 'taylor':
                            pruner.taylor_pruning(amount, dataloader, criterion)
                            
                        # 分析并保存结果
                        results = pruner.analyze_model_sparsity()
                        pruner.create_pruning_report(results, method_dir)
                        
                    # 保存模型
                    pruner.save_pruned_model(
                        method_dir / 'model_pruned.pth',
                        metadata={
                            'method': method,
                            'amount': amount,
                            'iterative': args.iterative,
                            'steps': args.steps if args.iterative else None
                        }
                    )
                    
        logger.info("\n所有剪枝测试完成！")
        
    except Exception as e:
        logger.error(f"剪枝过程中出错: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 