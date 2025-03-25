import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from models.segmentation.pruner import SegmentationPruner
from models.segmentation.dataset import SegmentationDataset

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='剪枝分割模型')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, required=True, help='模型路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备')
    
    # 剪枝参数
    parser.add_argument('--method', type=str, choices=['magnitude', 'l1_structured', 'channel'], default='magnitude', help='剪枝方法')
    parser.add_argument('--amount', type=float, default=0.5, help='剪枝比例')
    parser.add_argument('--global-pruning', action='store_true', help='是否全局剪枝')
    
    # 敏感度分析参数
    parser.add_argument('--analyze-sensitivity', action='store_true', help='是否进行敏感度分析')
    parser.add_argument('--test-image-dir', type=str, help='测试图像目录')
    parser.add_argument('--test-mask-dir', type=str, help='测试掩码目录')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载线程数')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='pruned_models', help='输出目录')
    
    return parser.parse_args()

def setup_logging(output_dir: Path):
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'prune.log'),
            logging.StreamHandler()
        ]
    )

def load_model(model_path: str, device: str):
    """加载模型"""
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def create_dataloader(
    image_dir: str,
    mask_dir: str,
    batch_size: int,
    num_workers: int
) -> DataLoader:
    """创建数据加载器"""
    dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        train=False
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

def format_results(results: dict) -> str:
    """格式化结果为易读的字符串"""
    output = []
    
    # 基本信息
    output.append("剪枝信息:")
    output.append(f"  方法: {results['method']}")
    output.append(f"  剪枝比例: {results['amount']}")
    output.append(f"  全局剪枝: {results['global_pruning']}")
    output.append("")
    
    # 原始稀疏度
    output.append("原始模型:")
    original = results['original_sparsity']
    output.append(f"  总参数量: {original['total_parameters']:,}")
    output.append(f"  零参数量: {original['zero_parameters']:,}")
    output.append(f"  稀疏度: {original['sparsity']:.4f}")
    output.append("")
    
    # 剪枝后稀疏度
    output.append("剪枝后模型:")
    pruned = results['pruned_sparsity']
    output.append(f"  总参数量: {pruned['total_parameters']:,}")
    output.append(f"  零参数量: {pruned['zero_parameters']:,}")
    output.append(f"  稀疏度: {pruned['sparsity']:.4f}")
    output.append("")
    
    # 敏感度分析结果
    if results['sensitivity_results']:
        output.append("敏感度分析:")
        for method, amounts in results['sensitivity_results'].items():
            output.append(f"  {method}:")
            for amount, loss in amounts.items():
                output.append(f"    剪枝比例 {amount}: 损失 {loss:.4f}")
    
    return "\n".join(output)

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    try:
        # 加载模型
        logger.info(f"加载模型: {args.model_path}")
        model = load_model(args.model_path, args.device)
        
        # 创建剪枝器
        pruner = SegmentationPruner(
            model=model,
            device=args.device
        )
        
        # 准备敏感度分析所需的数据加载器和损失函数
        dataloader = None
        criterion = None
        if args.analyze_sensitivity:
            if not (args.test_image_dir and args.test_mask_dir):
                raise ValueError("进行敏感度分析需要提供测试数据目录")
                
            logger.info("创建数据加载器...")
            dataloader = create_dataloader(
                image_dir=args.test_image_dir,
                mask_dir=args.test_mask_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            
            criterion = nn.CrossEntropyLoss()
        
        # 运行剪枝
        results = pruner.run_complete_pruning(
            method=args.method,
            amount=args.amount,
            global_pruning=args.global_pruning,
            save_path=output_dir / 'model_pruned.pth',
            analyze_sensitivity=args.analyze_sensitivity,
            dataloader=dataloader,
            criterion=criterion
        )
        
        # 格式化并保存结果
        formatted_results = format_results(results)
        with open(output_dir / 'prune_results.txt', 'w', encoding='utf-8') as f:
            f.write(formatted_results)
            
        # 保存原始结果
        with open(output_dir / 'prune_results.yaml', 'w') as f:
            yaml.dump(results, f)
            
        logger.info(f"剪枝完成！结果已保存至: {output_dir}")
        
    except Exception as e:
        logger.error(f"剪枝过程中出错: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 