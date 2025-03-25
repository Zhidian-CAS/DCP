import argparse
import logging
import yaml
from pathlib import Path
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from models.segmentation.segnet import SegNet
from models.segmentation.dataset import SegmentationDataset, create_data_loaders
from models.segmentation.augmentation import SegmentationAugmentation
from models.segmentation.metrics import SegmentationMetrics
from models.segmentation.visualization import SegmentationVisualizer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练分割模型')
    
    # 数据集参数
    parser.add_argument('--train-image-dir', type=str, required=True, help='训练图像目录')
    parser.add_argument('--train-mask-dir', type=str, required=True, help='训练掩码目录')
    parser.add_argument('--valid-image-dir', type=str, required=True, help='验证图像目录')
    parser.add_argument('--valid-mask-dir', type=str, required=True, help='验证掩码目录')
    parser.add_argument('--class-names', type=str, nargs='+', default=['background', 'colony'], help='类别名称')
    
    # 模型参数
    parser.add_argument('--num-classes', type=int, default=2, help='类别数量')
    parser.add_argument('--in-channels', type=int, default=3, help='输入通道数')
    parser.add_argument('--base-channels', type=int, default=64, help='基础通道数')
    parser.add_argument('--num-encoders', type=int, default=5, help='编码器数量')
    parser.add_argument('--use-attention', action='store_true', help='是否使用注意力机制')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--num-epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='权重衰减')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载进程数')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--log-interval', type=int, default=10, help='日志间隔')
    parser.add_argument('--save-interval', type=int, default=5, help='保存间隔')
    
    return parser.parse_args()

def setup_logging(output_dir: Path):
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )

def save_config(args, output_dir: Path):
    """保存配置"""
    config = vars(args)
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志和保存配置
    setup_logging(output_dir)
    save_config(args, output_dir)
    logger = logging.getLogger(__name__)
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 创建数据增强器
    train_transform = SegmentationAugmentation(train_mode=True)
    valid_transform = SegmentationAugmentation(train_mode=False)
    
    # 创建数据集
    train_dataset = SegmentationDataset(
        args.train_image_dir,
        args.train_mask_dir,
        transform=train_transform,
        class_names=args.class_names
    )
    valid_dataset = SegmentationDataset(
        args.valid_image_dir,
        args.valid_mask_dir,
        transform=valid_transform,
        class_names=args.class_names
    )
    
    # 创建数据加载器
    train_loader, valid_loader = create_data_loaders(
        train_dataset,
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 创建模型
    model = SegNet(
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        num_encoders=args.num_encoders,
        use_attention=args.use_attention
    ).to(device)
    
    # 计算类别权重
    class_weights = train_dataset.get_class_weights().to(device)
    
    # 创建优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate * 10,
        epochs=args.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # 创建评估器和可视化器
    metrics = SegmentationMetrics(args.num_classes)
    visualizer = SegmentationVisualizer(
        args.class_names,
        save_dir=output_dir / 'visualizations'
    )
    
    # 记录最佳性能
    best_metric = 0.0
    metrics_history = {
        'train_loss': [], 'train_dice': [], 'train_iou': [],
        'valid_loss': [], 'valid_dice': [], 'valid_iou': []
    }
    
    # 训练循环
    logger.info('开始训练...')
    for epoch in range(args.num_epochs):
        logger.info(f'Epoch {epoch+1}/{args.num_epochs}')
        
        # 训练
        model.train()
        metrics.reset()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 准备数据
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # 前向传播
            outputs = model(images)
            loss = model.get_loss(outputs, masks, class_weights)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # 更新指标
            train_loss += loss.item()
            pred_masks = torch.argmax(outputs, dim=1)
            metrics.update(pred_masks.cpu(), masks.cpu())
            
            # 记录日志
            if (batch_idx + 1) % args.log_interval == 0:
                logger.info(f'Train Batch [{batch_idx+1}/{len(train_loader)}] '
                          f'Loss: {loss.item():.4f}')
                
        # 计算训练指标
        train_metrics = metrics.get_metrics()
        train_loss = train_loss / len(train_loader)
        
        # 验证
        model.eval()
        metrics.reset()
        valid_loss = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                # 准备数据
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # 前向传播
                outputs = model(images)
                loss = model.get_loss(outputs, masks, class_weights)
                
                # 更新指标
                valid_loss += loss.item()
                pred_masks = torch.argmax(outputs, dim=1)
                metrics.update(pred_masks.cpu(), masks.cpu())
                
        # 计算验证指标
        valid_metrics = metrics.get_metrics()
        valid_loss = valid_loss / len(valid_loader)
        
        # 更新指标历史
        metrics_history['train_loss'].append(train_loss)
        metrics_history['train_dice'].append(train_metrics['mean_dice'])
        metrics_history['train_iou'].append(train_metrics['mean_iou'])
        metrics_history['valid_loss'].append(valid_loss)
        metrics_history['valid_dice'].append(valid_metrics['mean_dice'])
        metrics_history['valid_iou'].append(valid_metrics['mean_iou'])
        
        # 记录TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Dice/train', train_metrics['mean_dice'], epoch)
        writer.add_scalar('Dice/valid', valid_metrics['mean_dice'], epoch)
        writer.add_scalar('IoU/train', train_metrics['mean_iou'], epoch)
        writer.add_scalar('IoU/valid', valid_metrics['mean_iou'], epoch)
        
        # 保存可视化结果
        if (epoch + 1) % args.save_interval == 0:
            visualizer.visualize_batch(
                images.cpu(),
                pred_masks.cpu(),
                masks.cpu(),
                save_path=output_dir / 'visualizations' / f'epoch_{epoch+1}.png'
            )
            
        # 保存最佳模型
        current_metric = valid_metrics['mean_dice']
        if current_metric > best_metric:
            best_metric = current_metric
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': best_metric,
                'metrics': valid_metrics
            }, output_dir / 'best_model.pth')
            logger.info(f'保存最佳模型，指标: {best_metric:.4f}')
            
        # 绘制指标曲线
        visualizer.plot_metrics(
            metrics_history,
            save_path=output_dir / 'metrics.png'
        )
        
        # 打印指标
        logger.info(
            f'Epoch {epoch+1} - '
            f'Train Loss: {train_loss:.4f}, '
            f'Train Dice: {train_metrics["mean_dice"]:.4f}, '
            f'Train IoU: {train_metrics["mean_iou"]:.4f}, '
            f'Valid Loss: {valid_loss:.4f}, '
            f'Valid Dice: {valid_metrics["mean_dice"]:.4f}, '
            f'Valid IoU: {valid_metrics["mean_iou"]:.4f}'
        )
        
    writer.close()
    logger.info('训练完成！')

if __name__ == '__main__':
    main() 