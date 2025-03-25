import argparse
import logging
import yaml
from pathlib import Path
import torch
import numpy as np
import cv2
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

from models.segmentation.segnet import SegNet
from models.segmentation.dataset import SegmentationDataset, create_data_loaders
from models.segmentation.augmentation import SegmentationAugmentation
from models.segmentation.metrics import SegmentationMetrics
from models.segmentation.visualization import SegmentationVisualizer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试分割模型')
    
    # 数据集参数
    parser.add_argument('--test-image-dir', type=str, required=True, help='测试图像目录')
    parser.add_argument('--test-mask-dir', type=str, required=True, help='测试掩码目录')
    parser.add_argument('--class-names', type=str, nargs='+', default=['background', 'colony'], help='类别名称')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, required=True, help='模型路径')
    parser.add_argument('--num-classes', type=int, default=2, help='类别数量')
    parser.add_argument('--in-channels', type=int, default=3, help='输入通道数')
    parser.add_argument('--base-channels', type=int, default=64, help='基础通道数')
    parser.add_argument('--num-encoders', type=int, default=5, help='编码器数量')
    parser.add_argument('--use-attention', action='store_true', help='是否使用注意力机制')
    
    # 测试参数
    parser.add_argument('--batch-size', type=int, default=1, help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载进程数')
    parser.add_argument('--device', type=str, default='cuda', help='测试设备')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='置信度阈值')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='test_outputs', help='输出目录')
    parser.add_argument('--save-images', action='store_true', help='是否保存预测结果')
    parser.add_argument('--save-metrics', action='store_true', help='是否保存评估指标')
    
    return parser.parse_args()

def setup_logging(output_dir: Path):
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'test.log'),
            logging.StreamHandler()
        ]
    )

def load_model(model_path: str, device: torch.device, **kwargs) -> SegNet:
    """
    加载模型
    Args:
        model_path: 模型路径
        device: 设备
        **kwargs: 模型参数
    Returns:
        model: 加载的模型
    """
    # 创建模型
    model = SegNet(**kwargs).to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def test_single_image(
    model: SegNet,
    image_path: str,
    transform: SegmentationAugmentation,
    device: torch.device,
    conf_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    测试单张图像
    Args:
        model: 模型
        image_path: 图像路径
        transform: 数据增强器
        device: 设备
        conf_threshold: 置信度阈值
    Returns:
        pred_mask: 预测掩码
        pred_prob: 预测概率
    """
    # 加载图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 应用变换
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        pred_prob = torch.softmax(outputs, dim=1)
        pred_mask = torch.argmax(outputs, dim=1)
        
        # 应用置信度阈值
        max_prob = torch.max(pred_prob, dim=1)[0]
        pred_mask[max_prob < conf_threshold] = 0
        
    return pred_mask[0].cpu().numpy(), pred_prob[0].cpu().numpy()

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
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 创建数据增强器
    transform = SegmentationAugmentation(train_mode=False)
    
    # 创建数据集和数据加载器
    test_dataset = SegmentationDataset(
        args.test_image_dir,
        args.test_mask_dir,
        transform=transform,
        class_names=args.class_names
    )
    test_loader, _ = create_data_loaders(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    
    # 加载模型
    model = load_model(
        args.model_path,
        device,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        num_encoders=args.num_encoders,
        use_attention=args.use_attention
    )
    
    # 创建评估器和可视化器
    metrics = SegmentationMetrics(args.num_classes)
    visualizer = SegmentationVisualizer(
        args.class_names,
        save_dir=output_dir / 'visualizations'
    )
    
    # 测试
    logger.info('开始测试...')
    model.eval()
    metrics.reset()
    test_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            # 准备数据
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # 前向传播
            outputs = model(images)
            loss = model.get_loss(outputs, masks)
            
            # 获取预测结果
            pred_prob = torch.softmax(outputs, dim=1)
            pred_masks = torch.argmax(outputs, dim=1)
            
            # 应用置信度阈值
            max_prob = torch.max(pred_prob, dim=1)[0]
            pred_masks[max_prob < args.conf_threshold] = 0
            
            # 更新指标
            test_loss += loss.item()
            metrics.update(pred_masks.cpu(), masks.cpu())
            
            # 保存预测结果
            if args.save_images:
                for i in range(images.size(0)):
                    image_path = batch['image_path'][i]
                    save_name = Path(image_path).stem
                    
                    # 保存预测掩码
                    visualizer.visualize_prediction(
                        images[i].cpu().numpy().transpose(1, 2, 0),
                        pred_masks[i].cpu().numpy(),
                        masks[i].cpu().numpy(),
                        save_path=output_dir / 'predictions' / f'{save_name}.png'
                    )
                    
    # 计算评估指标
    test_metrics = metrics.get_metrics()
    test_loss = test_loss / len(test_loader)
    
    # 打印评估结果
    logger.info(
        f'Test Loss: {test_loss:.4f}, '
        f'Dice: {test_metrics["mean_dice"]:.4f}, '
        f'IoU: {test_metrics["mean_iou"]:.4f}, '
        f'Pixel Acc: {test_metrics["pixel_acc"]:.4f}, '
        f'Mean Acc: {test_metrics["mean_acc"]:.4f}'
    )
    
    # 保存评估指标
    if args.save_metrics:
        metrics_file = output_dir / 'metrics.yaml'
        metrics_dict = {
            'test_loss': test_loss,
            **test_metrics
        }
        with open(metrics_file, 'w') as f:
            yaml.dump(metrics_dict, f)
            
    # 绘制混淆矩阵
    metrics.plot_confusion_matrix(
        class_names=args.class_names,
        save_path=output_dir / 'confusion_matrix.png'
    )
    
    logger.info('测试完成！')

if __name__ == '__main__':
    main() 