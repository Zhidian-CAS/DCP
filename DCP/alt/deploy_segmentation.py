import argparse
import logging
import yaml
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from models.segmentation.deploy import SegmentationDeployer
from models.segmentation.visualization import SegmentationVisualizer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='部署分割模型')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, required=True, help='模型路径')
    parser.add_argument('--input-size', type=int, nargs=2, default=[512, 512], help='输入图像大小')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--batch-size', type=int, default=1, help='批次大小')
    
    # 部署参数
    parser.add_argument('--export-onnx', action='store_true', help='是否导出ONNX模型')
    parser.add_argument('--optimize-onnx', action='store_true', help='是否优化ONNX模型')
    parser.add_argument('--onnx-path', type=str, help='ONNX模型路径')
    
    # 测试参数
    parser.add_argument('--test-image-dir', type=str, help='测试图像目录')
    parser.add_argument('--test-mask-dir', type=str, help='测试掩码目录')
    parser.add_argument('--class-names', type=str, nargs='+', default=['background', 'colony'], help='类别名称')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='deploy_outputs', help='输出目录')
    parser.add_argument('--save-images', action='store_true', help='是否保存预测结果')
    parser.add_argument('--benchmark', action='store_true', help='是否进行性能测试')
    parser.add_argument('--num-runs', type=int, default=100, help='性能测试运行次数')
    
    return parser.parse_args()

def setup_logging(output_dir: Path):
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'deploy.log'),
            logging.StreamHandler()
        ]
    )

def load_test_images(
    image_dir: str,
    mask_dir: Optional[str] = None
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
    """
    加载测试图像
    Args:
        image_dir: 图像目录
        mask_dir: 掩码目录（可选）
    Returns:
        图像列表和掩码列表（可选）
    """
    image_paths = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
    images = []
    masks = None if mask_dir is None else []
    
    for image_path in tqdm(image_paths, desc='Loading images'):
        # 加载图像
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        
        # 加载掩码（如果提供）
        if mask_dir is not None:
            mask_path = Path(mask_dir) / image_path.name
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                masks.append(mask)
    
    return images, masks

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
    
    # 创建部署器
    deployer = SegmentationDeployer(
        model_path=args.model_path,
        conf_threshold=args.conf_threshold,
        input_size=tuple(args.input_size),
        batch_size=args.batch_size
    )
    
    # 创建可视化器
    visualizer = SegmentationVisualizer(
        args.class_names,
        save_dir=output_dir / 'visualizations'
    )
    
    # 导出ONNX模型
    if args.export_onnx:
        logger.info('导出ONNX模型...')
        onnx_path = args.onnx_path or str(output_dir / 'model.onnx')
        deployer.export_onnx(onnx_path)
        
        # 优化ONNX模型
        if args.optimize_onnx:
            logger.info('优化ONNX模型...')
            optimized_path = str(output_dir / 'model_optimized.onnx')
            deployer.optimize_onnx(onnx_path, optimized_path)
    
    # 加载测试数据
    if args.test_image_dir:
        logger.info('加载测试数据...')
        images, masks = load_test_images(args.test_image_dir, args.test_mask_dir)
        
        # 性能测试
        if args.benchmark:
            logger.info('进行性能测试...')
            stats = deployer.benchmark(images, num_runs=args.num_runs)
            
            # 打印性能指标
            logger.info(
                f'性能测试结果:\n'
                f'平均时间: {stats["mean_time"]:.2f}ms\n'
                f'标准差: {stats["std_time"]:.2f}ms\n'
                f'最小时间: {stats["min_time"]:.2f}ms\n'
                f'最大时间: {stats["max_time"]:.2f}ms\n'
                f'FPS: {stats["fps"]:.2f}'
            )
            
            # 保存性能指标
            with open(output_dir / 'benchmark.yaml', 'w') as f:
                yaml.dump(stats, f)
        
        # 使用ONNX模型推理
        if args.export_onnx:
            logger.info('使用ONNX模型进行推理...')
            onnx_path = args.onnx_path or str(output_dir / 'model.onnx')
            session = deployer.create_onnx_session(onnx_path)
            
            for i, image in enumerate(tqdm(images, desc='ONNX Inference')):
                # 预测
                pred_mask, pred_prob = deployer.predict_onnx(
                    session,
                    image,
                    return_prob=True
                )
                
                # 保存结果
                if args.save_images:
                    save_path = output_dir / 'predictions' / f'pred_{i}.png'
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 可视化
                    visualizer.visualize_prediction(
                        image,
                        pred_mask,
                        masks[i] if masks else None,
                        save_path=save_path
                    )
        
        # 使用PyTorch模型推理
        else:
            logger.info('使用PyTorch模型进行推理...')
            for i, image in enumerate(tqdm(images, desc='PyTorch Inference')):
                # 预测
                pred_mask, pred_prob = deployer.predict(
                    image,
                    return_prob=True
                )
                
                # 保存结果
                if args.save_images:
                    save_path = output_dir / 'predictions' / f'pred_{i}.png'
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 可视化
                    visualizer.visualize_prediction(
                        image,
                        pred_mask,
                        masks[i] if masks else None,
                        save_path=save_path
                    )
    
    logger.info('部署完成！')

if __name__ == '__main__':
    main() 