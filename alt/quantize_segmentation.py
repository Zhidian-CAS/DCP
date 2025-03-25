import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
from typing import List, Optional, Tuple
import yaml
from tqdm import tqdm

from models.segmentation.quantization import SegmentationQuantizer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='量化分割模型')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, required=True, help='模型路径')
    parser.add_argument('--input-size', type=int, nargs=2, default=[512, 512], help='输入图像大小')
    parser.add_argument('--num-bits', type=int, default=8, help='量化位数')
    
    # 数据参数
    parser.add_argument('--calibration-dir', type=str, required=True, help='校准数据目录')
    parser.add_argument('--test-dir', type=str, help='测试数据目录')
    parser.add_argument('--test-mask-dir', type=str, help='测试掩码目录')
    
    # 量化参数
    parser.add_argument('--quantization-type', type=str, choices=['static', 'dynamic'], default='static', help='量化类型')
    parser.add_argument('--export-onnx', action='store_true', help='是否导出ONNX模型')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='quantized_models', help='输出目录')
    
    return parser.parse_args()

def setup_logging(output_dir: Path):
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'quantize.log'),
            logging.StreamHandler()
        ]
    )

def load_images(
    image_dir: str,
    mask_dir: Optional[str] = None
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
    """
    加载图像
    Args:
        image_dir: 图像目录
        mask_dir: 掩码目录
    Returns:
        图像列表和掩码列表
    """
    # 获取图像路径
    image_paths = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
    images = []
    masks = None if mask_dir is None else []
    
    # 加载图像
    for image_path in tqdm(image_paths, desc='Loading images'):
        # 加载图像
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        
        # 加载掩码
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
    
    # 创建量化器
    quantizer = SegmentationQuantizer(
        model_path=args.model_path,
        input_size=tuple(args.input_size)
    )
    
    # 加载原始模型大小
    original_size = quantizer.get_model_size(quantizer.model)
    logger.info(f'原始模型大小: {original_size:.2f}MB')
    
    # 加载校准数据
    logger.info('加载校准数据...')
    calibration_data, _ = load_images(args.calibration_dir)
    
    # 加载测试数据
    test_data = None
    test_masks = None
    if args.test_dir:
        logger.info('加载测试数据...')
        test_data, test_masks = load_images(args.test_dir, args.test_mask_dir)
    
    # 量化模型
    logger.info(f'开始{args.quantization_type}量化...')
    if args.quantization_type == 'static':
        quantized_model = quantizer.prepare_static_quantization(
            calibration_data=calibration_data,
            num_bits=args.num_bits
        )
    else:
        quantized_model = quantizer.prepare_dynamic_quantization(
            num_bits=args.num_bits
        )
    
    # 获取量化后模型大小
    quantized_size = quantizer.get_model_size(quantized_model)
    compression_ratio = original_size / quantized_size
    logger.info(f'量化后模型大小: {quantized_size:.2f}MB')
    logger.info(f'压缩比: {compression_ratio:.2f}x')
    
    # 评估量化模型
    if test_data is not None:
        logger.info('评估量化模型...')
        metrics = quantizer.evaluate_quantized_model(
            quantized_model=quantized_model,
            test_data=test_data,
            test_masks=test_masks
        )
        
        # 保存评估结果
        metrics['model_size'] = quantized_size
        metrics['compression_ratio'] = compression_ratio
        with open(output_dir / 'metrics.yaml', 'w') as f:
            yaml.dump(metrics, f)
        
        # 打印评估结果
        logger.info('评估结果:')
        for metric, value in metrics.items():
            if value is not None:
                logger.info(f'{metric}: {value:.4f}')
    
    # 导出ONNX模型
    if args.export_onnx:
        logger.info('导出ONNX模型...')
        quantizer.export_quantized_model(
            quantized_model=quantized_model,
            save_path=str(output_dir / 'model_quantized.onnx')
        )
    
    logger.info('量化完成！')

if __name__ == '__main__':
    main() 