import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

from models.segmentation.preprocess import SegmentationPreprocessor

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='预处理分割数据')
    
    # 数据参数
    parser.add_argument('--image-dir', type=str, required=True, help='图像目录')
    parser.add_argument('--mask-dir', type=str, help='掩码目录')
    parser.add_argument('--input-size', type=int, nargs=2, default=[512, 512], help='输入图像大小')
    
    # 处理参数
    parser.add_argument('--num-threads', type=int, default=4, help='线程数')
    parser.add_argument('--augment', action='store_true', help='是否进行数据增强')
    parser.add_argument('--convert-format', type=str, choices=['coco', 'voc'], help='转换格式')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='preprocessed_data', help='输出目录')
    
    return parser.parse_args()

def setup_logging(output_dir: Path):
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'preprocess.log'),
            logging.StreamHandler()
        ]
    )

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
    
    # 创建预处理器
    preprocessor = SegmentationPreprocessor(
        input_size=tuple(args.input_size),
        num_threads=args.num_threads,
        save_dir=str(output_dir)
    )
    
    # 处理数据集
    logger.info('开始处理数据集...')
    processed_data = preprocessor.process_dataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        augment=args.augment,
        save=True
    )
    logger.info(f'处理完成，共处理 {len(processed_data)} 张图像')
    
    # 格式转换
    if args.convert_format:
        logger.info(f'开始转换为 {args.convert_format.upper()} 格式...')
        if args.convert_format == 'coco':
            # 转换为COCO格式
            preprocessor.convert_to_coco(
                image_dir=output_dir / 'images',
                mask_dir=output_dir / 'masks',
                save_path=output_dir / 'annotations.json'
            )
        else:
            # 转换为VOC格式
            preprocessor.convert_to_voc(
                image_dir=output_dir / 'images',
                mask_dir=output_dir / 'masks',
                save_dir=output_dir / 'voc'
            )
        logger.info('格式转换完成')
    
    logger.info('预处理完成！')

if __name__ == '__main__':
    main() 