import argparse
import logging
from pathlib import Path
import torch
import cv2
import numpy as np
import yaml
from models.segmentation.visualizer import SegmentationVisualizer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='可视化分割模型')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, required=True, help='模型路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备')
    
    # 可视化参数
    parser.add_argument('--image-path', type=str, help='输入图像路径')
    parser.add_argument('--mask-path', type=str, help='真实掩码路径')
    parser.add_argument('--sensitivity-results', type=str, help='敏感度分析结果路径')
    parser.add_argument('--performance-metrics', type=str, help='性能指标路径')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='visualization_results', help='输出目录')
    
    return parser.parse_args()

def setup_logging(output_dir: Path):
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'visualize.log'),
            logging.StreamHandler()
        ]
    )

def load_model(model_path: str, device: str):
    """加载模型"""
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def load_image(image_path: str) -> np.ndarray:
    """加载图像"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_mask(mask_path: str) -> np.ndarray:
    """加载掩码"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return mask

def load_sensitivity_results(results_path: str) -> dict:
    """加载敏感度分析结果"""
    with open(results_path, 'r') as f:
        return yaml.safe_load(f)

def load_performance_metrics(metrics_path: str) -> dict:
    """加载性能指标"""
    with open(metrics_path, 'r') as f:
        return yaml.safe_load(f)

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
        
        # 创建可视化器
        visualizer = SegmentationVisualizer(
            model=model,
            device=args.device
        )
        
        # 加载输入图像和掩码
        image = None
        mask = None
        if args.image_path:
            logger.info(f"加载图像: {args.image_path}")
            image = load_image(args.image_path)
            if args.mask_path:
                logger.info(f"加载掩码: {args.mask_path}")
                mask = load_mask(args.mask_path)
                
        # 加载敏感度分析结果
        sensitivity_results = None
        if args.sensitivity_results:
            logger.info(f"加载敏感度分析结果: {args.sensitivity_results}")
            sensitivity_results = load_sensitivity_results(args.sensitivity_results)
            
        # 加载性能指标
        performance_metrics = None
        if args.performance_metrics:
            logger.info(f"加载性能指标: {args.performance_metrics}")
            performance_metrics = load_performance_metrics(args.performance_metrics)
            
        # 创建可视化报告
        visualizer.create_visualization_report(
            output_dir=output_dir,
            image=image,
            mask=mask,
            sensitivity_results=sensitivity_results,
            performance_metrics=performance_metrics
        )
        
        logger.info(f"可视化完成！结果已保存至: {output_dir}")
        
    except Exception as e:
        logger.error(f"可视化过程中出错: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 