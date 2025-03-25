import argparse
import logging
from pathlib import Path
import torch
from models.segmentation.optimizer import SegmentationOptimizer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='分析分割模型性能并提供优化建议')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, required=True, help='模型路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备')
    
    # 分析参数
    parser.add_argument('--input-size', type=int, nargs=2, default=[512, 512], help='输入图像大小')
    parser.add_argument('--num-iterations', type=int, default=100, help='迭代次数')
    parser.add_argument('--warmup', type=int, default=10, help='预热次数')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='optimization_results', help='输出目录')
    
    return parser.parse_args()

def setup_logging(output_dir: Path):
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'optimize.log'),
            logging.StreamHandler()
        ]
    )

def load_model(model_path: str, device: str):
    """加载模型"""
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

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
        
        # 创建优化器
        optimizer = SegmentationOptimizer(
            model=model,
            device=args.device
        )
        
        # 运行性能分析
        logger.info("开始性能分析...")
        metrics = optimizer.run_complete_analysis(
            input_size=tuple(args.input_size),
            num_iterations=args.num_iterations,
            warmup=args.warmup,
            output_dir=output_dir
        )
        
        logger.info(f"性能分析完成！结果已保存至: {output_dir}")
        
    except Exception as e:
        logger.error(f"性能分析过程中出错: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 