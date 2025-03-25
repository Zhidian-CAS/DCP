import argparse
import logging
from pathlib import Path
import torch
import yaml
from models.segmentation.profiler import SegmentationProfiler

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='分析分割模型性能')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, required=True, help='模型路径')
    parser.add_argument('--input-size', type=int, nargs=2, default=[512, 512], help='输入图像大小')
    parser.add_argument('--batch-size', type=int, default=1, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备')
    
    # 分析参数
    parser.add_argument('--num-iterations', type=int, default=100, help='迭代次数')
    parser.add_argument('--warmup', type=int, default=10, help='预热次数')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='profile_results', help='输出目录')
    
    return parser.parse_args()

def setup_logging(output_dir: Path):
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'profile.log'),
            logging.StreamHandler()
        ]
    )

def load_model(model_path: str, device: str):
    """加载模型"""
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def format_results(results: dict) -> str:
    """格式化结果为易读的字符串"""
    output = []
    
    # 模型信息
    output.append("模型信息:")
    info = results['model_info']
    output.append(f"  批次大小: {info['batch_size']}")
    output.append(f"  输入大小: {info['input_size']}")
    output.append(f"  设备: {info['device']}")
    output.append("")
    
    # 模型大小
    output.append("模型大小:")
    size = results['model_size']
    output.append(f"  总参数量: {size['total_parameters']:,}")
    output.append(f"  可训练参数量: {size['trainable_parameters']:,}")
    output.append(f"  模型大小: {size['model_size_mb']:.2f}MB")
    output.append("")
    
    # 推理时间
    output.append("推理时间:")
    time = results['inference_time']
    output.append(f"  平均推理时间: {time['mean_inference_time']:.2f}ms")
    output.append(f"  标准差: {time['std_inference_time']:.2f}ms")
    output.append(f"  最小值: {time['min_inference_time']:.2f}ms")
    output.append(f"  最大值: {time['max_inference_time']:.2f}ms")
    output.append(f"  P90: {time['p90_inference_time']:.2f}ms")
    output.append(f"  P95: {time['p95_inference_time']:.2f}ms")
    output.append(f"  P99: {time['p99_inference_time']:.2f}ms")
    output.append("")
    
    # 内存使用
    output.append("内存使用:")
    memory = results['memory_usage']
    for key, value in memory.items():
        output.append(f"  {key}: {value:.2f}MB")
    output.append("")
    
    # GPU使用率（如果有）
    if 'gpu_utilization' in results:
        output.append("GPU使用率:")
        gpu = results['gpu_utilization']
        output.append(f"  GPU使用率: {gpu['gpu_utilization']:.2f}%")
        output.append(f"  GPU内存使用率: {gpu['gpu_memory_utilization']:.2f}%")
        output.append("")
    
    # 各层时间
    output.append("各层推理时间:")
    layers = results['layer_time']
    for layer_name, stats in layers.items():
        output.append(f"  {layer_name}:")
        output.append(f"    CPU时间: {stats['cpu_time_ms']:.2f}ms")
        output.append(f"    CUDA时间: {stats['cuda_time_ms']:.2f}ms")
        output.append(f"    输入形状: {stats['input_shape']}")
        output.append(f"    输出形状: {stats['output_shape']}")
    
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
        
        # 创建性能分析器
        profiler = SegmentationProfiler(
            model=model,
            input_size=tuple(args.input_size),
            batch_size=args.batch_size,
            device=args.device
        )
        
        # 运行性能分析
        results = profiler.run_complete_profile(
            num_iterations=args.num_iterations,
            warmup=args.warmup,
            save_path=output_dir / 'profile_results.json'
        )
        
        # 格式化并保存结果
        formatted_results = format_results(results)
        with open(output_dir / 'profile_results.txt', 'w', encoding='utf-8') as f:
            f.write(formatted_results)
            
        logger.info(f"性能分析完成！结果已保存至: {output_dir}")
        
    except Exception as e:
        logger.error(f"性能分析过程中出错: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 