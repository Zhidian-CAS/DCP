import argparse
import logging
from pathlib import Path
import torch
import pandas as pd
import yaml
from models.segmentation.train_visualizer import TrainVisualizer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='可视化训练过程')
    
    # 输入参数
    parser.add_argument('--train-metrics', type=str, required=True, help='训练指标CSV文件路径')
    parser.add_argument('--val-metrics', type=str, help='验证指标CSV文件路径')
    
    # 可视化参数
    parser.add_argument('--save-frequency', type=int, default=1, help='保存频率（每多少个epoch保存一次）')
    parser.add_argument('--show', action='store_true', help='是否显示图表')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='training_visualization', help='输出目录')
    
    return parser.parse_args()

def setup_logging(output_dir: Path):
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'visualize_training.log'),
            logging.StreamHandler()
        ]
    )

def load_metrics(csv_path: str) -> pd.DataFrame:
    """加载指标数据"""
    return pd.read_csv(csv_path)

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
        # 创建可视化器
        visualizer = TrainVisualizer(
            output_dir=output_dir,
            save_frequency=args.save_frequency,
            show=args.show
        )
        
        # 加载训练指标
        logger.info(f"加载训练指标: {args.train_metrics}")
        train_df = load_metrics(args.train_metrics)
        
        # 加载验证指标
        val_df = None
        if args.val_metrics:
            logger.info(f"加载验证指标: {args.val_metrics}")
            val_df = load_metrics(args.val_metrics)
            
        # 更新指标
        for _, row in train_df.iterrows():
            train_metrics = {
                'loss': row['loss'],
                'dice': row['dice'],
                'iou': row['iou']
            }
            
            val_metrics = None
            if val_df is not None:
                val_row = val_df[val_df['epoch'] == row['epoch']]
                if not val_row.empty:
                    val_metrics = {
                        'loss': val_row['loss'].iloc[0],
                        'dice': val_row['dice'].iloc[0],
                        'iou': val_row['iou'].iloc[0]
                    }
                    
            learning_rate = row.get('learning_rate', None)
            
            visualizer.update_metrics(
                epoch=int(row['epoch']),
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=learning_rate
            )
            
        # 创建训练报告
        visualizer.create_training_report()
        
        logger.info(f"可视化完成！结果已保存至: {output_dir}")
        
    except Exception as e:
        logger.error(f"可视化过程中出错: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 