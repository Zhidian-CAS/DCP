#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在线学习功能演示脚本
演示如何使用单细胞克隆系统的在线学习功能
"""

import os
import sys
import time
import logging
import argparse
from typing import Dict, Any
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from main import CloneSystem

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="单细胞克隆系统在线学习演示")
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="数据目录，包含图像和标注"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/online_learning_demo",
        help="输出目录"
    )
    
    parser.add_argument(
        "--plot-metrics",
        action="store_true",
        help="是否绘制指标变化图"
    )
    
    return parser.parse_args()

def setup_logging(output_dir: str):
    """配置日志"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "demo.log")),
            logging.StreamHandler()
        ]
    )

def load_feedback_data(data_dir: str) -> Dict[str, Dict[str, str]]:
    """加载反馈数据"""
    feedback_file = os.path.join(data_dir, "feedback.txt")
    feedback_data = {}
    
    if not os.path.exists(feedback_file):
        logging.warning(f"反馈文件不存在: {feedback_file}")
        return feedback_data
        
    try:
        with open(feedback_file, "r") as f:
            for line in f:
                # 格式: plate_id,colony_id,true_label
                plate_id, colony_id, true_label = line.strip().split(",")
                if plate_id not in feedback_data:
                    feedback_data[plate_id] = {}
                feedback_data[plate_id][colony_id] = true_label
                
        logging.info(f"已加载 {len(feedback_data)} 个培养板的反馈数据")
        return feedback_data
        
    except Exception as e:
        logging.error(f"加载反馈数据失败: {e}")
        return {}

def plot_metrics(metrics_history: Dict[str, list], output_dir: str):
    """绘制指标变化图"""
    plt.figure(figsize=(12, 8))
    
    # 1. 准确率变化
    plt.subplot(2, 2, 1)
    plt.plot(metrics_history['accuracy'], 'b-', label='准确率')
    plt.title('准确率变化')
    plt.xlabel('迭代次数')
    plt.ylabel('准确率')
    plt.grid(True)
    plt.legend()
    
    # 2. 置信度变化
    plt.subplot(2, 2, 2)
    plt.plot(metrics_history['mean_confidence'], 'g-', label='平均置信度')
    plt.title('置信度变化')
    plt.xlabel('迭代次数')
    plt.ylabel('置信度')
    plt.grid(True)
    plt.legend()
    
    # 3. 类别分布变化
    plt.subplot(2, 2, 3)
    class_names = list(metrics_history['class_distribution'][0].keys())
    for class_name in class_names:
        values = [dist[class_name] for dist in metrics_history['class_distribution']]
        plt.plot(values, label=class_name)
    plt.title('类别分布变化')
    plt.xlabel('迭代次数')
    plt.ylabel('比例')
    plt.grid(True)
    plt.legend()
    
    # 4. 反馈率变化
    plt.subplot(2, 2, 4)
    plt.plot(metrics_history['feedback_ratio'], 'r-', label='反馈率')
    plt.title('反馈率变化')
    plt.xlabel('迭代次数')
    plt.ylabel('反馈率')
    plt.grid(True)
    plt.legend()
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics.png'))
    plt.close()
    
    logging.info(f"指标变化图已保存到: {output_dir}/metrics.png")

def main():
    """主函数"""
    # 1. 解析参数
    args = parse_args()
    
    # 2. 配置日志
    setup_logging(args.output_dir)
    
    # 3. 加载反馈数据
    feedback_data = load_feedback_data(args.data_dir)
    if not feedback_data:
        logging.error("未找到反馈数据，演示终止")
        return
        
    try:
        # 4. 初始化系统
        system = CloneSystem(args.config)
        
        # 5. 启用在线学习
        system.toggle_online_learning(True)
        
        # 6. 记录指标历史
        metrics_history = {
            'accuracy': [],
            'mean_confidence': [],
            'class_distribution': [],
            'feedback_ratio': []
        }
        
        # 7. 处理每个培养板
        for plate_id, feedback in feedback_data.items():
            # 分析图像并获取结果
            results = system.scan_plate(plate_id, feedback)
            
            if 'error' in results:
                logging.error(f"处理培养板 {plate_id} 失败: {results['error']}")
                continue
                
            # 记录指标
            stats = results['stats']
            metrics_history['accuracy'].append(stats.get('accuracy', 0))
            metrics_history['mean_confidence'].append(stats['mean_confidence'])
            metrics_history['class_distribution'].append(stats['class_distribution'])
            metrics_history['feedback_ratio'].append(stats['feedback_ratio'])
            
            # 输出当前状态
            logging.info(f"\n培养板 {plate_id} 分析完成:")
            logging.info(f"- 准确率: {stats.get('accuracy', 0):.3f}")
            logging.info(f"- 平均置信度: {stats['mean_confidence']:.3f}")
            logging.info(f"- 反馈率: {stats['feedback_ratio']:.3f}")
            logging.info("- 类别分布:")
            for class_name, ratio in stats['class_distribution'].items():
                logging.info(f"  * {class_name}: {ratio:.3f}")
                
            # 获取模型信息
            model_info = system.get_model_info()
            logging.info("\n模型状态:")
            logging.info(f"- 版本: {model_info['version']}")
            logging.info(f"- 缓冲区样本数: {model_info['samples_in_buffer']}")
            logging.info(f"- 距离上次更新的样本数: {model_info['samples_since_update']}")
            
            # 暂停一下，方便观察
            time.sleep(1)
            
        # 8. 绘制指标变化图
        if args.plot_metrics:
            plot_metrics(metrics_history, args.output_dir)
            
        logging.info("\n演示完成!")
        
    except KeyboardInterrupt:
        logging.info("\n演示被用户中断")
    except Exception as e:
        logging.error(f"演示过程出错: {e}")
    finally:
        # 保存最终的模型状态
        system.force_model_update()

if __name__ == "__main__":
    main() 