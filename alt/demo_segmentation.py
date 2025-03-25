import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
from typing import List, Optional, Tuple
import time
from tqdm import tqdm

from models.segmentation.deploy import SegmentationDeployer
from models.segmentation.visualization import SegmentationVisualizer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='分割模型演示')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, required=True, help='模型路径')
    parser.add_argument('--input-size', type=int, nargs=2, default=[512, 512], help='输入图像大小')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--use-onnx', action='store_true', help='是否使用ONNX模型')
    
    # 输入参数
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image-path', type=str, help='输入图像路径')
    group.add_argument('--video-path', type=str, help='输入视频路径')
    group.add_argument('--camera-id', type=int, help='摄像头ID')
    
    # 可视化参数
    parser.add_argument('--class-names', type=str, nargs='+', default=['background', 'colony'], help='类别名称')
    parser.add_argument('--output-dir', type=str, default='demo_outputs', help='输出目录')
    parser.add_argument('--show', action='store_true', help='是否显示结果')
    parser.add_argument('--save-video', action='store_true', help='是否保存视频')
    
    return parser.parse_args()

def setup_logging(output_dir: Path):
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'demo.log'),
            logging.StreamHandler()
        ]
    )

def process_frame(
    frame: np.ndarray,
    deployer: SegmentationDeployer,
    visualizer: SegmentationVisualizer
) -> np.ndarray:
    """
    处理单帧图像
    Args:
        frame: 输入图像
        deployer: 部署器
        visualizer: 可视化器
    Returns:
        处理后的图像
    """
    # 预测
    pred_mask = deployer.predict(frame)
    
    # 可视化
    vis_frame = visualizer.visualize_prediction(
        image=frame,
        pred_mask=pred_mask
    )
    
    return vis_frame

def process_image(
    image_path: str,
    deployer: SegmentationDeployer,
    visualizer: SegmentationVisualizer,
    output_dir: Path,
    show: bool = False
) -> None:
    """
    处理图像
    Args:
        image_path: 图像路径
        deployer: 部署器
        visualizer: 可视化器
        output_dir: 输出目录
        show: 是否显示结果
    """
    # 加载图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 处理图像
    vis_image = process_frame(image, deployer, visualizer)
    
    # 保存结果
    save_path = output_dir / f'pred_{Path(image_path).name}'
    cv2.imwrite(
        str(save_path),
        cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    )
    
    # 显示结果
    if show:
        cv2.imshow('Prediction', cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def process_video(
    video_path: Optional[str],
    camera_id: Optional[int],
    deployer: SegmentationDeployer,
    visualizer: SegmentationVisualizer,
    output_dir: Path,
    show: bool = False,
    save_video: bool = False
) -> None:
    """
    处理视频
    Args:
        video_path: 视频路径
        camera_id: 摄像头ID
        deployer: 部署器
        visualizer: 可视化器
        output_dir: 输出目录
        show: 是否显示结果
        save_video: 是否保存视频
    """
    # 打开视频
    if video_path:
        cap = cv2.VideoCapture(video_path)
        source_name = Path(video_path).name
    else:
        cap = cv2.VideoCapture(camera_id)
        source_name = f'camera_{camera_id}'
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 创建视频写入器
    writer = None
    if save_video:
        save_path = str(output_dir / f'pred_{source_name}')
        writer = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
    
    # 处理视频
    try:
        while cap.isOpened():
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vis_frame = process_frame(frame, deployer, visualizer)
            
            # 保存视频
            if writer:
                writer.write(cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
            
            # 显示结果
            if show:
                cv2.imshow('Prediction', cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
                
                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    finally:
        # 释放资源
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

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
        input_size=tuple(args.input_size)
    )
    
    # 创建可视化器
    visualizer = SegmentationVisualizer(
        class_names=args.class_names,
        save_dir=output_dir / 'visualizations'
    )
    
    # 处理输入
    logger.info('开始处理...')
    try:
        if args.image_path:
            # 处理图像
            process_image(
                args.image_path,
                deployer,
                visualizer,
                output_dir,
                args.show
            )
        else:
            # 处理视频
            process_video(
                args.video_path,
                args.camera_id,
                deployer,
                visualizer,
                output_dir,
                args.show,
                args.save_video
            )
    except Exception as e:
        logger.error(f'处理时出错: {str(e)}')
    
    logger.info('处理完成！')

if __name__ == '__main__':
    main() 