import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import logging
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import time
from functools import wraps

def setup_seed(seed: int):
    """
    设置随机种子
    Args:
        seed: 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_transforms(config: Dict[str, bool]) -> A.Compose:
    """
    获取数据增强转换
    Args:
        config: 数据增强配置
    Returns:
        transforms: 数据增强转换
    """
    transforms = []
    
    # 基础转换
    transforms.extend([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # 训练时的数据增强
    if config.get('horizontal_flip', False):
        transforms.append(A.HorizontalFlip(p=0.5))
    if config.get('vertical_flip', False):
        transforms.append(A.VerticalFlip(p=0.5))
    if config.get('rotation', False):
        transforms.append(A.Rotate(limit=15, p=0.5))
    if config.get('color_jitter', False):
        transforms.append(A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5))
    if config.get('random_scale', False):
        transforms.append(A.RandomScale(scale_limit=0.2, p=0.5))
        
    return A.Compose(transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    加载图像
    Args:
        image_path: 图像路径
    Returns:
        image: 图像数组
    """
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_annotation(annotation_path: Union[str, Path]) -> Dict:
    """
    加载标注文件
    Args:
        annotation_path: 标注文件路径
    Returns:
        annotation: 标注字典
    """
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    return annotation

def save_annotation(annotation: Dict, save_path: Union[str, Path]):
    """
    保存标注文件
    Args:
        annotation: 标注字典
        save_path: 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(annotation, f, indent=4)

def convert_to_coco_format(annotations: List[Dict], image_dir: Union[str, Path]) -> Dict:
    """
    转换为COCO格式
    Args:
        annotations: 标注列表
        image_dir: 图像目录
    Returns:
        coco_format: COCO格式数据
    """
    coco_format = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # 添加类别
    categories = set()
    for ann in annotations:
        for obj in ann['objects']:
            categories.add(obj['class'])
    for i, category in enumerate(sorted(categories)):
        coco_format['categories'].append({
            'id': i + 1,
            'name': category,
            'supercategory': 'none'
        })
        
    # 添加图像和标注
    ann_id = 1
    for i, ann in enumerate(annotations):
        image_path = Path(image_dir) / ann['image']
        image = Image.open(image_path)
        
        # 添加图像信息
        coco_format['images'].append({
            'id': i + 1,
            'file_name': ann['image'],
            'height': image.height,
            'width': image.width
        })
        
        # 添加标注信息
        for obj in ann['objects']:
            category_id = next(cat['id'] for cat in coco_format['categories'] if cat['name'] == obj['class'])
            bbox = obj['bbox']
            
            coco_format['annotations'].append({
                'id': ann_id,
                'image_id': i + 1,
                'category_id': category_id,
                'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                'iscrowd': 0
            })
            ann_id += 1
            
    return coco_format

def convert_to_yolo_format(annotations: List[Dict], image_dir: Union[str, Path], save_dir: Union[str, Path]):
    """
    转换为YOLO格式
    Args:
        annotations: 标注列表
        image_dir: 图像目录
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    image_dir = Path(image_dir)
    
    # 创建保存目录
    (save_dir / 'images').mkdir(parents=True, exist_ok=True)
    (save_dir / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 获取类别
    categories = set()
    for ann in annotations:
        for obj in ann['objects']:
            categories.add(obj['class'])
    categories = sorted(list(categories))
    
    # 保存类别文件
    with open(save_dir / 'classes.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(categories))
        
    # 转换标注
    for ann in annotations:
        image_path = image_dir / ann['image']
        image = Image.open(image_path)
        width, height = image.size
        
        # 复制图像
        save_image_path = save_dir / 'images' / ann['image']
        image.save(save_image_path)
        
        # 转换标注
        label_path = save_dir / 'labels' / f"{Path(ann['image']).stem}.txt"
        with open(label_path, 'w', encoding='utf-8') as f:
            for obj in ann['objects']:
                category_id = categories.index(obj['class'])
                bbox = obj['bbox']
                
                # 转换为YOLO格式 (x_center, y_center, width, height)
                x_center = (bbox[0] + bbox[2]) / (2 * width)
                y_center = (bbox[1] + bbox[3]) / (2 * height)
                w = (bbox[2] - bbox[0]) / width
                h = (bbox[3] - bbox[1]) / height
                
                f.write(f"{category_id} {x_center} {y_center} {w} {h}\n")

def convert_to_voc_format(annotations: List[Dict], image_dir: Union[str, Path], save_dir: Union[str, Path]):
    """
    转换为VOC格式
    Args:
        annotations: 标注列表
        image_dir: 图像目录
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    image_dir = Path(image_dir)
    
    # 创建保存目录
    (save_dir / 'JPEGImages').mkdir(parents=True, exist_ok=True)
    (save_dir / 'Annotations').mkdir(parents=True, exist_ok=True)
    (save_dir / 'ImageSets' / 'Main').mkdir(parents=True, exist_ok=True)
    
    # 获取类别
    categories = set()
    for ann in annotations:
        for obj in ann['objects']:
            categories.add(obj['class'])
    categories = sorted(list(categories))
    
    # 保存类别文件
    with open(save_dir / 'classes.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(categories))
        
    # 转换标注
    for ann in annotations:
        image_path = image_dir / ann['image']
        image = Image.open(image_path)
        
        # 复制图像
        save_image_path = save_dir / 'JPEGImages' / ann['image']
        image.save(save_image_path)
        
        # 转换标注
        annotation_path = save_dir / 'Annotations' / f"{Path(ann['image']).stem}.xml"
        with open(annotation_path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write('<annotation>\n')
            f.write(f'  <filename>{ann["image"]}</filename>\n')
            f.write(f'  <size>\n')
            f.write(f'    <width>{image.width}</width>\n')
            f.write(f'    <height>{image.height}</height>\n')
            f.write(f'    <depth>3</depth>\n')
            f.write(f'  </size>\n')
            
            for obj in ann['objects']:
                bbox = obj['bbox']
                f.write(f'  <object>\n')
                f.write(f'    <name>{obj["class"]}</name>\n')
                f.write(f'    <bndbox>\n')
                f.write(f'      <xmin>{int(bbox[0])}</xmin>\n')
                f.write(f'      <ymin>{int(bbox[1])}</ymin>\n')
                f.write(f'      <xmax>{int(bbox[2])}</xmax>\n')
                f.write(f'      <ymax>{int(bbox[3])}</ymax>\n')
                f.write(f'    </bndbox>\n')
                f.write(f'  </object>\n')
                
            f.write('</annotation>\n')
            
    # 创建数据集划分文件
    image_ids = [Path(ann['image']).stem for ann in annotations]
    train_size = int(0.8 * len(image_ids))
    train_ids = random.sample(image_ids, train_size)
    val_ids = [id for id in image_ids if id not in train_ids]
    
    with open(save_dir / 'ImageSets' / 'Main' / 'train.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_ids))
    with open(save_dir / 'ImageSets' / 'Main' / 'val.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_ids))

def convert_model_to_onnx(model: nn.Module,
                         save_path: str,
                         input_shape: Tuple[int, ...] = (1, 3, 224, 224)):
    """
    将模型转换为ONNX格式
    Args:
        model: PyTorch模型
        save_path: 保存路径
        input_shape: 输入形状
    """
    # 创建示例输入
    dummy_input = torch.randn(input_shape)
    
    # 导出模型
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

def convert_model_to_torchscript(model: nn.Module,
                                save_path: str,
                                input_shape: Tuple[int, ...] = (1, 3, 224, 224)):
    """
    将模型转换为TorchScript格式
    Args:
        model: PyTorch模型
        save_path: 保存路径
        input_shape: 输入形状
    """
    # 创建示例输入
    dummy_input = torch.randn(input_shape)
    
    # 导出模型
    traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, save_path)

def count_parameters(model: nn.Module) -> int:
    """
    计算模型参数量
    Args:
        model: PyTorch模型
    Returns:
        num_params: 参数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model: nn.Module) -> float:
    """
    计算模型大小
    Args:
        model: PyTorch模型
    Returns:
        size_mb: 模型大小(MB)
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def timing_decorator(func):
    """
    计时装饰器
    Args:
        func: 被装饰的函数
    Returns:
        wrapper: 包装后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper

def setup_logging(log_dir: Optional[str] = None,
                 log_file: Optional[str] = None,
                 level: str = 'INFO'):
    """
    设置日志
    Args:
        log_dir: 日志目录
        log_file: 日志文件
        level: 日志级别
    """
    # 创建日志目录
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                log_dir / log_file
                if log_dir and log_file
                else None
            )
        ]
    ) 