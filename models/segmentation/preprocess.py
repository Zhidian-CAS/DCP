import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import shutil
from tqdm import tqdm
import logging
import albumentations as A
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET

class SegmentationPreprocessor:
    """分割数据预处理器"""
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (512, 512),
        num_threads: int = 4,
        save_dir: Optional[str] = None
    ):
        """
        初始化预处理器
        Args:
            input_size: 输入图像大小
            num_threads: 线程数
            save_dir: 保存目录
        """
        self.input_size = input_size
        self.num_threads = num_threads
        self.save_dir = Path(save_dir) if save_dir else None
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 创建保存目录
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建子目录
            self.image_dir = self.save_dir / 'images'
            self.mask_dir = self.save_dir / 'masks'
            self.image_dir.mkdir(exist_ok=True)
            self.mask_dir.mkdir(exist_ok=True)
    
    def resize_image(
        self,
        image: np.ndarray,
        interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """
        调整图像大小
        Args:
            image: 输入图像
            interpolation: 插值方法
        Returns:
            调整后的图像
        """
        return cv2.resize(image, self.input_size, interpolation=interpolation)
    
    def normalize_image(
        self,
        image: np.ndarray,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        归一化图像
        Args:
            image: 输入图像
            mean: 均值
            std: 标准差
        Returns:
            归一化后的图像
        """
        # 默认值
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
            
        # 转换为float32
        image = image.astype(np.float32) / 255.0
        
        # 归一化
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        image = (image - mean) / std
        
        return image
    
    def denormalize_image(
        self,
        image: np.ndarray,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        反归一化图像
        Args:
            image: 输入图像
            mean: 均值
            std: 标准差
        Returns:
            反归一化后的图像
        """
        # 默认值
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
            
        # 反归一化
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        image = image * std + mean
        
        # 转换回uint8
        image = (image * 255).clip(0, 255).astype(np.uint8)
        
        return image
    
    def augment_image(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        augmentation: Optional[A.Compose] = None
    ) -> Dict[str, np.ndarray]:
        """
        数据增强
        Args:
            image: 输入图像
            mask: 输入掩码
            augmentation: 数据增强配置
        Returns:
            增强后的图像和掩码
        """
        # 默认增强配置
        if augmentation is None:
            augmentation = A.Compose([
                A.RandomBrightnessContrast(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.GaussNoise(p=0.3),
                A.Blur(blur_limit=3, p=0.3),
            ])
        
        # 应用增强
        if mask is not None:
            transformed = augmentation(image=image, mask=mask)
            return {
                'image': transformed['image'],
                'mask': transformed['mask']
            }
        else:
            transformed = augmentation(image=image)
            return {'image': transformed['image']}
    
    def process_single_image(
        self,
        image_path: Union[str, Path],
        mask_path: Optional[Union[str, Path]] = None,
        augment: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        处理单张图像
        Args:
            image_path: 图像路径
            mask_path: 掩码路径
            augment: 是否进行数据增强
        Returns:
            处理后的图像和掩码
        """
        # 加载图像
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载掩码
        mask = None
        if mask_path and Path(mask_path).exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # 调整大小
        image = self.resize_image(image)
        if mask is not None:
            mask = self.resize_image(mask, interpolation=cv2.INTER_NEAREST)
        
        # 数据增强
        if augment:
            transformed = self.augment_image(image, mask)
            image = transformed['image']
            if mask is not None:
                mask = transformed['mask']
        
        # 归一化
        image = self.normalize_image(image)
        
        return {'image': image, 'mask': mask}
    
    def process_dataset(
        self,
        image_dir: Union[str, Path],
        mask_dir: Optional[Union[str, Path]] = None,
        augment: bool = False,
        save: bool = True
    ) -> List[Dict[str, np.ndarray]]:
        """
        处理数据集
        Args:
            image_dir: 图像目录
            mask_dir: 掩码目录
            augment: 是否进行数据增强
            save: 是否保存处理后的数据
        Returns:
            处理后的数据列表
        """
        # 获取图像路径
        image_paths = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
        self.logger.info(f'找到 {len(image_paths)} 张图像')
        
        # 多线程处理
        processed_data = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for image_path in image_paths:
                # 获取对应的掩码路径
                mask_path = None
                if mask_dir:
                    mask_path = Path(mask_dir) / image_path.name
                
                # 提交任务
                future = executor.submit(
                    self.process_single_image,
                    image_path,
                    mask_path,
                    augment
                )
                futures.append((image_path, future))
            
            # 获取结果
            for image_path, future in tqdm(futures, desc='Processing images'):
                try:
                    result = future.result()
                    processed_data.append(result)
                    
                    # 保存处理后的数据
                    if save and self.save_dir:
                        # 保存图像
                        save_image = self.denormalize_image(result['image'])
                        cv2.imwrite(
                            str(self.image_dir / image_path.name),
                            cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
                        )
                        
                        # 保存掩码
                        if result['mask'] is not None:
                            cv2.imwrite(
                                str(self.mask_dir / image_path.name),
                                result['mask']
                            )
                except Exception as e:
                    self.logger.error(f'处理图像 {image_path} 时出错: {str(e)}')
        
        return processed_data
    
    def convert_to_coco(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        save_path: Union[str, Path]
    ) -> None:
        """
        转换为COCO格式
        Args:
            image_dir: 图像目录
            mask_dir: 掩码目录
            save_path: 保存路径
        """
        # 创建COCO格式数据
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': 0, 'name': 'background'},
                {'id': 1, 'name': 'foreground'}
            ]
        }
        
        # 获取图像路径
        image_paths = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
        
        # 处理每张图像
        for image_id, image_path in enumerate(tqdm(image_paths, desc='Converting to COCO')):
            # 加载图像
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]
            
            # 添加图像信息
            coco_data['images'].append({
                'id': image_id,
                'file_name': image_path.name,
                'height': height,
                'width': width
            })
            
            # 加载掩码
            mask_path = Path(mask_dir) / image_path.name
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                # 添加标注信息
                from pycocotools import mask as mask_util
                rle = mask_util.encode(np.asfortranarray(mask))
                area = float(mask_util.area(rle))
                bbox = list(map(float, mask_util.toBbox(rle)))
                
                coco_data['annotations'].append({
                    'id': image_id,
                    'image_id': image_id,
                    'category_id': 1,
                    'segmentation': rle,
                    'area': area,
                    'bbox': bbox,
                    'iscrowd': 0
                })
        
        # 保存COCO格式数据
        with open(save_path, 'w') as f:
            json.dump(coco_data, f)
        
        self.logger.info(f'COCO格式数据已保存到: {save_path}')
    
    def convert_to_voc(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        save_dir: Union[str, Path]
    ) -> None:
        """
        转换为VOC格式
        Args:
            image_dir: 图像目录
            mask_dir: 掩码目录
            save_dir: 保存目录
        """
        # 创建VOC格式目录
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        image_save_dir = save_dir / 'JPEGImages'
        mask_save_dir = save_dir / 'SegmentationClass'
        anno_save_dir = save_dir / 'Annotations'
        image_save_dir.mkdir(exist_ok=True)
        mask_save_dir.mkdir(exist_ok=True)
        anno_save_dir.mkdir(exist_ok=True)
        
        # 获取图像路径
        image_paths = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
        
        # 处理每张图像
        for image_path in tqdm(image_paths, desc='Converting to VOC'):
            # 复制图像
            shutil.copy2(image_path, image_save_dir / image_path.name)
            
            # 复制掩码
            mask_path = Path(mask_dir) / image_path.name
            if mask_path.exists():
                shutil.copy2(mask_path, mask_save_dir / image_path.name)
                
                # 创建XML标注
                image = cv2.imread(str(image_path))
                height, width = image.shape[:2]
                
                root = ET.Element('annotation')
                ET.SubElement(root, 'filename').text = image_path.name
                
                size = ET.SubElement(root, 'size')
                ET.SubElement(size, 'width').text = str(width)
                ET.SubElement(size, 'height').text = str(height)
                ET.SubElement(size, 'depth').text = '3'
                
                # 保存XML
                tree = ET.ElementTree(root)
                tree.write(str(anno_save_dir / f'{image_path.stem}.xml'))
        
        self.logger.info(f'VOC格式数据已保存到: {save_dir}') 