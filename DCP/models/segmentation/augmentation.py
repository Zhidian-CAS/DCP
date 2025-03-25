import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2

class SegmentationAugmentation:
    """分割数据增强器"""
    def __init__(self,
                 input_size: Tuple[int, int] = (512, 512),
                 train_mode: bool = True,
                 p: float = 0.5):
        """
        初始化
        Args:
            input_size: 输入大小 (H, W)
            train_mode: 是否为训练模式
            p: 数据增强概率
        """
        self.input_size = input_size
        self.train_mode = train_mode
        self.p = p
        
        # 创建训练和验证的数据增强流水线
        if train_mode:
            self.transform = A.Compose([
                # 几何变换
                A.RandomResizedCrop(
                    height=input_size[0],
                    width=input_size[1],
                    scale=(0.8, 1.2),
                    ratio=(0.75, 1.33),
                    p=p
                ),
                A.Rotate(limit=45, p=p),
                A.HorizontalFlip(p=p),
                A.VerticalFlip(p=p),
                A.ShiftScaleRotate(
                    shift_limit=0.2,
                    scale_limit=0.2,
                    rotate_limit=45,
                    p=p
                ),
                A.ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03,
                    p=p
                ),
                
                # 像素级变换
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=p
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=(3, 7)),
                ], p=p),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                ], p=p),
                
                # 天气模拟
                A.OneOf([
                    A.RandomRain(
                        slant_lower=-10,
                        slant_upper=10,
                        drop_length=20,
                        drop_width=1,
                        drop_color=(200, 200, 200),
                        blur_value=7,
                        brightness_coefficient=0.7,
                        p=1.0
                    ),
                    A.RandomFog(
                        fog_coef_lower=0.3,
                        fog_coef_upper=0.5,
                        alpha_coef=0.08,
                        p=1.0
                    ),
                    A.RandomShadow(
                        num_shadows_lower=1,
                        num_shadows_upper=3,
                        shadow_dimension=5,
                        shadow_roi=(0, 0.5, 1, 1),
                        p=1.0
                    ),
                ], p=p),
                
                # 颜色变换
                A.OneOf([
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=1.0
                    ),
                    A.RGBShift(
                        r_shift_limit=20,
                        g_shift_limit=20,
                        b_shift_limit=20,
                        p=1.0
                    ),
                    A.ChannelShuffle(p=1.0),
                ], p=p),
                
                # 遮挡模拟
                A.OneOf([
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=32,
                        max_width=32,
                        fill_value=0,
                        p=1.0
                    ),
                    A.GridDropout(
                        ratio=0.3,
                        unit_size_min=32,
                        unit_size_max=64,
                        random_offset=True,
                        p=1.0
                    ),
                ], p=p),
                
                # 标准化
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(
                    height=input_size[0],
                    width=input_size[1]
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])
        
    def __call__(self, 
                 image: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> Dict[str, Union[np.ndarray, None]]:
        """
        应用数据增强
        Args:
            image: 输入图像 [H, W, C]
            mask: 分割掩码 [H, W]
        Returns:
            transformed: 增强后的数据字典
        """
        if mask is None:
            transformed = self.transform(image=image)
            return {
                'image': transformed['image'],
                'mask': None
            }
        else:
            transformed = self.transform(image=image, mask=mask)
            return {
                'image': transformed['image'],
                'mask': transformed['mask']
            }
            
    @staticmethod
    def denormalize(image: np.ndarray) -> np.ndarray:
        """
        反归一化图像
        Args:
            image: 归一化的图像 [C, H, W]
        Returns:
            image: 反归一化的图像 [H, W, C]
        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        image = image.transpose(1, 2, 0)
        image = image * std + mean
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        return image
        
    def visualize(self,
                 image: np.ndarray,
                 mask: Optional[np.ndarray] = None,
                 num_samples: int = 5) -> np.ndarray:
        """
        可视化数据增强效果
        Args:
            image: 输入图像 [H, W, C]
            mask: 分割掩码 [H, W]
            num_samples: 可视化样本数
        Returns:
            grid: 可视化网格图像
        """
        rows = []
        for i in range(num_samples):
            transformed = self(image, mask)
            aug_image = self.denormalize(transformed['image'])
            
            if mask is not None:
                aug_mask = transformed['mask']
                # 将掩码转换为彩色图像
                aug_mask_color = np.zeros((*aug_mask.shape, 3), dtype=np.uint8)
                aug_mask_color[aug_mask == 1] = [255, 0, 0]  # 红色表示前景
                
                # 叠加显示
                aug_image = cv2.addWeighted(aug_image, 0.7, aug_mask_color, 0.3, 0)
                
            rows.append(aug_image)
            
        # 创建网格图像
        grid = np.vstack(rows)
        return grid 

    @staticmethod
    def get_preprocessing_transform(input_size: tuple = (512, 512)):
        """
        获取预处理变换
        Args:
            input_size: 输入图像大小
        Returns:
            预处理变换
        """
        return A.Compose([
            A.Resize(
                height=input_size[0],
                width=input_size[1]
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ]) 