import cv2
import numpy as np
from typing import Tuple, Optional, Dict

class ImageProcessor:
    """图像处理工具类"""
    
    @staticmethod
    def preprocess_image(
        image: np.ndarray,
        resize: Optional[Tuple[int, int]] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """预处理图像"""
        if resize:
            image = cv2.resize(image, resize)
        
        if normalize:
            image = image.astype(np.float32) / 255.0
            
        return image
    
    @staticmethod
    def extract_features(
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """提取图像特征"""
        features = {}
        
        # 基本统计特征
        if mask is not None:
            roi = cv2.bitwise_and(image, image, mask=mask)
        else:
            roi = image
            
        features['mean'] = float(np.mean(roi))
        features['std'] = float(np.std(roi))
        features['min'] = float(np.min(roi))
        features['max'] = float(np.max(roi))
        
        # 纹理特征
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        glcm = ImageProcessor._calculate_glcm(gray)
        features.update(ImageProcessor._extract_glcm_features(glcm))
        
        return features
    
    @staticmethod
    def _calculate_glcm(image: np.ndarray) -> np.ndarray:
        """计算灰度共生矩阵"""
        glcm = np.zeros((256, 256))
        h, w = image.shape
        
        for i in range(h-1):
            for j in range(w-1):
                glcm[image[i,j], image[i,j+1]] += 1
                
        glcm = glcm / glcm.sum()
        return glcm
    
    @staticmethod
    def _extract_glcm_features(glcm: np.ndarray) -> Dict[str, float]:
        """从灰度共生矩阵提取特征"""
        features = {}
        
        # 对比度
        i, j = np.ogrid[0:256, 0:256]
        features['contrast'] = float(np.sum(glcm * ((i-j)**2)))
        
        # 能量
        features['energy'] = float(np.sum(glcm**2))
        
        # 同质性
        features['homogeneity'] = float(np.sum(glcm / (1 + (i-j)**2)))
        
        # 相关性
        mu_i = np.sum(i * glcm.sum(axis=1))
        mu_j = np.sum(j * glcm.sum(axis=0))
        sigma_i = np.sqrt(np.sum((i - mu_i)**2 * glcm.sum(axis=1)))
        sigma_j = np.sqrt(np.sum((j - mu_j)**2 * glcm.sum(axis=0)))
        
        if sigma_i * sigma_j != 0:
            features['correlation'] = float(np.sum(glcm * (i - mu_i) * (j - mu_j)) / (sigma_i * sigma_j))
        else:
            features['correlation'] = 0.0
            
        return features 