import cv2
import numpy as np
from config.config import Config

class ImageProcessor:
    @staticmethod
    def measure_focus(image, roi=None):
        """测量图像清晰度
        使用拉普拉斯算子计算图像的清晰度得分
        """
        if roi is not None:
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    @staticmethod
    def auto_focus(camera, stage, roi=None):
        """自动聚焦
        使用二分搜索方法找到最佳焦平面
        """
        config = Config.AUTOFOCUS
        best_focus = -float('inf')
        best_z = None
        
        # 粗调
        for z in range(config['start_position'], 
                      config['end_position'], 
                      config['step_size']):
            stage.move_to(z=z)
            image = camera.capture()
            focus_score = ImageProcessor.measure_focus(image, roi)
            
            if focus_score > best_focus:
                best_focus = focus_score
                best_z = z
                
        # 精调
        if best_z is not None:
            for z in range(best_z - config['step_size'],
                         best_z + config['step_size'],
                         config['fine_step_size']):
                stage.move_to(z=z)
                image = camera.capture()
                focus_score = ImageProcessor.measure_focus(image, roi)
                
                if focus_score > best_focus:
                    best_focus = focus_score
                    best_z = z
                    
        return best_z
    
    @staticmethod
    def detect_wells(image):
        """检测腔室
        使用传统图像处理方法检测腔室位置
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 自适应阈值分割
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2)
            
        # 查找轮廓
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
            
        wells = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if (Config.IMAGE_PROCESS['min_well_area'] < area < 
                Config.IMAGE_PROCESS['max_well_area']):
                x, y, w, h = cv2.boundingRect(contour)
                wells.append({
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2),
                    'area': area
                })
                
        return wells
    
    @staticmethod
    def sort_wells_s_shape(wells, num_cols):
        """S型排序腔室"""
        # 按y坐标分组
        rows = {}
        for well in wells:
            y = well['center'][1]
            row_idx = y // 100  # 假设每行间距约100像素
            if row_idx not in rows:
                rows[row_idx] = []
            rows[row_idx].append(well)
            
        # S型排序
        sorted_wells = []
        for row_idx in sorted(rows.keys()):
            row_wells = rows[row_idx]
            # 偶数行反向
            if row_idx % 2 == 0:
                row_wells.sort(key=lambda x: x['center'][0])
            else:
                row_wells.sort(key=lambda x: x['center'][0], reverse=True)
            sorted_wells.extend(row_wells)
            
        return sorted_wells
    
    @staticmethod
    def extract_well_roi(image, bbox, padding=10):
        """从图像中提取腔室ROI"""
        x, y, w, h = bbox
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        return image[y1:y2, x1:x2]
    
    @staticmethod
    def calculate_colony_area(mask):
        """计算菌落面积"""
        return np.sum(mask > 0)
    
    @staticmethod
    def calculate_gray_value(image, mask):
        """计算菌落区域的平均灰度值"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray[mask > 0]) 