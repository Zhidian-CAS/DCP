import cv2
import numpy as np
from config.config import Config

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(Config.CAMERA_ID)
        self.setup_camera()
        
    def setup_camera(self):
        """设置相机参数"""
        self.cap.set(cv2.CAP_PROP_EXPOSURE, Config.CAMERA_EXPOSURE)
        self.cap.set(cv2.CAP_PROP_GAIN, Config.CAMERA_GAIN)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.IMAGE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.IMAGE_HEIGHT)
        
    def capture(self):
        """捕获一帧图像"""
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to capture image")
        return frame
    
    def capture_multiple(self, num_frames=5):
        """捕获多帧图像并返回平均值以减少噪声"""
        frames = []
        for _ in range(num_frames):
            frame = self.capture()
            frames.append(frame)
        return np.mean(frames, axis=0).astype(np.uint8)
    
    def save_image(self, frame, path):
        """保存图像"""
        cv2.imwrite(path, frame)
        
    def close(self):
        """关闭相机"""
        self.cap.release()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 