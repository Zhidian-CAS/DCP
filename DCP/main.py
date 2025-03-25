import os
import sys
import time
import json
import yaml
import logging
import threading
import multiprocessing
import cv2
import numpy as np
from queue import Queue
from typing import Dict, List, Optional, Union, Any, Tuple, Protocol
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict, deque
import signal
import atexit
from abc import ABC, abstractmethod

@dataclass
class WellData:
    """孔板数据类"""
    id: int
    position: Tuple[int, int]
    image: np.ndarray
    mask: Optional[np.ndarray] = None
    colony_count: int = 0
    colony_sizes: List[float] = None
    mean_intensity: float = 0.0
    status: str = "pending"
    texture_features: Dict[str, float] = None
    shape_features: Dict[str, float] = None
    growth_rate: float = 0.0
    viability_score: float = 0.0
    colony_distribution: Dict[str, float] = None
    classification_results: Dict[str, Any] = None  # 添加分类结果字段

@dataclass
class SystemConfig:
    """系统配置数据类"""
    root_dir: Path
    config_file: Path
    log_dir: Path
    output_dir: Path
    max_workers: int
    buffer_size: int
    timeout: float
    debug: bool
    camera_config: Dict[str, Any]
    stage_config: Dict[str, Any]
    model_config: Dict[str, Any]
    processing_config: Dict[str, Any]

class ImageProcessor(Protocol):
    """图像处理接口"""
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        ...
        
    def detect_wells(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """检测孔板位置"""
        ...
        
    def segment_colonies(self, image: np.ndarray) -> np.ndarray:
        """分割菌落"""
        ...
        
    def analyze_colony(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """分析菌落特征"""
        ...

class HardwareController(ABC):
    """硬件控制器基类"""
    @abstractmethod
    def initialize(self) -> bool:
        """初始化硬件"""
        pass
        
    @abstractmethod
    def shutdown(self) -> None:
        """关闭硬件"""
        pass
        
    @abstractmethod
    def check_status(self) -> bool:
        """检查硬件状态"""
        pass

class CameraController(HardwareController):
    """相机控制器"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.camera = None
        self._lock = threading.Lock()
        
    def initialize(self) -> bool:
        try:
            self.camera = cv2.VideoCapture(self.config['device_id'])
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['height'])
            return self.camera.isOpened()
        except Exception as e:
            logging.error(f"相机初始化失败: {e}")
            return False
            
    def shutdown(self) -> None:
        if self.camera:
            self.camera.release()
            
    def check_status(self) -> bool:
        return self.camera and self.camera.isOpened()
        
    def capture(self) -> Optional[np.ndarray]:
        """捕获图像"""
        with self._lock:
            if not self.check_status():
                return None
            ret, frame = self.camera.read()
            return frame if ret else None
            
    def auto_focus(self) -> bool:
        """自动对焦"""
        # 实现自动对焦算法
        return True

class StageController(HardwareController):
    """载物台控制器"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stage = None
        self._lock = threading.Lock()
        self._position = {'x': 0, 'y': 0, 'z': 0}
        
    def initialize(self) -> bool:
        try:
            # 初始化载物台硬件
            self.stage = True  # 实际应该初始化真实的载物台设备
            return True
        except Exception as e:
            logging.error(f"载物台初始化失败: {e}")
            return False
            
    def shutdown(self) -> None:
        if self.stage:
            # 关闭载物台硬件
            pass
            
    def check_status(self) -> bool:
        return self.stage is not None
        
    def move_to(self, x: float = None, y: float = None, z: float = None) -> bool:
        """移动到指定位置"""
        with self._lock:
            if not self.check_status():
                return False
            if x is not None:
                self._position['x'] = x
            if y is not None:
                self._position['y'] = y
            if z is not None:
                self._position['z'] = z
            # 实际控制载物台移动
            return True
            
    def get_position(self) -> Dict[str, float]:
        """获取当前位置"""
        return dict(self._position)

class ColonyClassifier:
    """菌落分类器"""
    def __init__(self, model_config: Dict[str, Any]):
        self.config = model_config
        self.classifier = self._load_classifier()
        self.feature_scaler = self._load_scaler()
        self.pca = self._load_pca()
        
        # 设置分类参数
        self.feature_importance_threshold = self.config.get('feature_importance_threshold', 0.01)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.8)
        self.use_ensemble = self.config.get('use_ensemble', True)
        
        # 在线学习参数
        self.online_learning_enabled = self.config.get('online_learning_enabled', True)
        self.min_samples_for_update = self.config.get('min_samples_for_update', 10)
        self.max_buffer_size = self.config.get('max_buffer_size', 1000)
        self.update_interval = self.config.get('update_interval', 100)
        
        # 在线学习缓冲区
        self._training_buffer = []
        self._samples_since_update = 0
        self._model_version = 0
        self._update_lock = threading.Lock()
        
    def _load_classifier(self):
        """加载分类器模型"""
        try:
            import joblib
            model_path = self.config.get('classifier_model_path', 'models/colony_classifier.joblib')
            return joblib.load(model_path)
        except Exception as e:
            logging.error(f"分类器加载失败: {e}")
            return None
            
    def _load_scaler(self):
        """加载特征缩放器"""
        try:
            import joblib
            scaler_path = self.config.get('scaler_path', 'models/feature_scaler.joblib')
            return joblib.load(scaler_path)
        except Exception as e:
            logging.error(f"特征缩放器加载失败: {e}")
            return None
            
    def _load_pca(self):
        """加载PCA模型"""
        try:
            import joblib
            pca_path = self.config.get('pca_path', 'models/pca.joblib')
            return joblib.load(pca_path)
        except Exception as e:
            logging.error(f"PCA模型加载失败: {e}")
            return None
            
    def extract_features(self, colony_data: Dict[str, Any]) -> np.ndarray:
        """提取分类特征"""
        features = []
        
        # 1. 基本特征
        basic_features = colony_data['basic_features']
        features.extend([
            basic_features['area'],
            basic_features['perimeter'],
            basic_features['circularity'],
            basic_features['mean_intensity'],
            basic_features['std_intensity'],
            basic_features['intensity_range']
        ])
        
        # 2. 形状特征
        shape_features = colony_data['shape_features']
        if shape_features:
            features.extend([
                shape_features['extent'],
                shape_features['solidity'],
                shape_features['eccentricity'],
                shape_features['hu1'],
                shape_features['hu2'],
                shape_features['hu3'],
                shape_features['aspect_ratio']
            ])
            
        # 3. 纹理特征
        texture_features = colony_data['texture_features']
        if texture_features:
            # GLCM特征
            for distance in [1, 2]:
                for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
                    features.append(texture_features.get(f'{prop}_d{distance}', 0))
                    
            # LBP特征
            for scale in [1, 2, 4]:
                for i in range(10):
                    features.append(texture_features.get(f'lbp_s{scale}_b{i}', 0))
                    
            # Gabor特征
            for freq in [0.1, 0.2, 0.4]:
                for angle in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                    features.append(texture_features.get(f'gabor_f{freq:.1f}_a{angle:.1f}', 0))
                    
        return np.array(features)
        
    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """预处理特征"""
        # 1. 特征缩放
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform(features.reshape(1, -1))
            
        # 2. PCA降维
        if self.pca is not None:
            features = self.pca.transform(features)
            
        return features
        
    def classify(self, colony_data: Dict[str, Any]) -> Dict[str, Any]:
        """分类菌落"""
        if self.classifier is None:
            return {'class': 'unknown', 'confidence': 0.0}
            
        # 1. 特征提取
        features = self.extract_features(colony_data)
        
        # 2. 特征预处理
        processed_features = self.preprocess_features(features)
        
        # 3. 模型预测
        if self.use_ensemble:
            # 集成预测
            probabilities = self.classifier.predict_proba(processed_features)
            predicted_class = self.classifier.predict(processed_features)[0]
            confidence = np.max(probabilities)
        else:
            # 单模型预测
            predicted_class = self.classifier.predict(processed_features)[0]
            confidence = getattr(self.classifier, 'predict_proba', 
                lambda x: np.array([[0, 1] if self.classifier.predict(x)[0] 
                else [1, 0]]))(processed_features)[0][predicted_class]
                
        # 4. 置信度检查
        if confidence < self.confidence_threshold:
            predicted_class = 'uncertain'
            
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'features': features.tolist()
        }

    def _save_model(self, model, path: str, version: int):
        """保存模型"""
        try:
            import joblib
            save_path = f"{path}.v{version}"
            joblib.dump(model, save_path)
            logging.info(f"模型已保存到: {save_path}")
            return True
        except Exception as e:
            logging.error(f"模型保存失败: {e}")
            return False
            
    def _update_model(self):
        """更新模型"""
        if len(self._training_buffer) < self.min_samples_for_update:
            return False
            
        try:
            with self._update_lock:
                # 1. 提取特征和标签
                X = []
                y = []
                for sample in self._training_buffer:
                    features = sample['features']
                    label = sample['label']
                    X.append(features)
                    y.append(label)
                    
                X = np.array(X)
                y = np.array(y)
                
                # 2. 更新特征缩放器
                if self.feature_scaler is not None:
                    self.feature_scaler.partial_fit(X)
                    X_scaled = self.feature_scaler.transform(X)
                else:
                    X_scaled = X
                    
                # 3. 更新PCA
                if self.pca is not None:
                    self.pca.partial_fit(X_scaled)
                    X_pca = self.pca.transform(X_scaled)
                else:
                    X_pca = X_scaled
                    
                # 4. 更新分类器
                if hasattr(self.classifier, 'partial_fit'):
                    # 增量学习
                    classes = np.unique(y)
                    self.classifier.partial_fit(X_pca, y, classes=classes)
                else:
                    # 完全重训练
                    self.classifier.fit(X_pca, y)
                    
                # 5. 保存更新后的模型
                self._model_version += 1
                model_saved = (
                    self._save_model(self.classifier, self.config['classifier_model_path'], self._model_version) and
                    self._save_model(self.feature_scaler, self.config['scaler_path'], self._model_version) and
                    self._save_model(self.pca, self.config['pca_path'], self._model_version)
                )
                
                if model_saved:
                    # 清空缓冲区
                    self._training_buffer = []
                    self._samples_since_update = 0
                    logging.info(f"模型更新成功，版本: {self._model_version}")
                    return True
                else:
                    logging.error("模型更新失败")
                    return False
                    
        except Exception as e:
            logging.error(f"模型更新过程出错: {e}")
            return False
            
    def add_training_sample(self, features: np.ndarray, label: str):
        """添加训练样本"""
        if not self.online_learning_enabled:
            return
            
        # 添加到缓冲区
        self._training_buffer.append({
            'features': features,
            'label': label
        })
        
        # 限制缓冲区大小
        if len(self._training_buffer) > self.max_buffer_size:
            self._training_buffer.pop(0)
            
        self._samples_since_update += 1
        
        # 检查是否需要更新模型
        if self._samples_since_update >= self.update_interval:
            self._update_model()
            
    def classify_with_feedback(
        self,
        colony_data: Dict[str, Any],
        true_label: Optional[str] = None
    ) -> Dict[str, Any]:
        """分类菌落并接收反馈"""
        # 1. 进行分类
        result = self.classify(colony_data)
        
        # 2. 如果提供了真实标签，添加到训练数据
        if true_label is not None and self.online_learning_enabled:
            features = self.extract_features(colony_data)
            self.add_training_sample(features, true_label)
            
            # 更新结果
            result['true_label'] = true_label
            result['is_correct'] = (result['class'] == true_label)
            
        return result
        
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'version': self._model_version,
            'samples_in_buffer': len(self._training_buffer),
            'samples_since_update': self._samples_since_update,
            'online_learning_enabled': self.online_learning_enabled
        }
        
    def toggle_online_learning(self, enabled: bool):
        """切换在线学习状态"""
        self.online_learning_enabled = enabled
        logging.info(f"在线学习已{'启用' if enabled else '禁用'}")
        
    def force_model_update(self) -> bool:
        """强制更新模型"""
        return self._update_model()

class ColonyAnalyzer:
    """菌落分析器"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.classifier = ColonyClassifier(config.get('model', {}))
        self.image_processor = ImageProcessor(config.get('image', {}))
        
    def analyze_image_with_feedback(
        self,
        image: np.ndarray,
        feedback_data: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """分析图像并处理反馈数据"""
        # 1. 基础分析
        colonies = self.image_processor.detect_colonies(image)
        if not colonies:
            return {'error': '未检测到菌落'}
            
        # 2. 特征提取和分类
        classification_results = []
        for colony in colonies:
            # 检查是否有该菌落的反馈
            colony_id = colony.get('id')
            true_label = None
            if feedback_data and colony_id in feedback_data:
                true_label = feedback_data[colony_id]
                
            # 使用带反馈的分类方法
            result = self.classifier.classify_with_feedback(colony, true_label)
            colony['classification'] = result
            classification_results.append(result)
            
        # 3. 统计分析
        stats = self._analyze_classification_stats(classification_results)
        
        return {
            'colonies': colonies,
            'classification_results': classification_results,
            'stats': stats,
            'model_info': self.classifier.get_model_info()
        }
        
    def _analyze_classification_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析分类结果统计信息"""
        total = len(results)
        if total == 0:
            return {}
            
        # 统计各类别数量
        class_counts = {}
        confidence_sum = 0
        correct_count = 0
        feedback_count = 0
        
        for result in results:
            class_name = result['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidence_sum += result.get('confidence', 0)
            
            # 统计反馈相关信息
            if 'true_label' in result:
                feedback_count += 1
                if result.get('is_correct', False):
                    correct_count += 1
                    
        # 计算统计指标
        stats = {
            'total_colonies': total,
            'class_distribution': {
                k: v/total for k, v in class_counts.items()
            },
            'mean_confidence': confidence_sum / total,
            'feedback_ratio': feedback_count / total if total > 0 else 0
        }
        
        # 如果有反馈数据，添加准确率
        if feedback_count > 0:
            stats['accuracy'] = correct_count / feedback_count
            
        return stats
        
    def toggle_online_learning(self, enabled: bool):
        """切换在线学习状态"""
        self.classifier.toggle_online_learning(enabled)
        
    def force_model_update(self) -> bool:
        """强制更新模型"""
        return self.classifier.force_model_update()
        
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return self.classifier.get_model_info()

class ProcessManager:
    """进程管理器"""
    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self.processes: Dict[int, multiprocessing.Process] = {}
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self._stop_event = multiprocessing.Event()
        
    def start(self):
        """启动进程池"""
        for i in range(self.max_workers):
            p = multiprocessing.Process(
                target=self._worker_loop,
                args=(i, self.task_queue, self.result_queue, self._stop_event)
            )
            p.daemon = True
            p.start()
            self.processes[i] = p
            
    def stop(self):
        """停止所有进程"""
        self._stop_event.set()
        for p in self.processes.values():
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                
    @staticmethod
    def _worker_loop(
        worker_id: int,
        task_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
        stop_event: multiprocessing.Event
    ):
        """工作进程循环"""
        while not stop_event.is_set():
            try:
                task = task_queue.get(timeout=1)
                result = task()
                result_queue.put((worker_id, result))
            except Exception as e:
                result_queue.put((worker_id, e))

class TaskManager:
    """任务管理器"""
    def __init__(self, config: SystemConfig):
        self.config = config
        self.process_manager = ProcessManager(config.max_workers)
        self.task_buffer = deque(maxlen=config.buffer_size)
        self._results = []
        self._well_data = {}
        
    def submit_well_analysis(self, well_data: WellData):
        """提交孔板分析任务"""
        def analysis_task():
            # 实现孔板分析任务
            return well_data
            
        self.task_buffer.append(analysis_task)
        if len(self.task_buffer) >= self.config.buffer_size:
            self._flush_buffer()
            
    def _flush_buffer(self):
        """刷新任务缓冲区"""
        while self.task_buffer:
            task = self.task_buffer.popleft()
            self.process_manager.task_queue.put(task)
            
    def collect_results(self, timeout: Optional[float] = None) -> List[WellData]:
        """收集分析结果"""
        end_time = time.time() + (timeout or self.config.timeout)
        while time.time() < end_time:
            try:
                worker_id, result = self.process_manager.result_queue.get(timeout=0.1)
                if isinstance(result, Exception):
                    logging.error(f"Worker {worker_id} 发生错误: {result}")
                else:
                    self._results.append(result)
            except Exception:
                pass
        return self._results

class CloneSystem:
    """单细胞克隆系统"""
    def __init__(self, config_path: str):
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化组件
        self.hardware = HardwareController(self.config.get('hardware', {}))
        self.analyzer = ColonyAnalyzer(self.config)
        self.process_manager = ProcessManager(self.config.max_workers)
        self.task_manager = TaskManager(self.config)
        
        # 初始化状态
        self.is_running = False
        self.current_plate = None
        self.feedback_buffer = {}
        
    def scan_plate(self, plate_id: str, feedback_data: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """扫描培养板"""
        try:
            # 1. 准备扫描
            self.current_plate = plate_id
            self.hardware.initialize()
            
            # 2. 获取图像
            image = self.hardware.capture()
            if image is None:
                raise RuntimeError("图像获取失败")
                
            # 3. 分析图像（带反馈）
            results = self.analyzer.analyze_image_with_feedback(image, feedback_data)
            
            # 4. 保存结果
            self._save_results(plate_id, results)
            
            # 5. 更新反馈缓冲区
            if feedback_data:
                self.feedback_buffer[plate_id] = feedback_data
                
            return results
            
        except Exception as e:
            logging.error(f"扫描失败: {e}")
            return {'error': str(e)}
            
        finally:
            self.hardware.shutdown()
            
    def add_feedback(self, plate_id: str, colony_id: str, true_label: str):
        """添加分类反馈"""
        if plate_id not in self.feedback_buffer:
            self.feedback_buffer[plate_id] = {}
            
        self.feedback_buffer[plate_id][colony_id] = true_label
        logging.info(f"已添加反馈: 培养板 {plate_id}, 菌落 {colony_id}, 标签 {true_label}")
        
    def process_feedback(self):
        """处理所有待处理的反馈"""
        for plate_id, feedback in self.feedback_buffer.items():
            if feedback:
                # 重新分析带反馈的图像
                self.scan_plate(plate_id, feedback)
                logging.info(f"已处理培养板 {plate_id} 的反馈数据")
                
        # 清空反馈缓冲区
        self.feedback_buffer = {}
        
    def toggle_online_learning(self, enabled: bool):
        """切换在线学习状态"""
        self.analyzer.toggle_online_learning(enabled)
        logging.info(f"在线学习已{'启用' if enabled else '禁用'}")
        
    def force_model_update(self) -> bool:
        """强制更新模型"""
        success = self.analyzer.force_model_update()
        if success:
            logging.info("模型已更新")
        else:
            logging.warning("模型更新失败")
        return success
        
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return self.analyzer.get_model_info()
        
    def _save_results(self, plate_id: str, results: Dict[str, Any]):
        """保存分析结果"""
        try:
            # 1. 创建结果目录
            output_dir = os.path.join(self.config['output_dir'], plate_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # 2. 保存结果JSON
            results_path = os.path.join(output_dir, 'analysis_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            # 3. 保存统计信息
            stats_path = os.path.join(output_dir, 'stats.json')
            with open(stats_path, 'w') as f:
                json.dump(results.get('stats', {}), f, indent=2)
                
            # 4. 保存模型信息
            model_info_path = os.path.join(output_dir, 'model_info.json')
            with open(model_info_path, 'w') as f:
                json.dump(results.get('model_info', {}), f, indent=2)
                
            logging.info(f"结果已保存到: {output_dir}")
            
        except Exception as e:
            logging.error(f"结果保存失败: {e}")

    def _load_config(self, config_path: str) -> SystemConfig:
        """加载配置"""
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
            
        return SystemConfig(
            root_dir=Path(config_data['system']['root_dir']),
            config_file=Path(config_path),
            log_dir=Path(config_data['system']['log_dir']),
            output_dir=Path(config_data['system']['output_dir']),
            max_workers=config_data['processing']['max_workers'],
            buffer_size=config_data['processing']['buffer_size'],
            timeout=config_data['processing']['timeout'],
            debug=config_data['system']['debug'],
            camera_config=config_data['hardware']['camera'],
            stage_config=config_data['hardware']['stage'],
            model_config=config_data['model'],
            processing_config=config_data['processing']
        )

def main():
    """主函数"""
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='单细胞克隆自动化系统')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()
    
    # 创建并运行系统
    system = CloneSystem(args.config)
    system.run()

if __name__ == '__main__':
    main() 