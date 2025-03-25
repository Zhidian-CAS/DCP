import logging
import threading
from typing import Dict, Optional, Any
import numpy as np

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