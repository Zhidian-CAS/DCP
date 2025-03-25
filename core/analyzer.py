from typing import Dict, List, Optional, Any
import numpy as np

from .classifier import ColonyClassifier
from ..image.processor import ImageProcessor

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