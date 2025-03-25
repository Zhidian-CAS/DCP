import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path
import json
import yaml

from ..hardware.base import HardwareController
from ..core.analyzer import ColonyAnalyzer
from ..utils.process import ProcessManager
from ..utils.task import TaskManager
from ..utils.config import SystemConfig

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