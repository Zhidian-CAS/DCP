import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any
import logging
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.system import CloneSystem
from src.utils.logger import LoggerSetup
from src.utils.config import SystemConfig

def load_config(config_path: Path) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_directories(config: Dict[str, Any]) -> None:
    """创建必要的目录"""
    directories = [
        config['paths']['log_dir'],
        config['paths']['output_dir'],
        config['paths']['model_dir']
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    """主程序入口"""
    # 加载配置
    config_path = project_root / 'config.yaml'
    config = load_config(config_path)
    
    # 创建必要的目录
    create_directories(config)
    
    # 设置日志
    logger = LoggerSetup.setup_logger(
        name='clone_system',
        log_file=Path(config['paths']['log_dir']) / 'system.log',
        level=logging.DEBUG if config['debug'] else logging.INFO
    )
    
    try:
        # 创建系统实例
        system = CloneSystem(config)
        
        # 启动系统
        system.start()
        
        # 运行扫描示例
        plate_id = "plate_001"
        system.scan_plate(plate_id)
        
        # 等待一段时间以模拟系统运行
        time.sleep(5)
        
        # 停止系统
        system.stop()
        
    except Exception as e:
        logger.error(f"系统运行出错: {str(e)}")
        raise
    finally:
        # 确保系统正确关闭
        if 'system' in locals():
            system.stop()

if __name__ == "__main__":
    main() 