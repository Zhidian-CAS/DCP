import logging
import sys
from pathlib import Path
from typing import Optional

class LoggerSetup:
    """日志设置工具类"""
    
    @staticmethod
    def setup_logger(
        name: str,
        log_file: Optional[Path] = None,
        level: int = logging.INFO,
        format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ) -> logging.Logger:
        """
        设置并返回一个logger实例
        
        Args:
            name: logger名称
            log_file: 日志文件路径
            level: 日志级别
            format_str: 日志格式
            
        Returns:
            配置好的logger实例
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        formatter = logging.Formatter(format_str)
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 如果指定了日志文件，添加文件处理器
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        return logger 