from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

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