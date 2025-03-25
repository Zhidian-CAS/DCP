import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import shutil
import pandas as pd
from dataclasses import dataclass, asdict
import yaml

@dataclass
class DataSourceMetadata:
    """数据源元数据"""
    source_id: str
    source_type: str
    creation_date: str
    last_modified: str
    file_count: int
    total_size: int
    format: str
    version: str
    checksum: str
    description: Optional[str] = None
    tags: List[str] = None
    quality_metrics: Dict[str, float] = None

class DataManager:
    """数据管理器"""
    
    def __init__(
        self,
        data_root: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None
    ):
        """
        初始化数据管理器
        Args:
            data_root: 数据根目录
            config_path: 配置文件路径
        """
        self.data_root = Path(data_root)
        self.config_path = Path(config_path) if config_path else None
        self.logger = logging.getLogger(__name__)
        
        # 创建必要的目录
        self._create_directories()
        
        # 加载配置
        self.config = self._load_config()
        
        # 初始化数据源元数据
        self.metadata: Dict[str, DataSourceMetadata] = {}
        
    def _create_directories(self):
        """创建必要的目录结构"""
        directories = [
            self.data_root / 'raw',
            self.data_root / 'processed',
            self.data_root / 'backup',
            self.data_root / 'metadata',
            self.data_root / 'logs'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def _load_config(self) -> Dict:
        """加载配置文件"""
        if self.config_path and self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
        
    def add_data_source(
        self,
        source_id: str,
        source_type: str,
        data_path: Union[str, Path],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> DataSourceMetadata:
        """
        添加数据源
        Args:
            source_id: 数据源ID
            source_type: 数据源类型
            data_path: 数据路径
            description: 描述
            tags: 标签
        Returns:
            数据源元数据
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"数据路径不存在: {data_path}")
            
        # 计算数据源统计信息
        file_count = len(list(data_path.rglob('*')))
        total_size = sum(f.stat().st_size for f in data_path.rglob('*') if f.is_file())
        
        # 计算校验和
        checksum = self._calculate_checksum(data_path)
        
        # 创建元数据
        metadata = DataSourceMetadata(
            source_id=source_id,
            source_type=source_type,
            creation_date=datetime.now().isoformat(),
            last_modified=datetime.now().isoformat(),
            file_count=file_count,
            total_size=total_size,
            format=data_path.suffix[1:],
            version='1.0.0',
            checksum=checksum,
            description=description,
            tags=tags or [],
            quality_metrics={}
        )
        
        # 保存元数据
        self.metadata[source_id] = metadata
        self._save_metadata(source_id)
        
        # 记录日志
        self.logger.info(f"添加数据源: {source_id}")
        
        return metadata
        
    def _calculate_checksum(self, path: Path) -> str:
        """计算目录的校验和"""
        sha256_hash = hashlib.sha256()
        
        for file_path in sorted(path.rglob('*')):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    for byte_block in iter(lambda: f.read(4096), b''):
                        sha256_hash.update(byte_block)
                        
        return sha256_hash.hexdigest()
        
    def _save_metadata(self, source_id: str):
        """保存数据源元数据"""
        metadata_path = self.data_root / 'metadata' / f'{source_id}.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.metadata[source_id]), f, indent=2)
            
    def validate_data_quality(
        self,
        source_id: str,
        quality_metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        验证数据质量
        Args:
            source_id: 数据源ID
            quality_metrics: 质量指标
        Returns:
            是否通过验证
        """
        if source_id not in self.metadata:
            raise KeyError(f"数据源不存在: {source_id}")
            
        # 更新质量指标
        if quality_metrics:
            self.metadata[source_id].quality_metrics.update(quality_metrics)
            self._save_metadata(source_id)
            
        # 检查数据完整性
        current_checksum = self._calculate_checksum(
            self.data_root / 'raw' / source_id
        )
        if current_checksum != self.metadata[source_id].checksum:
            self.logger.warning(f"数据源 {source_id} 的校验和不匹配")
            return False
            
        # 检查质量指标
        for metric, threshold in self.config.get('quality_thresholds', {}).items():
            if metric in self.metadata[source_id].quality_metrics:
                if self.metadata[source_id].quality_metrics[metric] < threshold:
                    self.logger.warning(
                        f"数据源 {source_id} 的质量指标 {metric} 未达到阈值"
                    )
                    return False
                    
        return True
        
    def create_backup(self, source_id: str) -> str:
        """
        创建数据备份
        Args:
            source_id: 数据源ID
        Returns:
            备份路径
        """
        if source_id not in self.metadata:
            raise KeyError(f"数据源不存在: {source_id}")
            
        # 创建备份目录
        backup_dir = self.data_root / 'backup' / source_id
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成备份文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f'backup_{timestamp}'
        backup_path = backup_dir / backup_name
        
        # 复制数据
        source_path = self.data_root / 'raw' / source_id
        shutil.copytree(source_path, backup_path)
        
        # 记录日志
        self.logger.info(f"创建数据源 {source_id} 的备份: {backup_path}")
        
        return str(backup_path)
        
    def restore_backup(
        self,
        source_id: str,
        backup_name: Optional[str] = None
    ) -> bool:
        """
        恢复数据备份
        Args:
            source_id: 数据源ID
            backup_name: 备份名称
        Returns:
            是否恢复成功
        """
        if source_id not in self.metadata:
            raise KeyError(f"数据源不存在: {source_id}")
            
        backup_dir = self.data_root / 'backup' / source_id
        if not backup_dir.exists():
            raise FileNotFoundError(f"备份目录不存在: {backup_dir}")
            
        # 获取最新的备份
        if backup_name is None:
            backups = sorted(backup_dir.glob('backup_*'))
            if not backups:
                raise FileNotFoundError(f"没有找到备份: {source_id}")
            backup_path = backups[-1]
        else:
            backup_path = backup_dir / backup_name
            
        if not backup_path.exists():
            raise FileNotFoundError(f"备份不存在: {backup_path}")
            
        # 恢复数据
        source_path = self.data_root / 'raw' / source_id
        shutil.rmtree(source_path, ignore_errors=True)
        shutil.copytree(backup_path, source_path)
        
        # 更新元数据
        self.metadata[source_id].last_modified = datetime.now().isoformat()
        self._save_metadata(source_id)
        
        # 记录日志
        self.logger.info(f"恢复数据源 {source_id} 的备份: {backup_path}")
        
        return True
        
    def get_data_version(
        self,
        source_id: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取数据版本信息
        Args:
            source_id: 数据源ID
            version: 版本号
        Returns:
            版本信息
        """
        if source_id not in self.metadata:
            raise KeyError(f"数据源不存在: {source_id}")
            
        metadata = self.metadata[source_id]
        if version and version != metadata.version:
            raise ValueError(f"版本不存在: {version}")
            
        return {
            'source_id': source_id,
            'version': metadata.version,
            'creation_date': metadata.creation_date,
            'last_modified': metadata.last_modified,
            'file_count': metadata.file_count,
            'total_size': metadata.total_size,
            'checksum': metadata.checksum,
            'quality_metrics': metadata.quality_metrics
        }
        
    def update_data_version(
        self,
        source_id: str,
        new_version: str,
        description: Optional[str] = None
    ) -> bool:
        """
        更新数据版本
        Args:
            source_id: 数据源ID
            new_version: 新版本号
            description: 版本描述
        Returns:
            是否更新成功
        """
        if source_id not in self.metadata:
            raise KeyError(f"数据源不存在: {source_id}")
            
        # 创建新版本备份
        self.create_backup(source_id)
        
        # 更新元数据
        metadata = self.metadata[source_id]
        metadata.version = new_version
        metadata.last_modified = datetime.now().isoformat()
        if description:
            metadata.description = description
            
        self._save_metadata(source_id)
        
        # 记录日志
        self.logger.info(
            f"更新数据源 {source_id} 的版本: {new_version}"
        )
        
        return True 