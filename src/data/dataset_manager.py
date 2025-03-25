import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import yaml
from dataclasses import dataclass, asdict
import requests
import hashlib
import os

@dataclass
class DatasetCitation:
    """数据集引用信息"""
    title: str
    authors: List[str]
    year: int
    journal: str
    doi: Optional[str] = None
    url: Optional[str] = None
    citation_text: Optional[str] = None

@dataclass
class DatasetInfo:
    """数据集信息"""
    name: str
    version: str
    description: str
    accession_number: Optional[str] = None
    download_url: Optional[str] = None
    local_path: Optional[str] = None
    checksum: Optional[str] = None
    size: Optional[int] = None
    format: Optional[str] = None
    license: Optional[str] = None
    citation: Optional[DatasetCitation] = None
    splits: Optional[Dict[str, str]] = None
    metadata: Optional[Dict] = None

class DatasetManager:
    """数据集管理器"""
    
    def __init__(
        self,
        config_path: Union[str, Path],
        data_root: Optional[Union[str, Path]] = None
    ):
        """
        初始化数据集管理器
        Args:
            config_path: 配置文件路径
            data_root: 数据根目录
        """
        self.config_path = Path(config_path)
        self.data_root = Path(data_root) if data_root else Path('data')
        self.logger = logging.getLogger(__name__)
        
        # 加载配置
        self.config = self._load_config()
        
        # 初始化数据集信息
        self.datasets: Dict[str, DatasetInfo] = {}
        self._load_dataset_info()
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
        
    def _load_dataset_info(self):
        """加载数据集信息"""
        info_path = self.config_path.parent / 'dataset_info.json'
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for name, info in data.items():
                    self.datasets[name] = DatasetInfo(**info)
                    
    def _save_dataset_info(self):
        """保存数据集信息"""
        info_path = self.config_path.parent / 'dataset_info.json'
        data = {
            name: asdict(info)
            for name, info in self.datasets.items()
        }
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
    def add_dataset(
        self,
        name: str,
        version: str,
        description: str,
        accession_number: Optional[str] = None,
        download_url: Optional[str] = None,
        local_path: Optional[str] = None,
        checksum: Optional[str] = None,
        size: Optional[int] = None,
        format: Optional[str] = None,
        license: Optional[str] = None,
        citation: Optional[DatasetCitation] = None,
        splits: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict] = None
    ) -> DatasetInfo:
        """
        添加数据集
        Args:
            name: 数据集名称
            version: 版本号
            description: 描述
            accession_number: 访问号
            download_url: 下载链接
            local_path: 本地路径
            checksum: 校验和
            size: 大小
            format: 格式
            license: 许可证
            citation: 引用信息
            splits: 数据集划分
            metadata: 元数据
        Returns:
            数据集信息
        """
        dataset = DatasetInfo(
            name=name,
            version=version,
            description=description,
            accession_number=accession_number,
            download_url=download_url,
            local_path=local_path,
            checksum=checksum,
            size=size,
            format=format,
            license=license,
            citation=citation,
            splits=splits,
            metadata=metadata
        )
        
        self.datasets[name] = dataset
        self._save_dataset_info()
        
        return dataset
        
    def download_dataset(
        self,
        name: str,
        force: bool = False
    ) -> bool:
        """
        下载数据集
        Args:
            name: 数据集名称
            force: 是否强制重新下载
        Returns:
            是否下载成功
        """
        if name not in self.datasets:
            raise KeyError(f"数据集不存在: {name}")
            
        dataset = self.datasets[name]
        if not dataset.download_url:
            raise ValueError(f"数据集 {name} 没有下载链接")
            
        # 检查本地文件
        if dataset.local_path and Path(dataset.local_path).exists():
            if not force:
                self.logger.info(f"数据集 {name} 已存在")
                return True
                
        # 创建下载目录
        download_dir = self.data_root / name
        download_dir.mkdir(parents=True, exist_ok=True)
        
        # 下载数据集
        try:
            response = requests.get(dataset.download_url, stream=True)
            response.raise_for_status()
            
            # 保存文件
            file_path = download_dir / f"{name}.{dataset.format}"
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            # 更新本地路径
            dataset.local_path = str(file_path)
            
            # 计算校验和
            if dataset.checksum:
                actual_checksum = self._calculate_checksum(file_path)
                if actual_checksum != dataset.checksum:
                    raise ValueError(f"校验和不匹配: {actual_checksum} != {dataset.checksum}")
                    
            self._save_dataset_info()
            self.logger.info(f"数据集 {name} 下载成功")
            return True
            
        except Exception as e:
            self.logger.error(f"下载数据集 {name} 失败: {str(e)}")
            return False
            
    def _calculate_checksum(self, file_path: Path) -> str:
        """计算文件校验和"""
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b''):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
        
    def get_dataset_citation(self, name: str) -> str:
        """
        获取数据集引用信息
        Args:
            name: 数据集名称
        Returns:
            引用文本
        """
        if name not in self.datasets:
            raise KeyError(f"数据集不存在: {name}")
            
        dataset = self.datasets[name]
        if not dataset.citation:
            return f"{dataset.name} (version {dataset.version})"
            
        citation = dataset.citation
        if citation.citation_text:
            return citation.citation_text
            
        # 生成引用文本
        authors = ", ".join(citation.authors)
        citation_text = f"{authors} ({citation.year}). {citation.title}. {citation.journal}"
        
        if citation.doi:
            citation_text += f". doi: {citation.doi}"
        if citation.url:
            citation_text += f". URL: {citation.url}"
            
        return citation_text
        
    def get_dataset_splits(self, name: str) -> Dict[str, str]:
        """
        获取数据集划分信息
        Args:
            name: 数据集名称
        Returns:
            划分信息
        """
        if name not in self.datasets:
            raise KeyError(f"数据集不存在: {name}")
            
        dataset = self.datasets[name]
        if not dataset.splits:
            return {}
            
        return dataset.splits
        
    def verify_dataset(self, name: str) -> bool:
        """
        验证数据集
        Args:
            name: 数据集名称
        Returns:
            是否验证通过
        """
        if name not in self.datasets:
            raise KeyError(f"数据集不存在: {name}")
            
        dataset = self.datasets[name]
        if not dataset.local_path or not Path(dataset.local_path).exists():
            return False
            
        # 检查文件大小
        if dataset.size:
            actual_size = Path(dataset.local_path).stat().st_size
            if actual_size != dataset.size:
                return False
                
        # 检查校验和
        if dataset.checksum:
            actual_checksum = self._calculate_checksum(Path(dataset.local_path))
            if actual_checksum != dataset.checksum:
                return False
                
        return True 