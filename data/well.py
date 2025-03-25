from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

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