from abc import ABC, abstractmethod

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