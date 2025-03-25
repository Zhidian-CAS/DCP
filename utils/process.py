import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Any, List, Dict
from queue import Queue
import threading
import time

class ProcessManager:
    """进程管理工具类"""
    
    def __init__(self, max_workers: int, buffer_size: int):
        """
        初始化进程管理器
        
        Args:
            max_workers: 最大工作线程数
            buffer_size: 缓冲区大小
        """
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = Queue(maxsize=buffer_size)
        self.results: Dict[str, Future] = {}
        self.running = False
        self._lock = threading.Lock()
        
    def start(self):
        """启动进程管理器"""
        self.running = True
        self.logger.info("进程管理器已启动")
        
    def stop(self):
        """停止进程管理器"""
        self.running = False
        self.executor.shutdown(wait=True)
        self.logger.info("进程管理器已停止")
        
    def submit_task(self, task_id: str, func: Callable, *args, **kwargs) -> bool:
        """
        提交任务
        
        Args:
            task_id: 任务ID
            func: 要执行的函数
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            是否成功提交任务
        """
        if not self.running:
            self.logger.error("进程管理器未启动")
            return False
            
        try:
            future = self.executor.submit(func, *args, **kwargs)
            with self._lock:
                self.results[task_id] = future
            self.logger.debug(f"任务 {task_id} 已提交")
            return True
        except Exception as e:
            self.logger.error(f"提交任务 {task_id} 失败: {str(e)}")
            return False
            
    def get_result(self, task_id: str, timeout: float = None) -> Any:
        """
        获取任务结果
        
        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）
            
        Returns:
            任务结果
        """
        with self._lock:
            if task_id not in self.results:
                self.logger.error(f"任务 {task_id} 不存在")
                return None
                
            future = self.results[task_id]
            
        try:
            result = future.result(timeout=timeout)
            self.logger.debug(f"任务 {task_id} 已完成")
            return result
        except TimeoutError:
            self.logger.error(f"任务 {task_id} 超时")
            return None
        except Exception as e:
            self.logger.error(f"获取任务 {task_id} 结果失败: {str(e)}")
            return None
            
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功取消任务
        """
        with self._lock:
            if task_id not in self.results:
                self.logger.error(f"任务 {task_id} 不存在")
                return False
                
            future = self.results[task_id]
            
        cancelled = future.cancel()
        if cancelled:
            self.logger.debug(f"任务 {task_id} 已取消")
        else:
            self.logger.error(f"取消任务 {task_id} 失败")
            
        return cancelled 