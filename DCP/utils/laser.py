import serial
import time
from config.config import Config

class Laser:
    def __init__(self):
        self.serial = serial.Serial(
            port=Config.LASER_PORT,
            baudrate=Config.LASER_BAUDRATE,
            timeout=1
        )
        self.is_on = False
        self.initialize()
        
    def initialize(self):
        """初始化激光器"""
        self.turn_off()  # 确保初始状态为关闭
        time.sleep(1)
        
    def send_command(self, cmd):
        """发送指令到激光器"""
        cmd = cmd + '\n'
        self.serial.write(cmd.encode())
        response = self.serial.readline().decode().strip()
        return response
        
    def turn_on(self):
        """打开激光"""
        if not self.is_on:
            self.send_command('LASER:ON')
            self.is_on = True
            time.sleep(0.1)  # 等待激光稳定
            
    def turn_off(self):
        """关闭激光"""
        if self.is_on:
            self.send_command('LASER:OFF')
            self.is_on = False
            
    def set_power(self, power):
        """设置激光功率 (0-100%)"""
        power = max(0, min(100, power))
        self.send_command(f'LASER:POWER {power}')
        
    def fire(self, duration_ms=100):
        """发射激光
        duration_ms: 激光持续时间（毫秒）
        """
        self.turn_on()
        time.sleep(duration_ms / 1000.0)
        self.turn_off()
        
    def close(self):
        """关闭串口连接"""
        self.turn_off()
        self.serial.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 