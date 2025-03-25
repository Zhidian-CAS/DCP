import serial
import time
from config.config import Config

class Stage:
    def __init__(self):
        self.serial = serial.Serial(
            port=Config.STAGE_PORT,
            baudrate=Config.STAGE_BAUDRATE,
            timeout=Config.STAGE_TIMEOUT
        )
        self.current_position = {'x': 0, 'y': 0, 'z': 0}
        self.initialize()
        
    def initialize(self):
        """初始化平台"""
        self.send_command('G28')  # 回零点
        time.sleep(2)
        
    def send_command(self, cmd):
        """发送指令到控制器"""
        cmd = cmd + '\n'
        self.serial.write(cmd.encode())
        response = self.serial.readline().decode().strip()
        return response
        
    def move_to(self, x=None, y=None, z=None, speed=1000):
        """移动到指定位置"""
        cmd = 'G1'
        if x is not None:
            cmd += f' X{x}'
            self.current_position['x'] = x
        if y is not None:
            cmd += f' Y{y}'
            self.current_position['y'] = y
        if z is not None:
            cmd += f' Z{z}'
            self.current_position['z'] = z
        cmd += f' F{speed}'
        
        self.send_command(cmd)
        time.sleep(0.1)  # 等待移动完成
        
    def move_relative(self, dx=0, dy=0, dz=0, speed=1000):
        """相对移动"""
        new_x = self.current_position['x'] + dx
        new_y = self.current_position['y'] + dy
        new_z = self.current_position['z'] + dz
        self.move_to(new_x, new_y, new_z, speed)
        
    def get_position(self):
        """获取当前位置"""
        return self.current_position
        
    def close(self):
        """关闭串口连接"""
        self.serial.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 