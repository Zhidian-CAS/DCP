class Config:
    # 相机配置
    CAMERA_ID = 0
    CAMERA_EXPOSURE = -6
    CAMERA_GAIN = 10
    IMAGE_WIDTH = 1920
    IMAGE_HEIGHT = 1080

    # 3轴平台配置
    STAGE_PORT = 'COM3'
    STAGE_BAUDRATE = 115200
    STAGE_TIMEOUT = 1
    
    # 激光配置
    LASER_PORT = 'COM4'
    LASER_BAUDRATE = 115200
    
    # 目标检测模型配置
    DETECTION_MODEL = {
        'batch_size': 4,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'num_classes': 2,  # 背景和腔室
    }
    
    # 语义分割模型配置
    SEGMENTATION_MODEL = {
        'batch_size': 4,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'num_classes': 2,  # 背景和菌落
    }
    
    # 图像处理配置
    IMAGE_PROCESS = {
        'min_well_area': 1000,
        'max_well_area': 5000,
        'focus_roi_size': (100, 100),
        'focus_measure_threshold': 100,
    }
    
    # 数据路径配置
    DATA_PATH = {
        'raw_images': './data/raw_images',
        'labeled_data': './data/labeled_data',
        'processed_data': './data/processed_data',
        'detection_model': './models/detection/weights',
        'segmentation_model': './models/segmentation/weights',
    }
    
    # 自动聚焦配置
    AUTOFOCUS = {
        'start_position': -1000,
        'end_position': 1000,
        'step_size': 100,
        'fine_step_size': 10,
    } 