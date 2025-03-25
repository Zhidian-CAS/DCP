import logging
from pathlib import Path
from src.data.data_manager import DataManager

def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """主函数"""
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 初始化数据管理器
    data_root = Path('data')
    config_path = Path('config/data_config.yaml')
    data_manager = DataManager(data_root, config_path)
    
    try:
        # 添加图像数据源
        image_source = data_manager.add_data_source(
            source_id='colony_images_v1',
            source_type='image',
            data_path='data/raw/colony_images',
            description='菌落图像数据集',
            tags=['colony', 'microbiology', 'segmentation']
        )
        logger.info(f"添加数据源: {image_source.source_id}")
        
        # 添加标注数据源
        annotation_source = data_manager.add_data_source(
            source_id='colony_annotations_v1',
            source_type='annotation',
            data_path='data/raw/colony_annotations',
            description='菌落标注数据集',
            tags=['colony', 'annotation', 'segmentation']
        )
        logger.info(f"添加数据源: {annotation_source.source_id}")
        
        # 验证数据质量
        quality_metrics = {
            'image_quality': {
                'resolution': 1024,
                'noise_level': 0.05,
                'contrast': 0.4
            },
            'completeness': {
                'file_count': 150,
                'missing_files': 0.02
            },
            'consistency': {
                'checksum_match': 0.995,
                'format_variance': 0.05
            },
            'distribution': {
                'class_balance': 0.85,
                'outlier_ratio': 0.03
            }
        }
        
        # 验证图像数据源
        if data_manager.validate_data_quality('colony_images_v1', quality_metrics):
            logger.info("图像数据源验证通过")
        else:
            logger.warning("图像数据源验证失败")
            
        # 创建数据备份
        backup_path = data_manager.create_backup('colony_images_v1')
        logger.info(f"创建备份: {backup_path}")
        
        # 更新数据版本
        if data_manager.update_data_version(
            'colony_images_v1',
            '1.1.0',
            '添加新的图像增强方法'
        ):
            logger.info("数据版本更新成功")
            
        # 获取版本信息
        version_info = data_manager.get_data_version('colony_images_v1')
        logger.info(f"当前版本信息: {version_info}")
        
        # 恢复数据备份
        if data_manager.restore_backup('colony_images_v1'):
            logger.info("数据备份恢复成功")
            
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
        raise

if __name__ == '__main__':
    main() 