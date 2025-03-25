import logging
from pathlib import Path
from src.data.dataset_manager import DatasetManager, DatasetCitation

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # 初始化数据集管理器
    config_path = Path("config/dataset_config.yaml")
    data_root = Path("data")
    manager = DatasetManager(config_path, data_root)
    
    # 添加数据集
    citation = DatasetCitation(
        title="A Comprehensive Dataset for Colony Detection and Analysis",
        authors=["Smith, J.", "Johnson, K.", "Williams, L."],
        year=2023,
        journal="Scientific Data",
        doi="10.1038/s41597-023-00000-0",
        url="https://doi.org/10.1038/s41597-023-00000-0"
    )
    
    # 添加训练集
    train_dataset = manager.add_dataset(
        name="colony_train",
        version="1.0.0",
        description="菌落检测训练数据集",
        accession_number="DCP-TRAIN-001",
        download_url="https://example.com/datasets/colony_train_v1.0.0.zip",
        format="zip",
        size=1024000000,  # 1GB
        checksum="a1b2c3d4e5f6g7h8i9j0",  # SHA-256
        license="CC BY-NC-SA 4.0",
        citation=citation,
        splits={
            "train": "train",
            "validation": "val",
            "test": "test"
        },
        metadata={
            "image_count": 10000,
            "resolution": "2048x2048",
            "format": "PNG",
            "classes": ["background", "colony"]
        }
    )
    
    # 下载数据集
    logger.info("开始下载训练数据集...")
    if manager.download_dataset("colony_train"):
        logger.info("训练数据集下载成功")
    else:
        logger.error("训练数据集下载失败")
        
    # 验证数据集
    if manager.verify_dataset("colony_train"):
        logger.info("训练数据集验证通过")
    else:
        logger.error("训练数据集验证失败")
        
    # 获取数据集引用信息
    citation_text = manager.get_dataset_citation("colony_train")
    logger.info(f"数据集引用信息:\n{citation_text}")
    
    # 获取数据集划分信息
    splits = manager.get_dataset_splits("colony_train")
    logger.info(f"数据集划分信息:\n{splits}")
    
    # 添加验证集
    validation_dataset = manager.add_dataset(
        name="colony_validation",
        version="1.0.0",
        description="菌落检测验证数据集",
        accession_number="DCP-VAL-001",
        download_url="https://example.com/datasets/colony_validation_v1.0.0.zip",
        format="zip",
        size=102400000,  # 100MB
        checksum="b2c3d4e5f6g7h8i9j0k1",  # SHA-256
        license="CC BY-NC-SA 4.0",
        citation=citation,
        splits={
            "validation": "val"
        },
        metadata={
            "image_count": 1000,
            "resolution": "2048x2048",
            "format": "PNG",
            "classes": ["background", "colony"]
        }
    )
    
    # 下载验证集
    logger.info("开始下载验证数据集...")
    if manager.download_dataset("colony_validation"):
        logger.info("验证数据集下载成功")
    else:
        logger.error("验证数据集下载失败")
        
    # 验证数据集
    if manager.verify_dataset("colony_validation"):
        logger.info("验证数据集验证通过")
    else:
        logger.error("验证数据集验证失败")
        
    # 添加测试集
    test_dataset = manager.add_dataset(
        name="colony_test",
        version="1.0.0",
        description="菌落检测测试数据集",
        accession_number="DCP-TEST-001",
        download_url="https://example.com/datasets/colony_test_v1.0.0.zip",
        format="zip",
        size=102400000,  # 100MB
        checksum="c3d4e5f6g7h8i9j0k1l2",  # SHA-256
        license="CC BY-NC-SA 4.0",
        citation=citation,
        splits={
            "test": "test"
        },
        metadata={
            "image_count": 1000,
            "resolution": "2048x2048",
            "format": "PNG",
            "classes": ["background", "colony"]
        }
    )
    
    # 下载测试集
    logger.info("开始下载测试数据集...")
    if manager.download_dataset("colony_test"):
        logger.info("测试数据集下载成功")
    else:
        logger.error("测试数据集下载失败")
        
    # 验证数据集
    if manager.verify_dataset("colony_test"):
        logger.info("测试数据集验证通过")
    else:
        logger.error("测试数据集验证失败")

if __name__ == "__main__":
    main() 