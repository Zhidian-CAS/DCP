import os
import logging
import torch
from src.core.system import CloneSystem
from src.models.sota_comparison import SOTAComparator
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_demo():
    """Setup demo environment and load configuration."""
    # Load configuration
    config_path = 'configs/demo_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    return config, device

def run_detection_demo(config, device):
    """Run detection demo on sample images."""
    logger.info("Starting detection demo...")
    
    # Initialize system
    system = CloneSystem(config)
    system.start()
    
    # Process demo images
    demo_dir = 'data/demo/images'
    for image_name in os.listdir(demo_dir):
        if image_name.endswith('.jpg'):
            logger.info(f"Processing image: {image_name}")
            image_path = os.path.join(demo_dir, image_name)
            
            # Run detection
            results = system.detect_colonies(image_path)
            
            # Save results
            output_dir = 'results/demo/detection'
            os.makedirs(output_dir, exist_ok=True)
            system.save_results(results, os.path.join(output_dir, f"{image_name}_results.json"))
    
    system.stop()
    logger.info("Detection demo completed.")

def run_segmentation_demo(config, device):
    """Run segmentation demo on sample images."""
    logger.info("Starting segmentation demo...")
    
    # Initialize system
    system = CloneSystem(config)
    system.start()
    
    # Process demo images
    demo_dir = 'data/demo/images'
    for image_name in os.listdir(demo_dir):
        if image_name.endswith('.jpg'):
            logger.info(f"Processing image: {image_name}")
            image_path = os.path.join(demo_dir, image_name)
            
            # Run segmentation
            results = system.segment_colonies(image_path)
            
            # Save results
            output_dir = 'results/demo/segmentation'
            os.makedirs(output_dir, exist_ok=True)
            system.save_results(results, os.path.join(output_dir, f"{image_name}_results.json"))
    
    system.stop()
    logger.info("Segmentation demo completed.")

def run_comparison_demo(config, device):
    """Run comparison demo with SOTA models."""
    logger.info("Starting SOTA comparison demo...")
    
    # Initialize comparator
    comparator = SOTAComparator(config_path='configs/sota_models.yaml')
    
    # Load demo model
    model = CloneSystem(config).model
    model.to(device)
    
    # Run comparison
    results = comparator.compare_with_sota(
        model=model,
        model_name="demo_model",
        sota_name="segnet",
        test_dataloader=None,  # Use demo dataset
        device=device
    )
    
    # Save results
    output_dir = 'results/demo/comparison'
    os.makedirs(output_dir, exist_ok=True)
    comparator.save_comparison_results(results, output_dir)
    
    logger.info("SOTA comparison demo completed.")

def main():
    """Main function to run all demos."""
    logger.info("Starting DCP demo...")
    
    # Setup
    config, device = setup_demo()
    
    # Run demos
    run_detection_demo(config, device)
    run_segmentation_demo(config, device)
    run_comparison_demo(config, device)
    
    logger.info("All demos completed successfully.")

if __name__ == '__main__':
    main() 