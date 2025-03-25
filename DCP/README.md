# Digital Colony Picker

## Project Overview

The objective of this project is to create an integrated control and analysis software that fully automates the process of chip-based single cell culture, identification, and sorting. The software encompasses key functionalities such as image recognition and process control, enabling the identification of microchambers and detailed analysis of their internal images. By integrating with hardware control, it facilitates the precise location of microchambers on the chip, while an optical module supports the acquisition of target monoclonal cells. This comprehensive solution aims to streamline and enhance the efficiency of single cell manipulation and analysis.

### Key Features

- Colony Detection: Using improved Faster R-CNN architecture
- Colony Segmentation: Based on improved SegNet architecture
- Feature Extraction: Using Feature Pyramid Network (FPN)
- Classification Prediction: Supporting multi-class colony classification
- Online Learning: Supporting model online updates

### Technical Features

- Multi-model Integration: Combining detection and segmentation models
- Attention Mechanism: Including channel and spatial attention modules
- Adaptive Learning: Supporting online learning and model updates
- High-performance Inference: Optimized inference speed and memory usage

## Requirements

### Hardware Requirements
- GPU: NVIDIA GPU with 8GB+ VRAM
- CPU: 8 cores+
- Memory: 16GB+

### Software Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (GPU version)
- OpenCV 4.5+
- NumPy 1.21+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Zhidian-CAS/DCP.git
cd clone-system
```

2. Create virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Basic Usage

```python
from src.core.system import CloneSystem
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize system
system = CloneSystem(config)

# Start system
system.start()

# Run scanning
system.scan_plate("plate_001")

# Stop system
system.stop()
```

### 2. Model Training

```python
from src.models.segmentation.training_manager import TrainingManager

# Initialize training manager
manager = TrainingManager(config)

# Start training
manager.train(
    train_loader=train_loader,
    valid_loader=valid_loader,
    num_epochs=100
)
```

### 3. Model Evaluation

```python
from src.models.sota_comparison import SOTAComparator

# Initialize comparator
comparator = SOTAComparator(config_path='configs/sota_models.yaml')

# Compare with SOTA models
results = comparator.compare_with_sota(
    model=dcpmodel,
    model_name="dcpmodel",
    sota_name="segnet",
    test_dataloader=test_loader
)
```

## Pre-trained Models

The system uses the following pre-trained models:

### Segmentation Models
- SegNet (Accuracy: 0.964, IoU: 0.892)
- U-Net (Accuracy: 0.972, IoU: 0.901)

### Detection Models
- Faster R-CNN (mAP: 0.742, mAP50: 0.901)

### Backbone Networks
- ResNet Series (50/101/152)
- EfficientNet Series (B0/B1/B2)
- DenseNet Series (121/169/201)

## Dataset

### Training Set
- Name: colony_train
- Version: 1.0.0
- Accession Number: DCP-TRAIN-001
- Size: 1GB
- Image Count: 10,000

### Validation Set
- Name: colony_validation
- Version: 1.0.0
- Accession Number: DCP-VAL-001
- Size: 100MB
- Image Count: 1,000

### Test Set
- Name: colony_test
- Version: 1.0.0
- Accession Number: DCP-TEST-001
- Size: 100MB
- Image Count: 1,000

## Performance Metrics

### Detection Performance
- mAP@0.5: 0.92
- mAP@0.75: 0.85

### Segmentation Performance
- Dice Coefficient: 0.89
- IoU: 0.82

### Classification Performance
- Accuracy: 0.95
- F1 Score: 0.94

## Limitations and Notes

### Environmental Requirements
- Stable lighting conditions required
- Avoid strong reflections and shadows

### Image Requirements
- Minimum Resolution: 512x512
- Maximum Resolution: 1920x1080
- Supported Formats: JPG, PNG

### Usage Recommendations
1. Regular system calibration
2. Regular model updates
3. Data backup
4. Regular system maintenance

## Maintenance and Updates

### Version History
- v1.0.0 (2024-01-01): Initial version
- v1.1.0 (2024-02-01): Added online learning
- v1.2.0 (2024-03-01): Performance optimization

### Update Plans
1. Add more data augmentation methods
2. Optimize model architecture
3. Add more evaluation metrics
4. Improve online learning mechanism

## Issue Reporting

- Issue Tracking: [GitHub Issues](https://github.com/Zhidian-CAS/DCP/issues)


## License

MIT License

