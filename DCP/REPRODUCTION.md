# Reproduction Instructions

This document provides detailed instructions for reproducing the results of the Digital Colony Picker (DCP) system.

## Environment Setup

1. Create a new conda environment:
```bash
conda create -n dcp python=3.8
conda activate dcp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import torch; print(torch.__version__)"
```

## Data Preparation

1. Download the full dataset:
```bash
wget https://github.com/Zhidian-CAS/DCP/releases/download/v1.0.0/dcp_dataset.zip
```

2. Extract the dataset:
```bash
unzip dcp_dataset.zip -d data/
```

3. Verify data structure:
```bash
python scripts/verify_dataset.py
```

## Model Training

1. Train the detection model:
```bash
python scripts/train_detection.py \
    --config configs/train_detection.yaml \
    --data_dir data/train \
    --output_dir checkpoints/detection
```

2. Train the segmentation model:
```bash
python scripts/train_segmentation.py \
    --config configs/train_segmentation.yaml \
    --data_dir data/train \
    --output_dir checkpoints/segmentation
```

### Training Parameters

- **Detection Model**:
  - Learning rate: 0.001
  - Batch size: 32
  - Number of epochs: 100
  - Early stopping patience: 10
  - Optimizer: AdamW
  - Weight decay: 0.0001

- **Segmentation Model**:
  - Learning rate: 0.001
  - Batch size: 16
  - Number of epochs: 100
  - Early stopping patience: 10
  - Optimizer: AdamW
  - Weight decay: 0.0001

## Model Evaluation

1. Evaluate the detection model:
```bash
python scripts/evaluate_detection.py \
    --model_path checkpoints/detection/best_model.pth \
    --test_dir data/test \
    --output_dir results/detection
```

2. Evaluate the segmentation model:
```bash
python scripts/evaluate_segmentation.py \
    --model_path checkpoints/segmentation/best_model.pth \
    --test_dir data/test \
    --output_dir results/segmentation
```

## Expected Results

### Detection Performance
- mAP@0.5: 0.92 ± 0.02
- mAP@0.75: 0.85 ± 0.03
- Recall: 0.89 ± 0.02
- Precision: 0.91 ± 0.02

### Segmentation Performance
- Dice Coefficient: 0.89 ± 0.02
- IoU: 0.82 ± 0.03
- Pixel Accuracy: 0.95 ± 0.01

### Processing Speed
- Detection: 50ms/image
- Segmentation: 100ms/image
- Total: 150ms/image

## Common Issues and Solutions

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient checkpointing
   - Enable mixed precision training

2. **Training Instability**
   - Adjust learning rate
   - Use learning rate warmup
   - Check data normalization

3. **Poor Performance**
   - Verify data quality
   - Check model architecture
   - Adjust hyperparameters

## Additional Resources

- [Model Architecture Details](docs/architecture.md)
- [Dataset Documentation](docs/dataset.md)
- [Training Tips](docs/training_tips.md)
- [Troubleshooting Guide](docs/troubleshooting.md) 