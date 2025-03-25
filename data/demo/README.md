# Demo Dataset

This is a small demonstration dataset for the Digital Colony Picker (DCP) system.

## Dataset Information

- **Size**: 100MB
- **Image Count**: 10 images
- **Format**: JPG images with corresponding JSON annotations
- **Resolution**: 512x512 pixels
- **Content**: Simulated colony images with various conditions

## Directory Structure

```
data/demo/
├── images/           # Colony images
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── annotations/      # JSON annotation files
│   ├── image_001.json
│   ├── image_002.json
│   └── ...
└── README.md        # This file
```

## Download Instructions

1. Download the demo dataset:
   ```bash
   wget https://github.com/Zhidian-CAS/DCP/releases/download/v1.0.0/demo_dataset.zip
   ```

2. Extract the dataset:
   ```bash
   unzip demo_dataset.zip -d data/demo/
   ```

## Usage

This demo dataset can be used to:
1. Test the basic functionality of the DCP system
2. Verify the installation and configuration
3. Understand the data format and annotation structure

## Expected Results

When running the demo on this dataset, you should expect:
- Detection accuracy: >0.90
- Segmentation accuracy: >0.85
- Processing time: ~1-2 seconds per image 