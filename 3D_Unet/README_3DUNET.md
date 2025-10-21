# 3D Prostate MRI Segmentation with UNet3D

This project implements 3D UNet for semantic segmentation of prostate MRI scans using the Hip MRI dataset.

## Project Structure

```
PatternAnalysis-2025/
├── dataset.py          # Data loading and augmentation
├── modules.py          # 3D UNet model architectures
├── train.py           # Training script
├── predict.py         # Prediction and evaluation script
├── requirements.txt   # Python dependencies
└── Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-/
    └── data/
        └── HipMRI_study_complete_release_v1/
            ├── semantic_MRs_anon/      # MRI scans
            └── semantic_labels_anon/   # Segmentation labels
```

## Features

### Models
- **UNet3D**: Basic 3D UNet implementation based on the original paper
- **ImprovedUNet3D**: Enhanced version with:
  - Residual connections for better gradient flow
  - Dropout for regularization
  - Optional deep supervision
  - Batch normalization

### Data Processing
- Automatic NIfTI file loading
- Intensity normalization (percentile-based)
- Resampling to target resolution
- Comprehensive data augmentation:
  - Random flipping (3 axes)
  - Random rotation (90°, 180°, 270°)
  - Random intensity shift and scaling
  - Random Gaussian noise

### Training Features
- Combined Cross-Entropy and Dice loss
- Automatic mixed precision training (AMP) support
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping
- TensorBoard logging
- Checkpointing

### Evaluation
- Per-class Dice coefficient calculation
- Sliding window inference for full volumes
- Automatic evaluation on test set

## Installation

### Option 1: Using Conda (Recommended)

1. Create conda environment and install dependencies:
```bash
# Create environment
conda create -n unet3d python=3.10 -y

# Activate environment
conda activate unet3d

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements.txt
```

Or simply run the setup script on Windows:
```bash
setup.bat
```

### Option 2: Using pip

1. Install dependencies:
```bash
pip install -r requirements.txt
```

For CUDA support (recommended for GPU training):
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### 1. Data Preparation

The dataset should be organized as:
- MRI scans: `.nii.gz` files in `semantic_MRs_anon/`
- Labels: corresponding `_SEMANTIC_LFOV.nii.gz` files in `semantic_labels_anon/`

### 2. Training

Basic training with default parameters:
```bash
python train.py
```

Training with custom parameters:
```bash
python train.py \
    --model improved_unet3d \
    --epochs 200 \
    --batch_size 2 \
    --lr 0.001 \
    --base_filters 32 \
    --target_shape 128 128 64 \
    --use_amp \
    --output_dir outputs
```

Key parameters:
- `--model`: Choose between `unet3d` and `improved_unet3d`
- `--epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size (default: 2, adjust based on GPU memory)
- `--lr`: Learning rate (default: 0.001)
- `--base_filters`: Number of base filters (default: 32)
- `--target_shape`: Target shape for resampling [D H W] (default: [128, 128, 64])
- `--use_amp`: Enable automatic mixed precision training
- `--dropout`: Dropout rate for ImprovedUNet3D (default: 0.2)
- `--weight_ce`: Weight for cross-entropy loss (default: 0.5)
- `--weight_dice`: Weight for Dice loss (default: 0.5)
- `--patience`: Early stopping patience (default: 30)

### 3. Evaluation

Evaluate a trained model on the test set:
```bash
python predict.py \
    --checkpoint outputs/run_YYYYMMDD_HHMMSS/best_model.pth \
    --model improved_unet3d \
    --n_classes 5 \
    --save_predictions
```

This will:
- Load the trained model
- Evaluate on the test set
- Calculate Dice coefficients per class
- Check if all classes achieve Dice ≥ 0.7
- Optionally save prediction volumes

### 4. Monitoring Training

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir outputs/run_YYYYMMDD_HHMMSS/tensorboard
```

## Model Architecture

### Basic UNet3D
```
Encoder:
- Conv(1, 32) -> Conv(32, 32)
- MaxPool -> Conv(32, 64) -> Conv(64, 64)
- MaxPool -> Conv(64, 128) -> Conv(128, 128)
- MaxPool -> Conv(128, 256) -> Conv(256, 256)
- MaxPool -> Conv(256, 512) -> Conv(512, 512)

Decoder:
- UpConv(512, 256) + Skip -> Conv(512, 256) -> Conv(256, 256)
- UpConv(256, 128) + Skip -> Conv(256, 128) -> Conv(128, 128)
- UpConv(128, 64) + Skip -> Conv(128, 64) -> Conv(64, 64)
- UpConv(64, 32) + Skip -> Conv(64, 32) -> Conv(32, 32)
- Conv(32, n_classes)
```

### ImprovedUNet3D
- Replaces standard convolution blocks with residual blocks
- Adds dropout after each downsampling/upsampling
- Supports deep supervision during training

## Loss Function

Combined loss = α × CrossEntropy + β × Dice

- **Cross-Entropy Loss**: Penalizes incorrect class predictions
- **Dice Loss**: Directly optimizes the Dice coefficient metric
- Default weights: α = 0.5, β = 0.5

## Data Augmentation

Applied during training:
1. Random flipping along all three axes (p=0.5 each)
2. Random rotation by 90°, 180°, or 270° (p=0.5)
3. Random intensity shift [-0.1, 0.1] and scale [0.9, 1.1] (p=0.5)
4. Random Gaussian noise with σ=0.01 (p=0.5)

## Performance Tips

### For Limited GPU Memory:
- Reduce `--batch_size` to 1
- Reduce `--base_filters` to 16 or 24
- Reduce `--target_shape` to smaller dimensions (e.g., 96 96 48)

### For Better Performance:
- Use `--use_amp` for faster training with mixed precision
- Increase `--base_filters` to 48 or 64 if GPU memory allows
- Use larger `--target_shape` for better resolution (e.g., 160 160 80)
- Tune loss weights with `--weight_ce` and `--weight_dice`

### Training Time Estimates:
- Basic setup (batch_size=2, base_filters=32, shape=128×128×64):
  - RTX 3090: ~10-15 hours for 200 epochs
  - RTX 4090: ~7-10 hours for 200 epochs
  - V100: ~12-18 hours for 200 epochs

## Expected Results

Target: **Dice Coefficient ≥ 0.7** for all segmentation classes

Typical results with ImprovedUNet3D:
- Background: 0.95-0.99
- Prostate: 0.75-0.85
- Other tissues: 0.70-0.80

## Troubleshooting

### Out of Memory (OOM):
- Reduce batch size: `--batch_size 1`
- Reduce model size: `--base_filters 16`
- Reduce input size: `--target_shape 96 96 48`

### Slow Training:
- Enable AMP: `--use_amp`
- Reduce data loading workers if CPU bottleneck: `--num_workers 2`
- Use smaller validation set

### Poor Performance:
- Train longer: `--epochs 300`
- Adjust loss weights: `--weight_dice 0.7 --weight_ce 0.3`
- Increase model capacity: `--base_filters 48`
- Check data loading and augmentation

## References

1. Çiçek, Ö., et al. (2016). "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation." MICCAI.
2. Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI.
3. Isensee, F., et al. (2021). "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature Methods.

## License

This project is for educational purposes as part of COMP3710 at The University of Queensland.
