"""
Dataset loader for 3D Prostate MRI segmentation
Loads NIfTI format files and applies data augmentation using MONAI
"""

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
import random

# MONAI imports for medical image transforms
from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate90,
    RandScaleIntensity,
    RandShiftIntensity,
    RandGaussianNoise,
    RandAffine,
    Rand3DElastic
)


class ProstateDataset(Dataset):
    """Dataset class for loading 3D prostate MRI scans and labels"""
    
    def __init__(self, data_dir, label_dir, case_ids=None, transform=None, 
                 target_shape=(128, 128, 64), augment=False):
        """
        Args:
            data_dir: Directory containing MRI scans (.nii.gz files)
            label_dir: Directory containing segmentation labels
            case_ids: List of case IDs to include (for train/val/test split)
            transform: Optional transform to apply
            target_shape: Target shape for resampling (depth, height, width)
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_shape = target_shape
        self.augment = augment
        
        # Setup MONAI transforms for augmentation
        # Create separate transforms for image (bilinear) and label (nearest)
        if self.augment:
            # Spatial transforms for IMAGE (use bilinear interpolation)
            self.spatial_transforms_image = Compose([
                # Basic flips and rotations
                RandFlip(spatial_axis=0, prob=0.5),
                RandFlip(spatial_axis=1, prob=0.5),
                RandFlip(spatial_axis=2, prob=0.5),
                RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                
                # Affine transformations (rotation, scaling, translation)
                RandAffine(
                    prob=0.3,
                    rotate_range=(0.1, 0.1, 0.1),  # ±5.7 degrees in radians
                    scale_range=(0.1, 0.1, 0.1),   # ±10% scaling
                    translate_range=(10, 10, 5),   # small translations in voxels
                    mode='bilinear',
                    padding_mode='border',
                ),
                
                # Elastic deformation
                Rand3DElastic(
                    prob=0.3,
                    sigma_range=(5, 7),
                    magnitude_range=(50, 150),
                    mode='bilinear',
                    padding_mode='border',
                ),
            ])
            
            # Spatial transforms for LABEL (use nearest-neighbor)
            self.spatial_transforms_label = Compose([
                # Same transforms but with nearest-neighbor interpolation
                RandFlip(spatial_axis=0, prob=0.5),
                RandFlip(spatial_axis=1, prob=0.5),
                RandFlip(spatial_axis=2, prob=0.5),
                RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                
                RandAffine(
                    prob=0.3,
                    rotate_range=(0.1, 0.1, 0.1),
                    scale_range=(0.1, 0.1, 0.1),
                    translate_range=(10, 10, 5),
                    mode='nearest',  # Key difference: nearest for labels
                    padding_mode='border',
                ),
                
                Rand3DElastic(
                    prob=0.3,
                    sigma_range=(5, 7),
                    magnitude_range=(50, 150),
                    mode='nearest',  # Key difference: nearest for labels
                    padding_mode='border',
                ),
            ])
            
            # Intensity transforms (apply to image only)
            self.intensity_transforms = Compose([
                RandScaleIntensity(factors=0.1, prob=0.5),
                RandShiftIntensity(offsets=0.1, prob=0.5),
                RandGaussianNoise(prob=0.5, mean=0.0, std=0.01),
            ])
        else:
            self.spatial_transforms_image = None
            self.spatial_transforms_label = None
            self.intensity_transforms = None
        
        # Get all available files
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
        
        # Filter by case_ids if provided
        if case_ids is not None:
            self.files = []
            for case_id in case_ids:
                case_files = [f for f in all_files if f.startswith(f'Case_{case_id:03d}_')]
                self.files.extend(case_files)
        else:
            self.files = all_files
            
        self.files = sorted(self.files)
        print(f"Found {len(self.files)} files")
        print(f"Augmentation: {'MONAI transforms enabled' if self.augment else 'disabled'}")
        
    def __len__(self):
        return len(self.files)
    
    def load_nifti(self, filepath):
        """Load NIfTI file and return numpy array"""
        nii = nib.load(filepath)
        data = nii.get_fdata()
        return data
    
    def normalize(self, volume):
        """Normalize volume to [0, 1] range"""
        volume = volume.astype(np.float32)
        # Clip outliers
        p1, p99 = np.percentile(volume, (1, 99))
        volume = np.clip(volume, p1, p99)
        # Normalize to [0, 1]
        if volume.max() > volume.min():
            volume = (volume - volume.min()) / (volume.max() - volume.min())
        return volume
    
    def resample(self, volume, target_shape, is_label=False):
        """
        Resample volume to target shape
        
        Args:
            volume: Input volume to resample
            target_shape: Target shape (D, H, W)
            is_label: If True, use nearest-neighbor interpolation (order=0)
                     If False, use linear interpolation (order=1)
        """
        factors = [t/s for t, s in zip(target_shape, volume.shape)]
        order = 0 if is_label else 1  # Nearest-neighbor for labels, linear for images
        return zoom(volume, factors, order=order)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        # Construct label filename
        label_name = img_name.replace('.nii.gz', '_SEMANTIC_LFOV.nii.gz')
        if 'LFOV.nii.gz' in img_name and 'SEMANTIC' not in img_name:
            label_name = img_name.replace('_LFOV.nii.gz', '_SEMANTIC_LFOV.nii.gz')
        label_path = os.path.join(self.label_dir, label_name)
        
        # Load data
        image = self.load_nifti(img_path)
        label = self.load_nifti(label_path)
        
        # Normalize image
        image = self.normalize(image)
        
        # Resample to target shape (use nearest-neighbor for label)
        image = self.resample(image, self.target_shape, is_label=False)
        label = self.resample(label, self.target_shape, is_label=True)
        
        # Round label to nearest integer (for segmentation classes)
        label = np.round(label).astype(np.int64)
        
        # Add channel dimension for image
        image = image[np.newaxis, ...]  # (1, D, H, W)
        
        # Apply MONAI augmentation if enabled
        if self.augment and self.spatial_transforms_image is not None:
            # Set random seed to ensure image and label get same spatial transforms
            import random
            seed = random.randint(0, 2**32 - 1)
            
            # Apply spatial transforms to image (bilinear interpolation)
            self.spatial_transforms_image.set_random_state(seed=seed)
            image = self.spatial_transforms_image(image)
            
            # Apply spatial transforms to label (nearest-neighbor interpolation)
            self.spatial_transforms_label.set_random_state(seed=seed)
            label_with_channel = label[np.newaxis, ...]  # Add channel for transform
            label_with_channel = self.spatial_transforms_label(label_with_channel)
            label = label_with_channel[0]  # Remove channel dimension
            
            # Apply intensity transforms to image only
            image = self.intensity_transforms(image)
        
        # Convert to tensors (MONAI transforms may already return tensors)
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.ascontiguousarray(image)).float()
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(np.ascontiguousarray(label)).long()
        
        # Ensure correct types
        image = image.float()
        label = label.long()
        
        return image, label


def get_data_loaders(data_dir, label_dir, batch_size=2, target_shape=(128, 128, 64),
                     train_ratio=0.7, val_ratio=0.15, num_workers=4):
    """
    Create train, validation, and test data loaders
    
    Args:
        data_dir: Directory containing MRI scans
        label_dir: Directory containing labels
        batch_size: Batch size for training
        target_shape: Target shape for resampling
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Get unique case IDs
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
    case_ids = sorted(list(set([int(f.split('_')[1]) for f in all_files])))
    
    print(f"Total cases: {len(case_ids)}")
    print(f"Case IDs: {case_ids}")
    
    # Split into train/val/test
    random.seed(42)
    random.shuffle(case_ids)
    
    n_train = int(len(case_ids) * train_ratio)
    n_val = int(len(case_ids) * val_ratio)
    
    train_ids = case_ids[:n_train]
    val_ids = case_ids[n_train:n_train + n_val]
    test_ids = case_ids[n_train + n_val:]
    
    print(f"Train cases: {len(train_ids)} - {train_ids}")
    print(f"Val cases: {len(val_ids)} - {val_ids}")
    print(f"Test cases: {len(test_ids)} - {test_ids}")
    
    # Create datasets
    train_dataset = ProstateDataset(data_dir, label_dir, train_ids, 
                                   target_shape=target_shape, augment=True)
    val_dataset = ProstateDataset(data_dir, label_dir, val_ids,
                                 target_shape=target_shape, augment=False)
    test_dataset = ProstateDataset(data_dir, label_dir, test_ids,
                                  target_shape=target_shape, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1,
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset loading
    data_dir = r'Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_MRs_anon'
    label_dir = r'Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_labels_anon'
    
    train_loader, val_loader, test_loader = get_data_loaders(data_dir, label_dir, batch_size=2)
    
    print(f"\nDataset sizes:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test loading one batch
    for images, labels in train_loader:
        print(f"\nBatch shapes:")
        print(f"Images: {images.shape}")
        print(f"Labels: {labels.shape}")
        print(f"Unique labels: {torch.unique(labels)}")
        break
