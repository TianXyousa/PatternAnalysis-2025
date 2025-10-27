"""
Prediction script for 3D Prostate segmentation
"""

import os
import argparse
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from modules import UNet3D, ImprovedUNet3D, dice_coefficient_per_class
from dataset import ProstateDataset


def predict_volume(model, volume, device, patch_size=(128, 128, 64), overlap=16):
    """
    Predict segmentation for a full volume using sliding window approach
    
    Args:
        model: Trained model
        volume: Input volume (D, H, W)
        device: torch device
        patch_size: Size of patches for prediction
        overlap: Overlap between patches
    
    Returns:
        Predicted segmentation
    """
    model.eval()
    
    D, H, W = volume.shape
    pD, pH, pW = patch_size
    
    # Create output array
    prediction = np.zeros((model.n_classes, D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.float32)
    
    # Calculate stride
    stride_d = max(pD - overlap, 1)
    stride_h = max(pH - overlap, 1)
    stride_w = max(pW - overlap, 1)
    
    with torch.no_grad():
        # Sliding window
        for d in range(0, D, stride_d):
            for h in range(0, H, stride_h):
                for w in range(0, W, stride_w):
                    # Extract patch
                    d_end = min(d + pD, D)
                    h_end = min(h + pH, H)
                    w_end = min(w + pW, W)
                    
                    patch = volume[d:d_end, h:h_end, w:w_end]
                    
                    # Pad if necessary
                    if patch.shape != patch_size:
                        pad_d = pD - patch.shape[0]
                        pad_h = pH - patch.shape[1]
                        pad_w = pW - patch.shape[2]
                        patch = np.pad(patch, ((0, pad_d), (0, pad_h), (0, pad_w)), 
                                     mode='constant', constant_values=0)
                    
                    # Predict
                    patch_tensor = torch.from_numpy(patch[np.newaxis, np.newaxis, ...]).float().to(device)
                    output = model(patch_tensor)
                    output = torch.softmax(output, dim=1)
                    output = output.cpu().numpy()[0]
                    
                    # Remove padding
                    if d_end - d != pD or h_end - h != pH or w_end - w != pW:
                        output = output[:, :(d_end-d), :(h_end-h), :(w_end-w)]
                    
                    # Add to prediction
                    prediction[:, d:d_end, h:h_end, w:w_end] += output
                    count[d:d_end, h:h_end, w:w_end] += 1
    
    # Average overlapping predictions
    prediction = prediction / (count[np.newaxis, ...] + 1e-7)
    prediction = np.argmax(prediction, axis=0)
    
    return prediction


def create_visualization(image, ground_truth, prediction, slice_idx, dice_scores, 
                         output_path, class_names=None):
    """
    Create visualization comparing ground truth and prediction
    
    Args:
        image: Input image volume (D, H, W)
        ground_truth: Ground truth labels (D, H, W)
        prediction: Predicted labels (D, H, W)
        slice_idx: Slice index to visualize
        dice_scores: Dictionary of dice scores per class
        output_path: Path to save the visualization
        class_names: Optional list of class names
    """
    # Define colors for each class
    colors = [
        [0, 0, 0],        # Class 0: Background (black)
        [1, 0, 0],        # Class 1: Red
        [0, 1, 0],        # Class 2: Green
        [0, 0, 1],        # Class 3: Blue
        [1, 1, 0],        # Class 4: Yellow
        [1, 0, 1],        # Class 5: Magenta
        [0, 1, 1],        # Class 6: Cyan
    ]
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(colors))]
    
    # Create RGB images from labels
    def label_to_rgb(label):
        rgb = np.zeros((*label.shape, 3))
        for i, color in enumerate(colors):
            if i < label.max() + 1:
                rgb[label == i] = color
        return rgb
    
    # Get the slice
    img_slice = image[slice_idx]
    gt_slice = ground_truth[slice_idx]
    pred_slice = prediction[slice_idx]
    
    # Create RGB versions
    gt_rgb = label_to_rgb(gt_slice)
    pred_rgb = label_to_rgb(pred_slice)
    
    # Create overlay (50% image + 50% label)
    img_normalized = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-7)
    img_rgb = np.stack([img_normalized]*3, axis=-1)
    
    gt_overlay = 0.6 * img_rgb + 0.4 * gt_rgb
    pred_overlay = 0.6 * img_rgb + 0.4 * pred_rgb
    
    # Calculate error map (where prediction differs from ground truth)
    error_map = (gt_slice != pred_slice).astype(float)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Segmentation Results - Slice {slice_idx}', fontsize=16, fontweight='bold')
    
    # Row 1: Original image, Ground truth, Prediction
    axes[0, 0].imshow(img_slice, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_rgb)
    axes[0, 1].set_title('Ground Truth', fontsize=14)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_rgb)
    axes[0, 2].set_title('Prediction', fontsize=14)
    axes[0, 2].axis('off')
    
    # Row 2: GT Overlay, Prediction Overlay, Error Map
    axes[1, 0].imshow(gt_overlay)
    axes[1, 0].set_title('Ground Truth Overlay', fontsize=14)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_overlay)
    axes[1, 1].set_title('Prediction Overlay', fontsize=14)
    axes[1, 1].axis('off')
    
    error_display = axes[1, 2].imshow(error_map, cmap='Reds', vmin=0, vmax=1)
    axes[1, 2].set_title('Error Map (Red = Wrong)', fontsize=14)
    axes[1, 2].axis('off')
    plt.colorbar(error_display, ax=axes[1, 2], fraction=0.046)
    
    # Add legend for classes
    unique_classes = np.unique(np.concatenate([gt_slice.flatten(), pred_slice.flatten()]))
    patches = [mpatches.Patch(color=colors[i], label=f'{class_names[i]} (Dice: {dice_scores.get(f"class_{i}", 0):.3f})') 
               for i in unique_classes if i < len(class_names)]
    fig.legend(handles=patches, loc='lower center', ncol=min(len(patches), 6), 
               fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_multi_slice_visualization(image, ground_truth, prediction, dice_scores,
                                     output_path, num_slices=5, class_names=None):
    """
    Create visualization showing multiple slices
    
    Args:
        image: Input image volume (C, D, H, W) or (D, H, W)
        ground_truth: Ground truth labels (D, H, W)
        prediction: Predicted labels (D, H, W)
        dice_scores: Dictionary of dice scores per class
        output_path: Path to save the visualization
        num_slices: Number of slices to show
        class_names: Optional list of class names
    """
    # Handle different input shapes
    if len(image.shape) == 4:
        image = image[0]  # Remove channel dimension
    
    depth = image.shape[0]
    slice_indices = np.linspace(depth // 6, depth - depth // 6, num_slices, dtype=int)
    
    fig, axes = plt.subplots(num_slices, 4, figsize=(16, 4*num_slices))
    if num_slices == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Multi-Slice Segmentation Results', fontsize=16, fontweight='bold')
    
    colors = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    
    def label_to_rgb(label):
        rgb = np.zeros((*label.shape, 3))
        for i, color in enumerate(colors):
            if i < label.max() + 1:
                rgb[label == i] = color
        return rgb
    
    for idx, slice_idx in enumerate(slice_indices):
        img_slice = image[slice_idx]
        gt_slice = ground_truth[slice_idx]
        pred_slice = prediction[slice_idx]
        
        # Normalize image
        img_normalized = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-7)
        
        # Original image
        axes[idx, 0].imshow(img_slice, cmap='gray')
        axes[idx, 0].set_title(f'Slice {slice_idx}', fontsize=12)
        axes[idx, 0].axis('off')
        
        # Ground truth
        axes[idx, 1].imshow(label_to_rgb(gt_slice))
        axes[idx, 1].set_title('Ground Truth', fontsize=12)
        axes[idx, 1].axis('off')
        
        # Prediction
        axes[idx, 2].imshow(label_to_rgb(pred_slice))
        axes[idx, 2].set_title('Prediction', fontsize=12)
        axes[idx, 2].axis('off')
        
        # Error map
        error_map = (gt_slice != pred_slice).astype(float)
        axes[idx, 3].imshow(error_map, cmap='Reds', vmin=0, vmax=1)
        axes[idx, 3].set_title(f'Error: {error_map.mean()*100:.1f}%', fontsize=12)
        axes[idx, 3].axis('off')
    
    # Add overall dice scores
    dice_text = "Overall Dice Scores:\n"
    for key, value in sorted(dice_scores.items()):
        dice_text += f"{key}: {value:.3f}\n"
    
    fig.text(0.02, 0.02, dice_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_model(model, test_loader, device, n_classes, output_dir=None):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: torch device
        n_classes: Number of classes
        output_dir: Optional directory to save predictions and visualizations
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_dice_scores = {f'class_{i}': [] for i in range(n_classes)}
    
    # Create visualization directory
    viz_dir = None
    if output_dir is not None:
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Predict
            outputs = model(images)
            
            # Handle deep supervision (take main output)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Calculate Dice coefficient per class
            batch_dice = dice_coefficient_per_class(outputs, labels, n_classes)
            for key, value in batch_dice.items():
                all_dice_scores[key].append(value.item())
            
            # Get predictions and move to CPU
            pred = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            label = labels.cpu().numpy()[0]
            image = images.cpu().numpy()[0, 0]  # Remove batch and channel dims
            
            # Save prediction if output_dir is specified
            if output_dir is not None:
                # Save as NIfTI
                pred_nii = nib.Nifti1Image(pred.astype(np.int16), affine=np.eye(4))
                label_nii = nib.Nifti1Image(label.astype(np.int16), affine=np.eye(4))
                
                vol_dir = os.path.join(output_dir, 'volumes')
                os.makedirs(vol_dir, exist_ok=True)
                nib.save(pred_nii, os.path.join(vol_dir, f'pred_{idx:03d}.nii.gz'))
                nib.save(label_nii, os.path.join(vol_dir, f'label_{idx:03d}.nii.gz'))
                
                # Create visualizations
                if viz_dir is not None:
                    # Single slice visualization (middle slice)
                    mid_slice = pred.shape[0] // 2
                    vis_path = os.path.join(viz_dir, f'comparison_{idx:03d}_slice_{mid_slice}.png')
                    create_visualization(
                        image, label, pred, mid_slice, batch_dice, vis_path
                    )
                    
                    # Multi-slice visualization
                    multi_vis_path = os.path.join(viz_dir, f'multi_slice_{idx:03d}.png')
                    create_multi_slice_visualization(
                        image, label, pred, batch_dice, multi_vis_path, num_slices=5
                    )
            
            # Update progress bar with current dice
            mean_dice_current = np.mean([v.item() for v in batch_dice.values()])
            pbar.set_postfix({'dice': f'{mean_dice_current:.4f}'})
    
    # Calculate mean scores
    avg_dice_scores = {k: np.mean(v) for k, v in all_dice_scores.items()}
    mean_dice = np.mean(list(avg_dice_scores.values()))
    
    # Calculate std
    std_dice_scores = {k: np.std(v) for k, v in all_dice_scores.items()}
    std_mean_dice = np.std([np.mean(list(all_dice_scores.values()))])
    
    results = {
        'mean_dice': mean_dice,
        'std_mean_dice': std_mean_dice,
        'dice_per_class': avg_dice_scores,
        'std_per_class': std_dice_scores,
        'all_scores': all_dice_scores
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Predict with trained 3D UNet')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str,
                       default=r'Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_MRs_anon',
                       help='Path to MRI data directory')
    parser.add_argument('--label_dir', type=str,
                       default=r'Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_labels_anon',
                       help='Path to label directory')
    parser.add_argument('--output_dir', type=str, default='predictions',
                       help='Output directory for predictions')
    parser.add_argument('--model', type=str, default='improved_unet3d',
                       choices=['unet3d', 'improved_unet3d'],
                       help='Model architecture')
    parser.add_argument('--n_classes', type=int, default=6,
                       help='Number of classes')
    parser.add_argument('--base_filters', type=int, default=32,
                       help='Number of base filters')
    parser.add_argument('--target_shape', type=int, nargs=3, default=[128, 128, 64],
                       help='Target shape for resampling (D H W)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save prediction volumes')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Create visualization images (default: True)')
    parser.add_argument('--num_vis_slices', type=int, default=5,
                       help='Number of slices to show in multi-slice visualization')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    if args.model == 'unet3d':
        model = UNet3D(n_channels=1, n_classes=args.n_classes, base_filters=args.base_filters)
    elif args.model == 'improved_unet3d':
        model = ImprovedUNet3D(n_channels=1, n_classes=args.n_classes, 
                              base_filters=args.base_filters)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded. Best Dice from training: {checkpoint.get('best_dice', 'N/A')}")
    
    # Load test data
    from dataset import get_data_loaders
    _, _, test_loader = get_data_loaders(
        args.data_dir,
        args.label_dir,
        batch_size=args.batch_size,
        target_shape=tuple(args.target_shape),
        num_workers=2
    )
    
    # Evaluate
    output_dir_for_eval = args.output_dir if (args.save_predictions or args.visualize) else None
    
    print("\nEvaluating model on test set...")
    results = evaluate_model(model, test_loader, device, args.n_classes, output_dir_for_eval)
    
    # Print results
    print("\n" + "="*50)
    print("Test Results")
    print("="*50)
    print(f"Mean Dice Coefficient: {results['mean_dice']:.4f} ± {results['std_mean_dice']:.4f}")
    print("\nDice per class:")
    for class_name in sorted(results['dice_per_class'].keys()):
        dice = results['dice_per_class'][class_name]
        std = results['std_per_class'][class_name]
        print(f"  {class_name}: {dice:.4f} ± {std:.4f}")
    
    # Check if minimum Dice coefficient is met
    min_dice = min(results['dice_per_class'].values())
    print(f"\nMinimum Dice across all classes: {min_dice:.4f}")
    if min_dice >= 0.7:
        print("✓ SUCCESS: All classes have Dice coefficient >= 0.7")
    else:
        print("✗ FAIL: Some classes have Dice coefficient < 0.7")
        failing_classes = [k for k, v in results['dice_per_class'].items() if v < 0.7]
        print(f"  Failing classes: {failing_classes}")
    
    # Print output locations
    if args.save_predictions:
        print(f"\n📁 Prediction volumes saved to: {os.path.join(args.output_dir, 'volumes')}")
    if args.visualize:
        print(f"📊 Visualizations saved to: {os.path.join(args.output_dir, 'visualizations')}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    # Convert numpy types to Python types for JSON serialization
    results_serializable = {
        'mean_dice': float(results['mean_dice']),
        'std_mean_dice': float(results['std_mean_dice']),
        'dice_per_class': {k: float(v) for k, v in results['dice_per_class'].items()},
        'std_per_class': {k: float(v) for k, v in results['std_per_class'].items()},
        'min_dice': float(min_dice),
        'success': bool(min_dice >= 0.7)
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=4)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
