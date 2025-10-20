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


def evaluate_model(model, test_loader, device, n_classes, output_dir=None):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: torch device
        n_classes: Number of classes
        output_dir: Optional directory to save predictions
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_dice_scores = {f'class_{i}': [] for i in range(n_classes)}
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Predict
            outputs = model(images)
            
            # Calculate Dice coefficient per class
            batch_dice = dice_coefficient_per_class(outputs, labels, n_classes)
            for key, value in batch_dice.items():
                all_dice_scores[key].append(value.item())
            
            # Save prediction if output_dir is specified
            if output_dir is not None:
                pred = torch.argmax(outputs, dim=1).cpu().numpy()[0]
                label = labels.cpu().numpy()[0]
                
                # Save as NIfTI
                pred_nii = nib.Nifti1Image(pred.astype(np.int16), affine=np.eye(4))
                label_nii = nib.Nifti1Image(label.astype(np.int16), affine=np.eye(4))
                
                nib.save(pred_nii, os.path.join(output_dir, f'pred_{idx:03d}.nii.gz'))
                nib.save(label_nii, os.path.join(output_dir, f'label_{idx:03d}.nii.gz'))
    
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
    parser.add_argument('--n_classes', type=int, default=5,
                       help='Number of classes')
    parser.add_argument('--base_filters', type=int, default=32,
                       help='Number of base filters')
    parser.add_argument('--target_shape', type=int, nargs=3, default=[128, 128, 64],
                       help='Target shape for resampling (D H W)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save prediction volumes')
    
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
    pred_dir = os.path.join(args.output_dir, 'volumes') if args.save_predictions else None
    if pred_dir:
        os.makedirs(pred_dir, exist_ok=True)
    
    print("\nEvaluating model on test set...")
    results = evaluate_model(model, test_loader, device, args.n_classes, pred_dir)
    
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
