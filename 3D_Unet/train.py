"""
Training script for 3D Prostate segmentation with UNet3D
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from datetime import datetime

from modules import UNet3D, ImprovedUNet3D, dice_coefficient_per_class
from dataset import get_data_loaders


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, D, H, W) predictions (logits)
            target: (B, D, H, W) ground truth labels
        """
        pred = torch.softmax(pred, dim=1)
        
        # One-hot encode target
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        
        # Calculate Dice loss
        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss


class CombinedLoss(nn.Module):
    """Combined Cross Entropy and Dice Loss"""
    
    def __init__(self, weight_ce=0.5, weight_dice=0.5, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.weight_ce * ce + self.weight_dice * dice


class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision Loss for CAN3D
    Combines losses from multiple decoder layers
    """
    
    def __init__(self, weight_ce=0.5, weight_dice=0.5, aux_weights=(0.5, 0.3, 0.2)):
        super(DeepSupervisionLoss, self).__init__()
        self.main_loss = CombinedLoss(weight_ce, weight_dice)
        self.aux_weights = aux_weights
    
    def forward(self, outputs, target):
        """
        Args:
            outputs: tuple of (main_output, aux1, aux2, aux3) or single tensor
            target: ground truth labels
        """
        # If single output (not deep supervision)
        if not isinstance(outputs, tuple):
            return self.main_loss(outputs, target)
        
        # Deep supervision: compute weighted sum of losses
        main_output = outputs[0]
        aux_outputs = outputs[1:]
        
        # Main loss
        loss = self.main_loss(main_output, target)
        
        # Auxiliary losses
        for aux_out, weight in zip(aux_outputs, self.aux_weights):
            aux_loss = self.main_loss(aux_out, target)
            loss += weight * aux_loss
        
        return loss


def train_epoch(model, loader, criterion, optimizer, device, epoch, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    # Check if model uses deep supervision
    use_deep_supervision = getattr(model, 'use_deep_supervision', False)
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # Call model with deep_supervision flag for CAN3D
                if use_deep_supervision and hasattr(model, 'forward'):
                    outputs = model(images, deep_supervision=True)
                else:
                    outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Call model with deep_supervision flag for CAN3D
            if use_deep_supervision and hasattr(model, 'forward'):
                outputs = model(images, deep_supervision=True)
            else:
                outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device, n_classes):
    """Validate the model"""
    model.eval()
    total_loss = 0
    dice_scores = {f'class_{i}': [] for i in range(n_classes)}
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # During validation, always use deep_supervision=False to get only main output
            outputs = model(images, deep_supervision=False) if hasattr(model, 'forward') else model(images)
            
            # Handle tuple output (shouldn't happen with deep_supervision=False, but just in case)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Calculate Dice coefficient per class
            batch_dice = dice_coefficient_per_class(outputs, labels, n_classes)
            for key, value in batch_dice.items():
                dice_scores[key].append(value.item())
    
    avg_loss = total_loss / len(loader)
    avg_dice_scores = {k: np.mean(v) for k, v in dice_scores.items()}
    mean_dice = np.mean(list(avg_dice_scores.values()))
    
    return avg_loss, avg_dice_scores, mean_dice


def train(args):
    """Main training function"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    
    # Save arguments
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Get data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        args.data_dir,
        args.label_dir,
        batch_size=args.batch_size,
        target_shape=tuple(args.target_shape),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers
    )
    
    # Determine number of classes
    sample_images, sample_labels = next(iter(train_loader))
    n_classes = len(torch.unique(sample_labels))
    print(f"Number of classes: {n_classes}")
    print(f"Classes: {sorted(torch.unique(sample_labels).tolist())}")
    
    # Create model
    if args.model == 'unet3d':
        model = UNet3D(n_channels=1, n_classes=n_classes, base_filters=args.base_filters)
    elif args.model == 'improved_unet3d':
        model = ImprovedUNet3D(n_channels=1, n_classes=n_classes, 
                              base_filters=args.base_filters, dropout=args.dropout)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    model.use_deep_supervision = args.use_deep_supervision  # Store for training
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params:,}")
    
    # Setup loss function
    if args.use_deep_supervision:
        print("Using Deep Supervision Loss for CAN3D")
        criterion = DeepSupervisionLoss(weight_ce=args.weight_ce, weight_dice=args.weight_dice,
                                       aux_weights=(0.5, 0.3, 0.2))
    else:
        criterion = CombinedLoss(weight_ce=args.weight_ce, weight_dice=args.weight_dice)
    
    # Setup optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, 
                            weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if args.use_amp and device.type == 'cuda' else None
    
    # Training loop
    best_dice = 0.0
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, 
                                device, epoch, scaler)
        
        # Validate
        val_loss, val_dice_scores, mean_dice = validate(model, val_loader, 
                                                        criterion, device, n_classes)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/mean', mean_dice, epoch)
        for class_name, dice_score in val_dice_scores.items():
            writer.add_scalar(f'Dice/{class_name}', dice_score, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print results
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Mean Dice: {mean_dice:.4f}")
        for class_name, dice_score in val_dice_scores.items():
            print(f"  {class_name}: {dice_score:.4f}")
        
        # Update learning rate
        scheduler.step(mean_dice)
        
        # Save best model
        if mean_dice > best_dice:
            best_dice = mean_dice
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'dice_scores': val_dice_scores,
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"✓ Saved best model (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': mean_dice,
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # Test on test set
    print("\n" + "="*50)
    print("Testing on test set...")
    print("="*50)
    
    # Load best model
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_dice_scores, test_mean_dice = validate(model, test_loader, 
                                                           criterion, device, n_classes)
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Mean Dice: {test_mean_dice:.4f}")
    for class_name, dice_score in test_dice_scores.items():
        print(f"  {class_name}: {dice_score:.4f}")
    
    # Save test results
    test_results = {
        'test_loss': test_loss,
        'mean_dice': test_mean_dice,
        'dice_scores': test_dice_scores
    }
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)
    
    writer.close()
    print(f"\nTraining complete! Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train 3D UNet for Prostate Segmentation')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, 
                       default=r'Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_MRs_anon',
                       help='Path to MRI data directory')
    parser.add_argument('--label_dir', type=str,
                       default=r'Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_labels_anon',
                       help='Path to label directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for models and logs')
    parser.add_argument('--target_shape', type=int, nargs=3, default=[128, 128, 64],
                       help='Target shape for resampling (D H W)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation data ratio')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='improved_unet3d',
                       choices=['unet3d', 'improved_unet3d'],
                       help='Model architecture')
    parser.add_argument('--base_filters', type=int, default=32,
                       help='Number of base filters')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--use_deep_supervision', action='store_true',
                       help='Use deep supervision for CAN3D (improved_unet3d only)')
    parser.add_argument('--aux_weights', type=float, nargs=3, default=[0.5, 0.3, 0.2],
                       help='Weights for auxiliary outputs in deep supervision')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--weight_ce', type=float, default=0.5,
                       help='Weight for cross entropy loss')
    parser.add_argument('--weight_dice', type=float, default=0.5,
                       help='Weight for dice loss')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use automatic mixed precision')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
