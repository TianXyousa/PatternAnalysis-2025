"""
3D UNet architecture for medical image segmentation
Based on the original 3D UNet paper and improved variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: (Conv3D -> BatchNorm -> ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                     padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size,
                     padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, trilinear=False):
        super(Up, self).__init__()
        
        # Use trilinear upsampling or transposed convolution
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, 
                                        kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle padding if sizes don't match
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2,
                       diffZ // 2, diffZ - diffZ // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution layer"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D UNet for volumetric segmentation
    
    Args:
        n_channels: Number of input channels
        n_classes: Number of output classes
        base_filters: Number of filters in the first layer
        trilinear: Use trilinear upsampling instead of transposed conv
    """
    
    def __init__(self, n_channels=1, n_classes=2, base_filters=32, trilinear=False):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        
        # Bottleneck
        factor = 2 if trilinear else 1
        self.down4 = Down(base_filters * 8, base_filters * 16 // factor)
        
        # Decoder
        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, trilinear)
        self.up2 = Up(base_filters * 8, base_filters * 4 // factor, trilinear)
        self.up3 = Up(base_filters * 4, base_filters * 2 // factor, trilinear)
        self.up4 = Up(base_filters * 2, base_filters, trilinear)
        
        # Output
        self.outc = OutConv(base_filters, n_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits


class ImprovedUNet3D(nn.Module):
    """
    Improved 3D UNet with residual connections and deep supervision
    Based on state-of-the-art medical image segmentation architectures
    """
    
    def __init__(self, n_channels=1, n_classes=2, base_filters=32, dropout=0.2):
        super(ImprovedUNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder with residual connections
        self.inc = ResidualBlock(n_channels, base_filters)
        self.down1 = DownResidual(base_filters, base_filters * 2, dropout)
        self.down2 = DownResidual(base_filters * 2, base_filters * 4, dropout)
        self.down3 = DownResidual(base_filters * 4, base_filters * 8, dropout)
        self.down4 = DownResidual(base_filters * 8, base_filters * 16, dropout)
        
        # Decoder with attention
        self.up1 = UpResidual(base_filters * 16, base_filters * 8, dropout)
        self.up2 = UpResidual(base_filters * 8, base_filters * 4, dropout)
        self.up3 = UpResidual(base_filters * 4, base_filters * 2, dropout)
        self.up4 = UpResidual(base_filters * 2, base_filters, dropout)
        
        # Output
        self.outc = OutConv(base_filters, n_classes)
        
        # Deep supervision outputs (optional)
        self.out1 = OutConv(base_filters * 8, n_classes)
        self.out2 = OutConv(base_filters * 4, n_classes)
        self.out3 = OutConv(base_filters * 2, n_classes)
    
    def forward(self, x, deep_supervision=False):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        d4 = self.up1(x5, x4)
        d3 = self.up2(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up4(d2, x1)
        
        # Main output
        out = self.outc(d1)
        
        if deep_supervision and self.training:
            # Additional outputs for deep supervision
            out1 = F.interpolate(self.out1(d4), size=out.shape[2:], 
                               mode='trilinear', align_corners=True)
            out2 = F.interpolate(self.out2(d3), size=out.shape[2:],
                               mode='trilinear', align_corners=True)
            out3 = F.interpolate(self.out3(d2), size=out.shape[2:],
                               mode='trilinear', align_corners=True)
            return out, out1, out2, out3
        
        return out


class ResidualBlock(nn.Module):
    """Residual block with two convolutions"""
    
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class DownResidual(nn.Module):
    """Downsampling with residual block"""
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(DownResidual, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.res_block = ResidualBlock(in_channels, out_channels)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else None
    
    def forward(self, x):
        x = self.maxpool(x)
        x = self.res_block(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class UpResidual(nn.Module):
    """Upsampling with residual block"""
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UpResidual, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.res_block = ResidualBlock(in_channels, out_channels)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else None
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle padding if sizes don't match
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2,
                       diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        x = self.res_block(x)
        if self.dropout:
            x = self.dropout(x)
        return x


def dice_coefficient(pred, target, smooth=1e-5):
    """
    Calculate Dice coefficient for evaluation
    
    Args:
        pred: Predicted segmentation (after argmax)
        target: Ground truth segmentation
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice


def dice_coefficient_per_class(pred, target, n_classes, smooth=1e-5):
    """
    Calculate Dice coefficient for each class
    
    Args:
        pred: Predicted segmentation (B, C, D, H, W) logits
        target: Ground truth segmentation (B, D, H, W)
        n_classes: Number of classes
        smooth: Smoothing factor
    
    Returns:
        Dictionary with dice score per class
    """
    pred = torch.argmax(pred, dim=1)
    dice_scores = {}
    
    for i in range(n_classes):
        pred_i = (pred == i).float()
        target_i = (target == i).float()
        dice_scores[f'class_{i}'] = dice_coefficient(pred_i, target_i, smooth)
    
    return dice_scores


if __name__ == '__main__':
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test basic UNet3D
    model = UNet3D(n_channels=1, n_classes=5, base_filters=32)
    model = model.to(device)
    
    # Test forward pass
    x = torch.randn(1, 1, 64, 64, 64).to(device)
    out = model(x)
    print(f"UNet3D output shape: {out.shape}")
    
    # Test improved UNet3D
    model_improved = ImprovedUNet3D(n_channels=1, n_classes=5, base_filters=32)
    model_improved = model_improved.to(device)
    
    out = model_improved(x)
    print(f"ImprovedUNet3D output shape: {out.shape}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters (UNet3D): {n_params:,}")
    
    n_params_improved = sum(p.numel() for p in model_improved.parameters() if p.requires_grad)
    print(f"Number of parameters (ImprovedUNet3D): {n_params_improved:,}")
