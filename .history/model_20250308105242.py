import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.in_norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.in_norm(out)
        out = self.relu(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return out

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.in_norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        out = self.conv(x)
        out = self.in_norm(out)
        out = self.relu(out)
        return out

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        
        # Encoder
        self.enc1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.enc2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.enc3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        
        # Residual blocks
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        
        # Decoder
        self.dec1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.dec2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.dec3 = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)
        
    def forward(self, x):
        # Encode
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        
        # Transform
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        
        # Decode
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        return x