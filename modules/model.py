import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLocalizationModel(nn.Module):
    def __init__(self, input_channels=12):
        super(SimpleLocalizationModel, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final layer
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        # Encoding with pooling
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, 2)
        
        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, 2)
        
        e3 = self.enc3(p2)
        
        # Decoding with upsampling
        d1 = self.dec1(e3)
        u1 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        
        d2 = self.dec2(u1)
        u2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Final output
        output = torch.sigmoid(self.final(u2))
        
        return output