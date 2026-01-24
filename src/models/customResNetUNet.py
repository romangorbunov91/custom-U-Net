import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Union, List, Tuple, Optional
from models.customResNet import customResNet

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class _customResNetUNet(nn.Module):

    def __init__(self,
                in_channels: int,
                out_channels: int,
                features: Optional[List[int]],
                backbone: nn.Module
                ):
        super(_customResNetUNet, self).__init__()
        
        self.encoder_blocks = backbone.layers
        self.decoder_blocks = nn.ModuleList()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.decoder_blocks.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(
                DoubleConv(feature * 2, feature)
            )

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        print(f"Уровней encoder: {len(features)}")
        print(f"Конфигурация каналов: {features}")
    
    def forward(self, x):

        skip_connections = []

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        # Reverse order.
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder_blocks), 2):
            x = self.decoder_blocks[idx](x)
            
            skip_connection = skip_connections[idx // 2]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], 
                                mode='bilinear', align_corners=True)
            
            x = torch.cat([skip_connection, x], dim=1)
            
            x = self.decoder_blocks[idx + 1](x)

        x = self.final_conv(x)
        output = torch.sigmoid(x)
        return output

def customResNetUNet(
    in_channels: int,
    out_channels: int,
    features: List[int],
    backbone_layers_config: List[int],
    backbone_in_channels: int,
    backbone_layer0_channels: int,
    backbone_pretrained: bool,
    backbone_checkpoints_path: Union[str, Path] = None,
    device: torch.device = None,
    backbone_zero_init_residual: bool = False
    ):
    
    customResNetModel = customResNet(
            layers_config = backbone_layers_config,
            in_channels = backbone_in_channels,
            layer0_channels = backbone_layer0_channels,
            num_classes = None,
            pretrained = backbone_pretrained,
            checkpoints_path = Path(backbone_checkpoints_path),
            device = device,
            zero_init_residual = backbone_zero_init_residual
        )
    
    return _customResNetUNet(
        in_channels = in_channels,
        out_channels = out_channels,
        features = features,
        backbone = customResNetModel
    )