import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
from .customResNet import customResNet

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
                out_channels: int,
                features: List[int],
                backbone: nn.Module
                ):
        super(_customResNetUNet, self).__init__()

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        # Use the input of `ResNet` as the very 1st encoder.
        self.encoder_blocks.append(
            nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.activation,
                backbone.maxpool
            )
        )
        # Skip layer[0].
        for layer in backbone.layers[1:]:
            self.encoder_blocks.append(layer)

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.decoder_blocks.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(
                DoubleConv(feature * 2, feature)
            )

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        print(f"Encoder features by level: {features}")
    
    def forward(self, x):

        skip_connections = []
        
        for idx, encoder_block in enumerate(self.encoder_blocks):
            encoder_in_shape = x.shape
            x = encoder_block(x)
            skip_connections.append(
                F.interpolate(x, size=encoder_in_shape[2:],
                            mode='bilinear', align_corners=True)
                            )
        
        x = self.bottleneck(x)
        
        # Reverse order.
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder_blocks), 2):
            x = self.decoder_blocks[idx](x)
            
            skip_connection = skip_connections[idx // 2]
                     
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
    backbone_layer0_channels: int,
    backbone_pretrained: bool,
    backbone_checkpoints_file: Union[str, Path] = None,
    device: torch.device = None,
    backbone_zero_init_residual: bool = False
    ):
    
    customResNetModel, _, _ = customResNet(
            layers_config = backbone_layers_config,
            in_channels = in_channels,
            layer0_channels = backbone_layer0_channels,
            num_classes = None,
            pretrained = backbone_pretrained,
            checkpoints_file = None if backbone_checkpoints_file is None else Path(backbone_checkpoints_file),
            device = device,
            zero_init_residual = backbone_zero_init_residual
        )
    customResNetModel = customResNetModel.to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        customResNetModel = nn.DataParallel(customResNetModel)

    return _customResNetUNet(
        out_channels = out_channels,
        features = features,
        backbone = customResNetModel
    )