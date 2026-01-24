import torch
import torch.nn as nn
from pathlib import Path
from .model_utilizer import load_net, update_optimizer
from typing import Union, List, Tuple, Optional

class BasicBlock(nn.Module):
    """
    Базовый блок ResNet с residual connection.
    
    Args:
        in_channels (int): количество входных каналов.
        out_channels (int): количество выходных каналов.
        activation (str): функция активации.
        kernel_size (int): размер ядра свертки (должен быть нечётным).
        stride (int): шаг свертки.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int
        ):

        super(BasicBlock, self).__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve spatial dimensions.")
        
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Identity()
        if (stride != 1) or (in_channels != out_channels):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.activation(out)
        return out

class _customResNet(nn.Module):
    
    def __init__(self,
                 block,
                 layers_config,
                 in_channels: int,
                 layer0_channels: int,
                 num_classes: int,
                 zero_init_residual: bool = False):
        
        super(_customResNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Initial layers.        
        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = layer0_channels,
            kernel_size = 7,
            stride = 2,
            padding = 3,
            bias = False
        )
        
        self.bn1 = nn.BatchNorm2d(layer0_channels)
        self.activation = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Main layers.
        self.layers = nn.ModuleList()
        for order, layer_num in enumerate(layers_config):
            if order == 0:
                extend_flag = False
            else:
                extend_flag = True
            layer = self._make_layer(
                block = block,
                out_channels = layer0_channels * 2**order,
                layer_size = layer_num,
                extend_flag = extend_flag
                )
            self.layers.append(layer)

        if self.num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(
                in_features = layer0_channels * 2**(len(layers_config)-1),
                out_features = num_classes
                )

        # Init weights (optional, but recommended).
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                        module.weight,
                        mode="fan_out",
                        nonlinearity="relu"
                        )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        # Zero-init batch norm in residual branches (improves convergence).
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)
    
    def _make_layer(self,
                    block,
                    out_channels,
                    layer_size: int,
                    extend_flag = True
                    ):
        layers = []
        for idx in range(layer_size):
            if idx == 0:
                if extend_flag:
                    in_channels = out_channels // 2
                else:
                    in_channels = out_channels
                layers.append(block(in_channels, out_channels, kernel_size=3, stride=2))
            else:
                layers.append(block(out_channels, out_channels, kernel_size=3, stride=1))

        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)
        
        for layer in self.layers:
            x = layer(x)
        
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1) # безопасный аналог x.view(x.size(0), -1).
            x = self.fc(x)
        return x
    
def customResNet(
    layers_config: List[int],
    in_channels: int,
    layer0_channels: int,
    num_classes: int,
    pretrained: bool = False,
    checkpoints_file: Union[str, Path] = None,
    model_config = None,
    device: torch.device = None,
    zero_init_residual: bool = False
    ):
    
    net = _customResNet(
        block = BasicBlock,
        layers_config = layers_config,
        in_channels = in_channels,
        layer0_channels = layer0_channels,
        num_classes = num_classes,
        zero_init_residual = zero_init_residual
    )
      
    if not pretrained:
        checkpoints_file = None
    
    net, epoch, optim_dict, _ = load_net(
        net = net,
        checkpoints_file = checkpoints_file,
        device = device
        )

    optimizer = update_optimizer(
        net = net,
        optim = model_config.get('solver_type'),
        decay = model_config.get('weight_decay'),
        lr = model_config.get('base_lr')
        )
    if optim_dict is not None:
        optimizer.load_state_dict(optim_dict)
        print(f"Resuming training 'customResNet' from epoch {epoch} using {model_config.get('solver_type')}.")
    else:
        print(f"Starting training 'customResNet' from scratch using {model_config.get('solver_type')}.")
        
    return net, epoch, optimizer