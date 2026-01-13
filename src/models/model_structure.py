import torch
import torch.nn as nn

def set_activation(activation: str) -> nn.Module:
    if not isinstance(activation, str):
        raise TypeError(f"Activation must be a string, got {type(activation).__name__}")
    match activation.lower():
        case "relu":
            return nn.ReLU(inplace=True)
        case "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01, inplace=True)
        case "elu":
            return nn.ELU(alpha=1.0, inplace=False)
        case "gelu":
            return nn.GELU()
        case _:
            raise ValueError(
                    f"Unsupported activation: '{activation}'. "
                    "Supported: 'relu', 'leaky_relu', 'elu', 'gelu'."
                )

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
            activation: str,
            kernel_size: int,
            stride: int
        ):

        super().__init__()

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
        self.activation = set_activation(activation)
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

class customResNet(nn.Module):
    
    def __init__(self,
                 block,
                 layers_config,
                 activation: str,
                 in_channels: int,
                 layer0_channels: int,
                 num_classes: int,
                 zero_init_residual: bool = False):
        
        super().__init__()
        
        self.layer0_channels = layer0_channels

        # Initial layers.        
        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = self.layer0_channels,
            kernel_size = 7,
            stride = 2,
            padding = 3,
            bias = False
        )
        
        self.bn1 = nn.BatchNorm2d(self.layer0_channels)
        self.activation = set_activation(activation)        
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
                out_channels = self.layer0_channels * 2**order,
                layer_size = layer_num,
                activation = activation,
                extend_flag = extend_flag
                )
            self.layers.append(layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            in_features = self.layer0_channels * 2**(len(layers_config)-1),
            out_features = num_classes
            )

        # Init weights (optional, but recommended).
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if activation == "leaky_relu":
                    leaky_slope = 0.01
                    nn.init.kaiming_normal_(
                        module.weight, 
                        mode="fan_out", 
                        nonlinearity="leaky_relu", 
                        a=leaky_slope
                    )
                # activation == "relu" or other.
                else:
                    nn.init.kaiming_normal_(
                        module.weight,
                        mode="fan_out",
                        nonlinearity="relu"
                        )
                if activation == "elu":
                    # Используем ReLU gain + эмпирический коэффициент
                    # (gain_EL U ≈ 1.55, gain_ReLU = √2 ≈ 1.414 → scale ≈ 1.55/1.414 ≈ 1.096)
                    module.weight.data *= 1.096
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
                    activation: str,
                    extend_flag = True
                    ):
        layers = []
        for idx in range(layer_size):
            if idx == 0:
                if extend_flag:
                    in_channels = out_channels // 2
                else:
                    in_channels = out_channels
                layers.append(block(in_channels, out_channels, activation, kernel_size=3, stride=2))
            else:
                layers.append(block(out_channels, out_channels, activation, kernel_size=3, stride=1))

        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print('input:', x.shape)
        x = self.conv1(x)
        #print('conv1:', x.shape)
        x = self.bn1(x)
        #print('bn1:', x.shape)
        x = self.activation(x)
        #print('activation:', x.shape)
        x = self.maxpool(x)
        #print('maxpool:', x.shape)
        
        for layer in self.layers:
            x = layer(x)
            #print('layer:', x.shape)

        x = self.avgpool(x)
        #print('avgpool:', x.shape)
        x = torch.flatten(x, 1) # безопасный аналог x.view(x.size(0), -1).
        #print('flatten:', x.shape)
        x = self.fc(x)
        #print('output:', x.shape)
        return x
    
def customResNet18(
    num_classes: int,
    layers_config,
    activation: str,
    in_channels: int,
    layer0_channels: int,
    zero_init_residual=False
    ):
    return customResNet(
        block = BasicBlock,
        layers_config = layers_config,
        activation = activation,
        in_channels = in_channels,
        layer0_channels = layer0_channels,
        num_classes = num_classes,
        zero_init_residual = zero_init_residual
    )