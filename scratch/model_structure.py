import torch.nn as nn

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        scale = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.se = SqueezeAndExcitation(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        return x

class ResidualDSBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=True):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_ch, out_ch, stride, use_se)
        if stride == 1 and in_ch == out_ch:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        return self.conv(x) + self.skip(x)

class LightSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LightTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LightSelfAttention(dim, num_heads, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class EfficientHybrid5M(nn.Module):
    def __init__(self,
            num_classes: int,
            embed_dim=384,
            transformer_layers=2,
            attn_heads=6
            ):
        super().__init__()

        # Stem: 3x128x128 -> 32x64x64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )

        # Stage 1: 32 -> 64, 64x64
        self.stage1 = nn.Sequential(
            ResidualDSBlock(32, 64, stride=1),
            ResidualDSBlock(64, 64, stride=1)
        )

        # Stage 2: 64 -> 128, 32x32
        self.stage2 = nn.Sequential(
            ResidualDSBlock(64, 128, stride=2),
            ResidualDSBlock(128, 128, stride=1)
        )

        # Stage 3: 128 -> 256, 16x16
        self.stage3 = nn.Sequential(
            ResidualDSBlock(128, 256, stride=2),
            ResidualDSBlock(256, 256, stride=1)
        )

        # Stage 4: 256 -> embed_dim (384), 8x8
        self.stage4 = nn.Sequential(
            ResidualDSBlock(256, embed_dim, stride=2),
            ResidualDSBlock(embed_dim, embed_dim, stride=1)
        )

        self.patch_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)

        # Transformer
        self.transformer = nn.Sequential(
            *[LightTransformerBlock(embed_dim, num_heads=attn_heads, mlp_ratio=2.0, drop=0.1)
              for _ in range(transformer_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.stem(x)          # 32 x 64 x 64
        x = self.stage1(x)        # 64 x 64 x 64
        x = self.stage2(x)        # 128 x 32 x 32
        x = self.stage3(x)        # 256 x 16 x 16
        x = self.stage4(x)        # 384 x 8 x 8

        x = self.patch_proj(x)    # 384 x 8 x 8
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, 64, 384)

        x = self.transformer(x)   # (B, 64, 384)
        x = x.mean(dim=1)         # (B, 384)

        x = self.norm(x)
        x = self.head(x)
        return x