import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def sinusoidal_embedding(t, dim, device):
    half_dim = dim // 2
    # 這裡的 10000 是標準 Transformer/DDPM 設定
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    # 確保 t 是 [Batch, 1] 的整數
    emb = t[:, None].float() * emb[None, :] 
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.same_channels = (in_channels == out_channels)
        self.block = nn.Sequential(
            nn.GroupNorm(8, in_channels),    # GroupNorm
            nn.SiLU(),                       # SiLU
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),   # GroupNorm
            nn.SiLU(),                       # SiLU
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if not self.same_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.block(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool=True):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels)
        self.res2 = ResidualBlock(out_channels, out_channels)
        self.use_pool = use_pool
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        skip1 = self.res1(x)
        skip2 = self.res2(skip1)
        out = skip2
        if self.use_pool:
            out = self.pool(skip2)
        return out, skip1, skip2

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip1_channels, skip2_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.res1 = ResidualBlock(in_channels + skip1_channels, out_channels)
        self.res2 = ResidualBlock(out_channels + skip2_channels, out_channels)

    def forward(self, x, skip1, skip2):
        x = self.upsample(x)
        if x.shape[2:] != skip1.shape[2:]:
            x = F.interpolate(x, size=skip1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip1, x], dim=1)
        x = self.res1(x)

        if x.shape[2:] != skip2.shape[2:]:
            x = F.interpolate(x, size=skip2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip2, x], dim=1)
        x = self.res2(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=9, cond_channels=12, time_dim=32):
        super(UNet, self).__init__()
        self.out_channels = out_channels
        self.time_dim = time_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 32)
        )
        self.target_shape = None
        
        # Encoder
        self.init_conv_channels = 64
        self.down = nn.Conv2d(in_channels + cond_channels, self.init_conv_channels, kernel_size=3, padding=1)
        current_channels = self.init_conv_channels + 32 # 64 + 32 = 96
        self.down1 = DownBlock(current_channels, 64, use_pool=False)
        self.time = ResidualBlock(32, 32)
        self.down1 = DownBlock(current_channels, 64, use_pool=False)  # 第一層通常不下採樣，這裡保留你原本的邏輯
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 384)
        
        # Bottleneck
        self.res = ResidualBlock(384, 384)
        
        # Decoder
        self.up1 = UpBlock(384, 384, 384, 256)
        self.up2 = UpBlock(256, 256, 256, 128)
        self.up3 = UpBlock(128, 128, 128, 64)
        self.up4 = UpBlock(64, 64, 64, 64)     # 新增：對應 down1

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x, cond, t):  # <--- [修正] 移除 beta
        # 不需要傳入 beta，只需要 t
        device = x.device
        
        # 直接對 t 進行 Embedding
        t_emb = sinusoidal_embedding(t, self.time_dim, device)
        t_emb = self.time_embed(t_emb)
        t_emb = t_emb[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))

        # Main Path
        x = torch.cat([x, cond], dim=1)
        x = self.down(x)
        
        t_emb = self.time(t_emb)
        x = torch.cat([x, t_emb], dim=1)
       
        d1, skip1, skip2 = self.down1(x)
        d2, skip3, skip4 = self.down2(d1)
        d3, skip5, skip6 = self.down3(d2)
        d4, skip7, skip8 = self.down4(d3)
        
        bottleneck = self.res(d4)

        u1 = self.up1(bottleneck, skip7, skip8)
        u2 = self.up2(u1, skip5, skip6)
        u3 = self.up3(u2, skip3, skip4)
        u4 = self.up4(u3, skip1, skip2) # 補上最外層的跳躍連接

        output = self.final_conv(u4)

        # Align size if needed
        if hasattr(self, 'target_shape') and self.target_shape is not None: 
            _, _, H, W = self.target_shape
            if output.shape[2:] != (H, W):
                output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)

        return output