import torch
import math
import torch.nn as nn

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_groups=8):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attention(h, h, h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, in_channels = 3, channels = 64, network_depth = 4, num_res_block = 2, attention_resolution = 16, image_size = 32, time_emb_dim=256, num_classes = 10):
        super().__init__()
        self.num_res_block = num_res_block
        self.init_conv = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(channels),
            nn.Linear(channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, time_emb_dim) # num_classes + 1 : l'index 10 = "non conditionné" (null token)
        self.num_classes = num_classes

        self.downsamples = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(network_depth):
            if i == network_depth - 1:
                self.downsamples.append(nn.Identity())
                self.upsamples.append(nn.Identity())
            else:       
                self.downsamples.append(Downsample(channels * 2 ** i))
                self.upsamples.append(Upsample(channels * 2 ** (network_depth - 1 - i)))

        self.encoder = nn.ModuleList()
        current_channels = channels
        current_res = image_size
        for depth in range(network_depth):
            out_channels = channels * 2 ** depth
            for i in range(num_res_block):
                resblock =  ResidualBlock(current_channels, out_channels, time_emb_dim)
                attn =  AttentionBlock(out_channels) if current_res == attention_resolution else nn.Identity()
                self.encoder.append(nn.ModuleList([resblock, attn]))
                current_channels = out_channels
            current_res //= 2

        self.bottleneck = nn.ModuleList([
            ResidualBlock(current_channels, current_channels, time_emb_dim),
            AttentionBlock(current_channels),
            ResidualBlock(current_channels, current_channels, time_emb_dim),
        ])

        self.decoder = nn.ModuleList()
        current_res *= 2                # On n'a pas downsamplé au dernier block de l'encoder
        dec_channels = current_channels
        for depth in range(network_depth):
            out_channels = dec_channels // 2 ** depth
            skip_channels = out_channels
            for i in range(num_res_block):
                resblock = ResidualBlock(current_channels + skip_channels, out_channels, time_emb_dim)
                attn =  AttentionBlock(out_channels) if current_res == attention_resolution else nn.Identity()
                self.decoder.append(nn.ModuleList([resblock, attn]))
                current_channels = out_channels
                skip_channels = 0       # Car la concaténation n'est faite qu'avant le premier Resblock
            current_res *= 2

        self.norm = nn.GroupNorm(8, current_channels)
        self.activation = nn.SiLU()
        self.out_conv = nn.Conv2d(current_channels, in_channels, 1)
            
    def forward(self, x, t, class_label = None):
        t_emb = self.time_mlp(t)
        if class_label is not None:
            t_emb = t_emb + self.class_emb(class_label)
        x = self.init_conv(x)
        skip_co = []   
        for i, down in enumerate(self.downsamples):
            for module in [self.encoder[i * self.num_res_block], self.encoder[i * self.num_res_block + 1]]:
                resblock, attn = module
                x = resblock(x, t_emb)
                x = attn(x)
            skip_co.append(x)
            x = down(x)

        x = self.bottleneck[0](x, t_emb)
        x = self.bottleneck[1](x)
        x = self.bottleneck[2](x, t_emb)

        for i, up in enumerate(self.upsamples):
            x = torch.cat([x, skip_co[len(skip_co) - (i + 1)]], dim = 1)
            for module in [self.decoder[i * self.num_res_block], self.decoder[i * self.num_res_block + 1]]:
                resblock, attn = module
                x = resblock(x, t_emb)
                x = attn(x)
            x = up(x)

        x = self.norm(x)
        x = self.activation(x)
        x = self.out_conv(x)

        return x