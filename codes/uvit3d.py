import torch
import torch.nn as nn
import math
try:  # package import (codes.uvit3d)
    from .timm import trunc_normal_, Mlp
except ImportError:  # direct Jupyter/script execution from the codes directory
    from timm import trunc_normal_, Mlp
import einops
from einops import rearrange
import torch.utils.checkpoint

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops

        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')



def timestep_embedding(timesteps, dim):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    device = timesteps.device
    half_dim = dim // 2  # 384
    embeddings = math.log(10000)/(half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
    embeddings = timesteps[:, None] * embeddings[None, :]
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


        return embeddings


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x



def unpatchify(x, spe=104):
    C = spe   # 104
   
    h = w = int(x.shape[1] ** .5)       # 64
    assert h * w == x.shape[1] and C == x.shape[2]
    
    x = einops.rearrange(x, 'B (h w) C -> B 1 C h w', h=h)
    return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.use_skip = skip
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)  # 梯度检查点函数，减少内存占用
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.use_skip:
            if skip is None or skip.shape != x.shape:
                raise ValueError("UViT additive skip connection requires matching token shapes")
            x = x + skip
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# NUM_SPECTRAL = 200    # IP
# NUM_SPECTRAL = 270    # HongHu
NUM_SPECTRAL = 200      # Indian Pines (dataset-specific default)


class UViT(nn.Module):
    def __init__(self, in_chans=1, num_spe=NUM_SPECTRAL, embed_dim=4*NUM_SPECTRAL, patch_size=64, depth=6, num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=True, use_checkpoint=False,
                 conv=True, skip=True):
        super().__init__()

        self.embed_dim = embed_dim  # num_features for consistency with other models

        # self.L = num_tokens
        self.in_chans = in_chans
        self.num_spe = num_spe
        self.patchsize = patch_size

        self.features = []

        # self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_chans, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=8*self.num_spe, out_channels=self.embed_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(),
        )

        self.time_embed = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.SiLU(),
                nn.Linear(4 * embed_dim, embed_dim),
            )

       
        # Input-dependent dynamic positional encoding with a depthwise 3x3 conv.
        self.pos_embed = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3,
                                   padding=1, groups=self.embed_dim)

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = self.num_spe
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = nn.Conv3d(self.in_chans, self.in_chans, 3, padding=1) if conv else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, timesteps, feature=False):
        if x.ndim != 5 or x.shape[1] != self.in_chans or x.shape[2] != self.num_spe:
            raise ValueError(f"UViT expects (B,{self.in_chans},{self.num_spe},H,W), got {tuple(x.shape)}")
        if x.shape[-2:] != (self.patchsize, self.patchsize):
            raise ValueError(f"UViT was configured for {self.patchsize}x{self.patchsize} patches, got {tuple(x.shape[-2:])}")
        if timesteps.ndim != 1 or timesteps.shape[0] != x.shape[0]:
            raise ValueError("timesteps must contain one index per input sample")
       
                x = self.conv3d(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d(x)

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token[(...,) + (None,) * 2]
        x = x + time_token

        x = x + self.pos_embed(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        
        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            residual = skips.pop()
            x = blk(x, residual)
            if feature:
                self.features.append(x.detach())

        x = self.norm(x)
        x = self.decoder_pred(x)
        
        x = unpatchify(x, self.num_spe)    # reshape to 1xSPExHxW
        x = self.final_layer(x)
        return x

    def return_features(self):
        temp_features = []
        temp_features = self.features[:]
    
        self.features = []
        return temp_features


if __name__ == "__main__":
    model = UViT()
    t = torch.full((1,), 100, dtype=torch.long)
    a = torch.randn((20, 1, 104, 64, 64))
    model(a, t)
    
