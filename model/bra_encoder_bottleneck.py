import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from model.bra_block import Block
from timm.models.layers import to_2tuple

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.conv = nn.Conv2d(dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.bn = nn.BatchNorm2d(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        B, H, W, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B,C,H,W)
        x = self.conv(x)
        x = self.bn(x)
        x = x.view(B, -1, 2 * C)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dim=64):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.in_chans = in_chans
        self.downsample_layers_conv1 = nn.Conv2d(in_chans, embed_dim//2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.downsample_layers_bn1 = nn.BatchNorm2d(embed_dim//2)
        self.downsample_layers_gelu = nn.GELU()
        self.downsample_layers_conv2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.downsample_layers_bn2 = nn.BatchNorm2d(embed_dim)
    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.downsample_layers_conv1(x)
        x = self.downsample_layers_bn1(x)
        x = self.downsample_layers_gelu(x)
        x = self.downsample_layers_conv2(x)
        x = self.downsample_layers_bn2(x).flatten(2).transpose(1,2)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim,input_resolution, depth, embed_dim, num_heads,drop_path_rate=0.,
                 layer_scale_init_value=-1, topks=[8, 8, -1, -1],qk_dims=[96, 192, 384, 768], n_win=7,
                 kv_per_wins=[2, 2, -1, -1], kv_downsample_kernels=[4, 2, 1, 1], kv_downsample_ratios=[4, 2, 1, 1],
                 kv_downsample_mode='ada_avgpool', param_attention='qkvo', param_routing=False, diff_routing=False,
                 soft_routing=False, pre_norm=True, mlp_ratios=[4, 4, 4, 4], mlp_dwconv=False, side_dwconv=5,
                 qk_scale=None, before_attn_dwconv=3, auto_pad=False, norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.layernorm=norm_layer
        self.use_checkpoint = use_checkpoint

        # stochastic depth 随机深度衰减规则
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum([depth]))]
        cur = 0
        # build blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  input_resolution=input_resolution,
                  drop_path=dp_rates[cur + i],
                  layer_scale_init_value=layer_scale_init_value,
                  num_heads=num_heads,
                  n_win=n_win,
                  qk_dim=qk_dims,
                  qk_scale=qk_scale,
                  kv_per_win=kv_per_wins,
                  kv_downsample_ratio=kv_downsample_ratios,
                  kv_downsample_kernel=kv_downsample_kernels,
                  kv_downsample_mode=kv_downsample_mode,
                  topk=topks,
                  param_attention=param_attention,
                  param_routing=param_routing,
                  diff_routing=diff_routing,
                  soft_routing=soft_routing,
                  mlp_ratio=mlp_ratios,
                  mlp_dwconv=mlp_dwconv,
                  side_dwconv=side_dwconv,
                  before_attn_dwconv=before_attn_dwconv,
                  pre_norm=pre_norm,
                  auto_pad=auto_pad)
            for i in range(depth)])
        # patch merging layer
        if downsample is not None:
            self.downsample_layer = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample_layer = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample_layer is not None:
            x = self.downsample_layer(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample_layer is not None:
            flops += self.downsample_layer.flops()
        return flops
