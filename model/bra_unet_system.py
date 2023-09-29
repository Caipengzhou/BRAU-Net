import torch
import torch.nn as nn
from model.bra_decoder_expandx4 import BasicLayer_up,PatchExpand,FinalPatchExpand_X4
from model.bra_encoder_bottleneck import BasicLayer,PatchEmbedding,PatchMerging
from timm.models.layers import trunc_normal_
class BRAUnetSystem(nn.Module):
    def __init__(self, img_size: object = 224, in_chans: object = 3, num_classes: object = 1000, n_win: object = 8,
                 embed_dim: object = 64, depths: object = [2, 2, 2, 2], depths_decoder: object = [1, 2, 2, 2], num_heads: object = [2, 4, 8, 16],
                 topks: object = [1, 4, 16, -2], kv_per_wins: object = [-1, -1, -1, -1], kv_downsample_kernels: object = [4, 2, 1, 1],
                 kv_downsample_ratios: object = [4, 2, 1, 1],
                 mlp_ratios: object = [3, 3, 3, 3], qk_dims: object = [64, 128, 256, 512], drop_rate: object = 0., drop_path_rate: object = 0.1,
                 norm_layer: object = nn.LayerNorm,
                 patch_norm: object = True,
                 final_upsample: object = "expand_first",
                 **kwargs: object) -> object:
        super().__init__()
        print("BRAUnetSystem expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(depths, depths_decoder, drop_path_rate, num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)  # num_layers=4
        self.embed_dim = embed_dim  # 64
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbedding(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim if self.patch_norm else None)

        patches_resolution = [img_size//4, img_size//4]
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)


        self.stages = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                embed_dim=int(embed_dim * 2 ** i_layer),
                num_heads=num_heads[i_layer],
                drop_path_rate=0.2,
                layer_scale_init_value=-1,
                topks=topks[i_layer],
                qk_dims=int(embed_dim * 2 ** i_layer),
                n_win=n_win,
                kv_per_wins=kv_per_wins[i_layer],
                kv_downsample_kernels=kv_downsample_kernels[i_layer],
                kv_downsample_ratios=kv_downsample_ratios[i_layer],
                kv_downsample_mode='identity',
                param_attention='qkvo',
                param_routing=False,
                diff_routing=False,
                soft_routing=False,
                pre_norm=True,
                mlp_ratios=mlp_ratios[i_layer],
                mlp_dwconv=False,
                side_dwconv=5,
                qk_scale=None,
                before_attn_dwconv=3,
                auto_pad=False,
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=False)
            self.stages.append(layer)


        # build decoder layers 搭建解码器
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                                  self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                    patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    depth=depths_decoder[i_layer],
                    embed_dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    num_heads=num_heads[(self.num_layers-1-i_layer)],
                    drop_path_rate=0.2,
                    layer_scale_init_value=-1,
                    topks=topks[3-i_layer],
                    qk_dims=qk_dims[3-i_layer],
                    n_win=n_win,
                    kv_per_wins=kv_per_wins[3-i_layer],
                    kv_downsample_kernels=[3-i_layer],
                    kv_downsample_ratios=[3-i_layer],
                    kv_downsample_mode='identity',
                    param_attention='qkvo',
                    param_routing=False,
                    diff_routing=False,
                    soft_routing=False,
                    pre_norm=True,
                    mlp_ratios=mlp_ratios[3-i_layer],
                    mlp_dwconv=False,
                    side_dwconv=5,
                    qk_scale=None,
                    before_attn_dwconv=3,
                    auto_pad=False,
                    norm_layer=nn.LayerNorm,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=False)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
        self.norm_up = norm_layer(self.embed_dim)


        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // 4, img_size // 4),
                                          dim_scale=4, dim=embed_dim)
        self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x_downsample = []
        x_downsample.append(x)
        x = self.stages[0](x)
        x_downsample.append(x)
        x = self.stages[1](x)
        x_downsample.append(x)
        x = self.stages[2](x)
        x_downsample.append(x)
        x = self.stages[3](x)
        x_downsample.append(x)
        return x, x_downsample


    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        x = self.norm_up(x)  # B L C
        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"
        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)
        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.stages):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
