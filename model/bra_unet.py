# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import torch.nn as nn
from model.bra_unet_system import BRAUnetSystem
logger = logging.getLogger(__name__)

class BRAUnet(nn.Module):
    def __init__(self,img_size=256, num_classes=1, n_win=8, embed_dim=96):
        super(BRAUnet, self).__init__()
        self.bra_unet = BRAUnetSystem(img_size=img_size,
                                in_chans=3,
                                num_classes=num_classes,
                                n_win=n_win,
                                embed_dim=embed_dim,
                                depths=[4,4,8,4],
                                depths_decoder=[4,8,4,4],
                                num_heads=[2,4,8,16],
                                mlp_ratios=[4,4,4,4],
                                drop_rate=0.0,
                                drop_path_rate=0.2,
                                patch_norm=True,
                                topks=[4, 8, 16, -2],
                                kv_per_wins=[-1,-1,-1,-1],
                                kv_downsample_kernels=[4,2,1,1],
                                kv_downsample_ratios=[4,2,1,1],
                                qk_dims=[96,192,384,768])

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.bra_unet(x)
        return logits

 