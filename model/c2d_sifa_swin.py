from functools import partial

import torch
import torch.nn as nn
import numpy as np

from torch.nn.modules.utils import _pair

from layer.drop import DropPath
from .c2d_swin_vit import C2D_SWIN_ViT
from .swin_vit import BasicLayer
from .swin_vit import PatchMerging
from .swin_vit import SwinTransformerBlock
from .swin_vit import window_partition, window_reverse
from model.defcor import DefAgg, DefCorFixW

from .model_factory import register_model


class SIFASwinTransformerBlock(SwinTransformerBlock):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, K=3, cor_dilation=1, cor_group=1, clip_len=4, fix_offset=True):
        super().__init__(dim, input_resolution, num_heads, window_size, shift_size,
                         mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path,
                         act_layer, norm_layer)
        pad_num = (cor_dilation * (K - 1) + 1) // 2
        self.off_channels_ = 2 * K * K
        self.kernel_size = _pair(K)
        self.width = dim
        self.clip_len = clip_len

        ### inter-frame attention ###
        #self.conv_offset = nn.Conv3d(self.width, self.off_channels_, 
        #                         kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        #self.conv_offset.weight.data.zero_()
        self.def_cor = DefCorFixW(in_channels=self.width, times=clip_len, kernel_size=(K,K), stride=1, padding=pad_num, 
                          dilation=cor_dilation, defcor_groups=cor_group)
        self.def_agg = DefAgg(in_channels=self.width, times=clip_len, kernel_size=(K,K), stride=1, padding=pad_num, 
                          dilation=cor_dilation, defagg_groups=cor_group)
        
        ## the init tool ###
        #self.fc_tda = nn.Linear(self.width, self.width, bias=False)
        #nn.init.constant_(self.fc_tda.weight, 0)

        self.conv = nn.Conv2d(dim, dim, kernel_size=[1, 3], padding=[0, 1], groups=1, bias=False)

        #self.relu_tda = nn.ReLU(inplace=True)
        self.tda_norm = norm_layer(self.width)
        #self.tda_norm = nn.BatchNorm3d(self.width)
        nn.init.constant_(self.tda_norm.bias, 0)
        nn.init.constant_(self.tda_norm.weight, 0)

    def forward_rtc(self, x):
        NT, L, C = x.size()
        n_batch = NT // self.clip_len
        shortcut = x
        x = x.transpose(1,2) # nt x c x l
        x = x.view(n_batch, self.clip_len, C, self.input_resolution[0], self.input_resolution[1]).transpose(1,2) # NCTHW

        x_tmp = x.clone()
        x_tmp[:,:,1:,:,:] = x_tmp[:,:,:-1,:,:]
        x_tmp = (torch.sigmoid(x - x_tmp) * x) + x
        #offset = self.conv_offset(x_tmp)
        offset = nn.Parameter(torch.zeros(n_batch, self.off_channels_, self.clip_len, self.input_resolution[0], self.input_resolution[1])).cuda()
        corre_weight = self.def_cor(x, offset)
        x_agg = self.def_agg(x, offset, corre_weight)
        mask = torch.ones(x.size()).cuda()
        mask[:,:,-1,:,:] = 0
        mask.requires_grad = False
        x_shift = x_agg.clone()
        x_shift[:,:,:-1,:,:] = x_shift[:,:,1:,:,:]
        x = x_shift * mask
        
        x = x.transpose(1,2).reshape(n_batch*self.clip_len, C, -1).transpose(1,2)
        #x = self.fc_tda(x)
        x = self.tda_norm(x)
        x = shortcut + x

        return x

    def forward_vtc(self, x):
        n_segment = self.clip_len
        nt, l, c = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, l, c).transpose(1, 3)
        x = self.conv(x)
        x = x.transpose(1, 3).reshape(n_batch * n_segment, l, c)
        return x

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        #x = self.forward_rtc(x)
        #if self.shift_size == 0:
        #    x += self.forward_vtc(x)
        if self.shift_size == 0:
            x = self.forward_rtc(x)
        else:
            x += self.forward_vtc(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
                

class SIFABasicLayer(BasicLayer):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 K=3, cor_dilation=1, cor_group=1, clip_len=4):
        super().__init__(dim, input_resolution, depth, num_heads, window_size,
                         mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, 
                         norm_layer, downsample, use_checkpoint)
        # build blocks
        self.blocks = nn.ModuleList([
            SIFASwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                   num_heads=num_heads, window_size=window_size,
                                   shift_size=0 if (i % 2 == 0) else window_size // 2,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop, attn_drop=attn_drop,
                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                   norm_layer=norm_layer, K=K, cor_dilation=cor_dilation, 
                                   cor_group=cor_group, clip_len=clip_len)
            for i in range(depth)])


class C2D_SIFA_SWIN_ViT(C2D_SWIN_ViT):
    def __init__(self, blocks=[], img_size=224, early_stride=4, patch_size=4, in_chans=3, num_classes=1000, embed_dim=768, 
                 depths=[2, 2, 6, 2],num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, dropout_ratio=0., norm_layer=nn.LayerNorm, 
                 ape=False, patch_norm=True, use_checkpoint=False, K=3, cor_dilation=1, clip_len=4):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__(img_size, early_stride, patch_size, in_chans, num_classes, embed_dim, depths,
                         num_heads, window_size, mlp_ratio, qkv_bias, qk_scale,
                         drop_rate, attn_drop_rate, drop_path_rate, dropout_ratio, norm_layer, ape, patch_norm, use_checkpoint)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if blocks[i_layer] == SIFABasicLayer:
                layer = SIFABasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                       input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                          self.patches_resolution[1] // (2 ** i_layer)),
                                       depth=depths[i_layer],
                                       num_heads=num_heads[i_layer],
                                       window_size=window_size,
                                       mlp_ratio=self.mlp_ratio,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       drop=drop_rate, attn_drop=attn_drop_rate,
                                       drop_path=self.dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                       norm_layer=norm_layer,
                                       downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                       use_checkpoint=use_checkpoint,
                                       K=K, cor_dilation=cor_dilation, clip_len=clip_len)
            else:
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                        input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                          self.patches_resolution[1] // (2 ** i_layer)),
                                        depth=depths[i_layer],
                                        num_heads=num_heads[i_layer],
                                        window_size=window_size,
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=self.dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                        norm_layer=norm_layer,
                                        downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                        use_checkpoint=use_checkpoint)
            self.layers.append(layer)


def transfer_weights(state_dict, early_stride):
    new_state_dict = {}
    for k, v in state_dict.items():
        v = v.detach().numpy()
        if k == 'patch_embed.proj.weight':
            shape = v.shape
            v = np.reshape(v, newshape=[shape[0], shape[1], 1, shape[2], shape[3]])
            if early_stride != 1:
                s1 = early_stride // 2
                s2 = early_stride - early_stride // 2 - 1
                v = np.concatenate((np.zeros(shape=(shape[0], shape[1], s1, shape[2], shape[3])), v, 
                                    np.zeros(shape=(shape[0], shape[1], s2, shape[2], shape[3]))), axis=2)
        new_state_dict[k] = torch.from_numpy(v)
        if 'mlp.fc1.weight' in k:
            shape = v.shape
            kernel = np.zeros(shape=(shape[1], shape[1], 1, 3))
            new_key = k.replace('mlp.fc1.weight', 'conv.weight')
            new_state_dict[new_key] = torch.from_numpy(kernel)
        new_state_dict[k] = torch.from_numpy(v)
    return new_state_dict    


@register_model
def c2d_sifa_swin_vit_s_p4_w7(pooling_arch=None, num_classes=1000, dropout_ratio=0., image_size=224, early_stride=4, clip_length=16, sifa_kernel=None):
    """ SWIN-ViT model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    """
    K = 3
    if sifa_kernel: K = sifa_kernel[0]
    blocks = [SIFABasicLayer, SIFABasicLayer, SIFABasicLayer, SIFABasicLayer]
    model = C2D_SIFA_SWIN_ViT(img_size=image_size, early_stride=early_stride, 
                             drop_path_rate=0.3, window_size=7,
                             patch_size=4, in_chans=3, num_classes=num_classes, 
                             embed_dim=96, depths=[2, 2, 18, 2],
                             num_heads=[3, 6, 12, 24], dropout_ratio=dropout_ratio,
                             clip_len=(clip_length//early_stride),
                             K=K, cor_dilation=1, blocks=blocks)
    return model


@register_model
def c2d_sifa_swin_vit_b_p4_w7(pooling_arch=None, num_classes=1000, dropout_ratio=0., image_size=224, early_stride=4, clip_length=16, sifa_kernel=None):
    """ SWIN-ViT model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    """
    K = 3
    if sifa_kernel: K = sifa_kernel[0]
    blocks = [SIFABasicLayer, SIFABasicLayer, SIFABasicLayer, SIFABasicLayer]
    model = C2D_SIFA_SWIN_ViT(img_size=image_size, early_stride=early_stride, 
                             drop_path_rate=0.5, window_size=7,
                             patch_size=4, in_chans=3, num_classes=num_classes, 
                             embed_dim=128, depths=[2, 2, 18, 2],
                             num_heads=[4, 8, 16, 32], dropout_ratio=dropout_ratio,
                             clip_len=(clip_length//early_stride),
                             K=K, cor_dilation=1, blocks=blocks)
    return model


@register_model
def c2d_sifa_swin_vit_l_p4_w7(pooling_arch=None, num_classes=1000, dropout_ratio=0., image_size=224, early_stride=4, clip_length=16, sifa_kernel=None):
    """ SWIN-ViT model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    """
    K = 3
    if sifa_kernel: K = sifa_kernel[0]
    blocks = [SIFABasicLayer, SIFABasicLayer, SIFABasicLayer, SIFABasicLayer]
    model = C2D_SIFA_SWIN_ViT(img_size=image_size, early_stride=early_stride, 
                             drop_path_rate=0.5, window_size=7,
                             patch_size=4, in_chans=3, num_classes=num_classes, 
                             embed_dim=192, depths=[2, 2, 18, 2],
                             num_heads=[6, 12, 24, 48], dropout_ratio=dropout_ratio,
                             clip_len=(clip_length//early_stride),
                             K=K, cor_dilation=1, blocks=blocks)
    return model
