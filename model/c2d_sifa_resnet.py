import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from functools import partial
from model.defcor import DefAgg, DefCorFixW
from .model_factory import register_model


__all__ = ['C2D_SIFA_ResNet','c2d_sifa_resnet50', 'c2d_sifa_resnet101']


def conv1x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """1x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=[1, 3, 3], stride=[1, stride, stride],
                     padding=[0, dilation, dilation], groups=groups, bias=False, dilation=[1, dilation, dilation])


def conv3x1x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x1x1 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=[3, 1, 1], stride=[stride, 1, 1],
                     padding=[dilation, 0, 0], groups=groups, bias=False, dilation=[dilation, 1, 1])


def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=[1, stride, stride], bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        self.width = width
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv1x3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class GBottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(GBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        self.width = width
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv1x3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # global context
        self.fc1 = nn.Linear(planes * self.expansion, planes * self.expansion // 16)
        self.fc1_relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(planes * self.expansion // 16, planes * self.expansion)        

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        #out += identity
        #out = self.relu(out)

        gc = self.fc2(self.fc1_relu(self.fc1(torch.flatten(self.avgpool(out.clone()), 1))))
        out += identity + gc.view(gc.size(0), gc.size(1), 1, 1, 1)
        out = self.relu(out)     
        
        return out


# --- temporal deformable correlation ---- #

class DEFLocalTransBottleneck(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, K=3, cor_dilation=1, 
                 cor_group=1, global_context=False, clip_len=4):    
        super().__init__(inplanes, planes, stride, downsample, groups,
                         base_width, dilation, norm_layer)
        self.group_width = K * K * cor_group
        
        pad_num = (cor_dilation * (K - 1) + 1) // 2
        self.off_channels_ = 2 * K * K
        self.kernel_size = _pair(K)
        self.conv_offset = nn.Conv3d(self.width, self.off_channels_, 
                                     kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.conv_offset.weight.data.zero_()
        self.clip_len = clip_len
        
        self.def_cor = DefCorFixW(in_channels=self.width, times=clip_len, kernel_size=(K,K), stride=1, padding=pad_num, 
                              dilation=cor_dilation, defcor_groups=cor_group)
        self.def_agg = DefAgg(in_channels=self.width, times=clip_len, kernel_size=(K,K), stride=1, padding=pad_num, 
                              dilation=cor_dilation, defagg_groups=cor_group)

        
        self.global_context = global_context
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # insert the aggregate block
        out_tmp = out.clone()
        out_tmp[:,:,1:,:,:] = out_tmp[:,:,:-1,:,:]
        out_tmp = (torch.sigmoid(out - out_tmp) * out) + out # [sig(f_{t}-f_{t-1})*f_{t}]*f_{t}
        offset = self.conv_offset(out_tmp)        
        
        corre_weight = self.def_cor(out, offset)
        out_agg = self.def_agg(out, offset, corre_weight)
        
        mask = torch.ones(out.size()).cuda()
        mask[:,:,-1,:,:] = 0
        mask.requires_grad = False
        out_shift = out_agg.clone()
        out_shift[:,:,:-1,:,:] = out_shift[:,:,1:,:,:]
        out = out + out_shift * mask
        # conv3 
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ---- The implemenration of whole network ---- #

class C2D_SIFA_ResNet(nn.Module):
    def __init__(self, block, layers, pooling_arch, num_classes=400, dropout_ratio=0.5, zero_init_residual=True,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, deep_stem=False, block_partial=2, sifa_kernel=[1, 3, 3, 3], 
                 dilation_size=[1, 1, 1, 1], cor_group_num=[1, 1, 1, 1], clip_length=16):
        super(C2D_SIFA_ResNet, self).__init__()       
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d if not deep_stem else partial(nn.BatchNorm3d, eps=2e-5)
        self._norm_layer = norm_layer
        self._deep_stem = deep_stem

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if not deep_stem:
            self.conv1 = nn.Conv3d(3, self.inplanes, kernel_size=[4, 7, 7], stride=[4, 2, 2], padding=[0, 3, 3],
                                    bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.conv1 = nn.Conv3d(3, self.inplanes, kernel_size=[4, 3, 3], stride=[4, 2, 2], padding=[0, 1, 1],
                                    bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv3d(self.inplanes, self.inplanes, kernel_size=[1, 3, 3], stride=[1, 1, 1], padding=[0, 1, 1],
                                    bias=False)
            self.bn2 = norm_layer(self.inplanes)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv3d(self.inplanes, self.inplanes * 2, kernel_size=[1, 3, 3], stride=[1, 1, 1], padding=[0, 1, 1],
                                    bias=False)
            self.bn3 = norm_layer(self.inplanes * 2)
            self.relu3 = nn.ReLU(inplace=True)
            
            self.inplanes *= 2
        self.maxpool = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])
        
        
        self.layer1 = self._return_layers(block[0], 64, layers[0], 
                                          dilate=replace_stride_with_dilation[0], K_size=sifa_kernel[0],
                                          dilation_size=dilation_size[0], group_num=cor_group_num[0], 
                                          clip_size=clip_length//4)
        self.layer2 = self._return_layers(block[1], 128, layers[1], stride=2,
                                          dilate=replace_stride_with_dilation[0], K_size=sifa_kernel[1], 
                                          dilation_size=dilation_size[1], group_num=cor_group_num[1], 
                                          clip_size=clip_length//4)
        self.layer3 = self._return_layers(block[2], 256, layers[2], stride=2,
                                          dilate=replace_stride_with_dilation[1], K_size=sifa_kernel[2], 
                                          dilation_size=dilation_size[2], group_num=cor_group_num[2], 
                                          clip_size=clip_length//4)  
        self.layer4 = self._return_layers(block[3], 512, layers[3], stride=2,
                                          dilate=replace_stride_with_dilation[2], K_size=sifa_kernel[3], 
                                          dilation_size=dilation_size[3], group_num=cor_group_num[3], 
                                          clip_size=clip_length//4)
        self.inplanes = 256 * block[0].expansion
        self.maxpool_dual = nn.MaxPool3d(kernel_size=[2, 1, 1], stride=[2, 1, 1], padding=[0, 0, 0])
        self.pool = pooling_arch(input_dim=512 * block[0].expansion)
        self.pool_dual = pooling_arch(input_dim=512 * block[0].expansion)
        # the other branch is the dual global bottleneck
        self.layer4_dual = self._return_layers(block[3], 512, layers[3], stride=1, 
                                               dilate=replace_stride_with_dilation[2], K_size=sifa_kernel[3],
                                               dilation_size=dilation_size[3], group_num=cor_group_num[3],
                                               clip_size=clip_length//8)

        self.drop = nn.Dropout(dropout_ratio)
        self.drop_dual = nn.Dropout(dropout_ratio)

        self.fc = nn.Linear(self.pool.output_dim, num_classes)
        self.fc_dual = nn.Linear(self.pool_dual.output_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DEFLocalTransBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, GBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  
                elif isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)                  
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, K_size=3, 
                    dilation_size=1, group_num=1, clip_size=4):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block[0].expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block[0].expansion, stride),
                norm_layer(planes * block[0].expansion),
            )
        additional_block = [DEFLocalTransBottleneck]
        layers = []
        if block[0] in additional_block:
            layers.append(block[0](self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer, 
                                K=K_size, cor_dilation=dilation_size, cor_group=group_num, clip_len=clip_size))
        else:
            layers.append(block[0](self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
        
        self.inplanes = planes * block[0].expansion
        for i in range(1, blocks):
            if block[i] in additional_block:
                layers.append(block[i](self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, 
                                    K=K_size, cor_dilation=dilation_size, cor_group=group_num, clip_len=clip_size))
            else:
                layers.append(block[i](self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _return_layers(self, block, planes, blocks, stride=1, dilate=False, K_size=3, 
                       dilation_size=1, group_num=1, clip_size=4):
        if block == Bottleneck:
            layers = self._make_layer([block for i in range(blocks)], planes, blocks, stride=stride, dilate=dilate,
                                      K_size=K_size, dilation_size=dilation_size, group_num=group_num, clip_size=clip_size)
            return layers
        else:
            block_insert = (GBottleneck, block)
            layers = self._make_layer([block_insert[i % 2] for i in range(blocks)], planes, blocks, stride=stride, dilate=dilate,
                                      K_size=K_size, dilation_size=dilation_size, group_num=group_num, clip_size=clip_size)
            return layers 

    def forward(self, x):
        if not self._deep_stem:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x_origin = self.layer4(x)
        x_dual = self.maxpool_dual(x)
        x_dual = self.layer4_dual(x_dual)
        x_g_origin = self.pool(x_origin)
        x_g_dual = self.pool_dual(x_dual)
        x_g_origin = self.drop(x_g_origin)
        x_g_dual = self.drop_dual(x_g_dual)
        x1_origin = self.fc(x_g_origin)
        x1_dual = self.fc_dual(x_g_dual)

        return [x1_origin, x1_dual]


def transfer_weights(state_dict, early_stride):
    new_state_dict = {}
    for k, v in state_dict.items():
        v = v.detach().numpy()
        if ('conv' in k) or ('downsample.0' in k):
            shape = v.shape
            v = np.reshape(v, newshape=[shape[0], shape[1], 1, shape[2], shape[3]])
            if (not ('layer' in k)) and ('conv1' in k):  # first conv7x7 layer
                if early_stride != 1:
                    s1 = early_stride // 2
                    s2 = early_stride - early_stride // 2 - 1
                    v = np.concatenate((np.zeros(shape=(shape[0], shape[1], s1, shape[2], shape[3])), v, np.zeros(shape=(shape[0], shape[1], s2, shape[2], shape[3]))), axis=2)
        new_state_dict[k] = torch.from_numpy(v)
        new_state_dict[k.replace('layer4', 'layer4_dual')] = torch.from_numpy(v)
    return new_state_dict


@register_model
def _c2d_sifa_resnet(arch, block, layers, pooling_arch, **kwargs):
    model = C2D_SIFA_ResNet(block, layers, pooling_arch, **kwargs)
    return model


@register_model
def c2d_sifa_resnet50(pooling_arch, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pooling_arch: choose the pooling architecture for last conv/res layer
    """
    block_array = [Bottleneck, DEFLocalTransBottleneck, DEFLocalTransBottleneck, DEFLocalTransBottleneck]
    return _c2d_sifa_resnet('resnet50', block_array, [3, 4, 6, 3], pooling_arch,
                   **kwargs)


@register_model
def c2d_sifa_resnet101(pooling_arch, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pooling_arch: choose the pooling architecture for last conv/res layer
    """
    block_array = [Bottleneck, DEFLocalTransBottleneck, DEFLocalTransBottleneck, DEFLocalTransBottleneck]
    return _c2d_sifa_resnet('resnet101', block_array, [3, 4, 23, 3], pooling_arch,
                   **kwargs)

