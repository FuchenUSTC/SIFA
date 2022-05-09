#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import math
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

import _ext as _backend


### Deformabel correlation
class _DefCor(Function):
    @staticmethod
    def forward(
        ctx, input, offset, weight, stride, padding, dilation, defcor_groups
    ):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.defcor_groups = defcor_groups
        output = _backend.defcor_forward(
            input,
            weight,
            offset,
            ctx.kernel_size[0],
            ctx.kernel_size[1],
            ctx.stride[0],
            ctx.stride[1],
            ctx.padding[0],
            ctx.padding[1],
            ctx.dilation[0],
            ctx.dilation[1],
            ctx.defcor_groups,
        )
        ctx.save_for_backward(input, offset, weight)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors
        grad_input, grad_offset, grad_weight = _backend.defcor_backward(
            input,
            weight,
            offset,
            grad_output,
            ctx.kernel_size[0],
            ctx.kernel_size[1],
            ctx.stride[0],
            ctx.stride[1],
            ctx.padding[0],
            ctx.padding[1],
            ctx.dilation[0],
            ctx.dilation[1],
            ctx.defcor_groups,
        )

        return grad_input, grad_offset, grad_weight, None, None, None, None

    @staticmethod
    def symbolic(
        g, input, offset, weight, stride, padding, dilation, defcor_groups
    ):
        from torch.nn.modules.utils import _pair

        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        # as of trt 7, the dcn operation will be translated again by modifying the onnx file
        # so the exporting code is kept to resemble the forward()
        return g.op(
            "DefCor",
            input,
            offset,
            weight,
            stride_i=stride,
            padding_i=padding,
            dilation_i=dilation,
            defcor_groups_i=defcor_groups,
        )


def_cor = _DefCor.apply


class DefCorFixW(nn.Module):
    def __init__(
        self,
        in_channels,
        times,
        kernel_size,
        stride,
        padding,
        dilation=1,
        defcor_groups=1,
    ):
        super(DefCorFixW, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.defcor_groups = defcor_groups

        self.weight = nn.Parameter(torch.Tensor(in_channels, times, *self.kernel_size))
        self.reset_parameters()
        self.weight.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        #for k in self.kernel_size:
        #    n *= k
        stdv = 1.0 / n
        nn.init.constant_(self.weight.data, stdv)

    def forward(self, input, offset):
        assert (
            2 * self.kernel_size[0] * self.kernel_size[1]
            == offset.shape[1]
        )
        return def_cor(
            input,
            offset,
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.defcor_groups,
        )


### Deformabel aggregation 
class _DefAgg(Function):
    @staticmethod
    def forward(
        ctx, input, offset, weight, kernel_size, stride, padding, dilation, defagg_groups
    ):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(kernel_size)
        ctx.defagg_groups = defagg_groups
        output = _backend.defagg_forward(
            input,
            weight,
            offset,
            ctx.kernel_size[0],
            ctx.kernel_size[1],
            ctx.stride[0],
            ctx.stride[1],
            ctx.padding[0],
            ctx.padding[1],
            ctx.dilation[0],
            ctx.dilation[1],
            ctx.defagg_groups,
        )
        ctx.save_for_backward(input, offset, weight)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors
        grad_input, grad_offset, grad_weight = _backend.defagg_backward(
            input,
            weight,
            offset,
            grad_output,
            ctx.kernel_size[0],
            ctx.kernel_size[1],
            ctx.stride[0],
            ctx.stride[1],
            ctx.padding[0],
            ctx.padding[1],
            ctx.dilation[0],
            ctx.dilation[1],
            ctx.defagg_groups,
        )

        return grad_input, grad_offset, grad_weight, None, None, None, None, None

    @staticmethod
    def symbolic(
        g, input, offset, weight, stride, padding, dilation, defagg_groups
    ):
        from torch.nn.modules.utils import _pair

        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        # as of trt 7, the dcn operation will be translated again by modifying the onnx file
        # so the exporting code is kept to resemble the forward()
        return g.op(
            "DefAgg",
            input,
            offset,
            weight,
            stride_i=stride,
            padding_i=padding,
            dilation_i=dilation,
            defagg_groups_i=defagg_groups,
        )


def_agg = _DefAgg.apply


class DefAgg(nn.Module):
    def __init__(
        self,
        in_channels,
        times,
        kernel_size,
        stride,
        padding,
        dilation=1,
        defagg_groups=1,
    ):
        super(DefAgg, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.defagg_groups = defagg_groups

    def forward(self, input, offset, weight):
        assert (
            2 * self.kernel_size[0] * self.kernel_size[1]
            == offset.shape[1]
        )
        return def_agg(
            input,
            offset,
            weight,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.defagg_groups,
        )

