#pragma once
#include <torch/extension.h>
#include <ATen/div_rtn.h>

at::Tensor
defagg_cuda_forward(const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::Tensor &offset,
                    const int kernel_h,
                    const int kernel_w,
                    const int stride_h,
                    const int stride_w,
                    const int pad_h,
                    const int pad_w,
                    const int dilation_h,
                    const int dilation_w,
                    const int defagg_group);

std::vector<at::Tensor>
defagg_cuda_backward(const at::Tensor &input,
                     const at::Tensor &weight,
                     const at::Tensor &offset,
                     const at::Tensor &grad_output,
                     int kernel_h, int kernel_w,
                     int stride_h, int stride_w,
                     int pad_h, int pad_w,
                     int dilation_h, int dilation_w,
                     int defagg_group);

at::Tensor
defcor_cuda_forward(const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::Tensor &offset,
                    const int kernel_h,
                    const int kernel_w,
                    const int stride_h,
                    const int stride_w,
                    const int pad_h,
                    const int pad_w,
                    const int dilation_h,
                    const int dilation_w,
                    const int defcor_group);

std::vector<at::Tensor>
defcor_cuda_backward(const at::Tensor &input,
                     const at::Tensor &weight,
                     const at::Tensor &offset,
                     const at::Tensor &grad_output,
                     int kernel_h, int kernel_w,
                     int stride_h, int stride_w,
                     int pad_h, int pad_w,
                     int dilation_h, int dilation_w,
                     int defcor_group);