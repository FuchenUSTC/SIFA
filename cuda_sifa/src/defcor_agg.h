#pragma once

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

at::Tensor
defcor_forward(const at::Tensor &input,
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
               const int defcor_group)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return defcor_cuda_forward(input, weight, offset,
                                   kernel_h, kernel_w,
                                   stride_h, stride_w,
                                   pad_h, pad_w,
                                   dilation_h, dilation_w,
                                   defcor_group);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else{
        //return defcor_cpu_forward(input, weight, offset,
        //                           kernel_h, kernel_w,
        //                           stride_h, stride_w,
        //                           pad_h, pad_w,
        //                           dilation_h, dilation_w,
        //                           defcor_group);
    }
}

std::vector<at::Tensor>
defcor_backward(const at::Tensor &input,
                const at::Tensor &weight,
                const at::Tensor &offset,
                const at::Tensor &grad_output,
                int kernel_h, int kernel_w,
                int stride_h, int stride_w,
                int pad_h, int pad_w,
                int dilation_h, int dilation_w,
                int defcor_group)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return defcor_cuda_backward(input,
                                    weight,
                                    offset,
                                    grad_output,
                                    kernel_h, kernel_w,
                                    stride_h, stride_w,
                                    pad_h, pad_w,
                                    dilation_h, dilation_w,
                                    defcor_group);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else{
        //return defcor_cpu_backward(input,
        //                            weight,
        //                            offset,
        //                            mask,
        //                            grad_output,
        //                            kernel_h, kernel_w,
        //                            stride_h, stride_w,
        //                            pad_h, pad_w,
        //                            dilation_h, dilation_w,
        //                            defcor_group);
    }
}

at::Tensor
defagg_forward(const at::Tensor &input,
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
               const int defagg_group)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return defagg_cuda_forward(input, weight, offset,
                                   kernel_h, kernel_w,
                                   stride_h, stride_w,
                                   pad_h, pad_w,
                                   dilation_h, dilation_w,
                                   defagg_group);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else{
        //return defagg_cpu_forward(input, weight, offset,
        //                           kernel_h, kernel_w,
        //                           stride_h, stride_w,
        //                           pad_h, pad_w,
        //                           dilation_h, dilation_w,
        //                           defagg_group);
    }
}

std::vector<at::Tensor>
defagg_backward(const at::Tensor &input,
                const at::Tensor &weight,
                const at::Tensor &offset,
                const at::Tensor &grad_output,
                int kernel_h, int kernel_w,
                int stride_h, int stride_w,
                int pad_h, int pad_w,
                int dilation_h, int dilation_w,
                int defagg_group)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return defagg_cuda_backward(input,
                                    weight,
                                    offset,
                                    grad_output,
                                    kernel_h, kernel_w,
                                    stride_h, stride_w,
                                    pad_h, pad_w,
                                    dilation_h, dilation_w,
                                    defagg_group);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else{
        //return defagg_cpu_backward(input,
        //                            weight,
        //                            offset,
        //                            mask,
        //                            grad_output,
        //                            kernel_h, kernel_w,
        //                            stride_h, stride_w,
        //                            pad_h, pad_w,
        //                            dilation_h, dilation_w,
        //                            defagg_group);
    }
}