// auth: Fuchen Long
// date: 2021/7/18

#include <vector>
#include "defcor_agg_im2col_cuda.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

//THCState *state = at::globalContext().lazyInitCUDA();

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
                    const int defcor_group)
{
    using scalar_t = float;
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, weight, bias, offset, mask));
    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(offset.is_cuda(), "offset must be a CUDA tensor");
    
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int times = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);

    const int channels_ = weight.size(0);
    const int w_times = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    AT_ASSERTM(kernel_h_ == kernel_h && kernel_w_ == kernel_w,
               "Input shape and kernel shape wont match: (%d x %d vs %d x %d).", kernel_h_, kernel_w, kernel_h_, kernel_w_);

    AT_ASSERTM(channels == channels_,
               "Input shape and kernel channels wont match: (%d vs %d).", channels, channels_);

    AT_ASSERTM(times == w_times,
                "Input shape and kernel times wont match: (%d vs %d).", times, w_times);

    //const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    //const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    const int time_out = times;
    const int height_out = height;
    const int width_out = width;
    const int channel_out = kernel_h * kernel_w * defcor_group;


    auto output = at::zeros({batch, channel_out, time_out, height_out, width_out}, input.options());


    // compute the deformable correlation
    defcor_im2col_forward_cuda(c10::cuda::getCurrentCUDAStream(),
                               input.data_ptr<scalar_t>(),
                               offset.data_ptr<scalar_t>(),
                               weight.data_ptr<scalar_t>(),
                               batch, channels, times, height, width,
                               height_out, width_out, channel_out, kernel_h, kernel_w,
                               pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                               defcor_group, output.data_ptr<scalar_t>());

    return output;
}

std::vector<at::Tensor> defcor_cuda_backward(const at::Tensor &input,
                                             const at::Tensor &weight,
                                             const at::Tensor &offset,
                                             const at::Tensor &grad_output,
                                             int kernel_h, int kernel_w,
                                             int stride_h, int stride_w,
                                             int pad_h, int pad_w,
                                             int dilation_h, int dilation_w,
                                             int defcor_group)
{


    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(offset.is_cuda(), "offset must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int times = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);

    const int channels_ = weight.size(0);
    const int w_times = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    AT_ASSERTM(kernel_h_ == kernel_h && kernel_w_ == kernel_w,
               "Input shape and kernel shape wont match: (%d x %d vs %d x %d).", kernel_h_, kernel_w, kernel_h_, kernel_w_);

    AT_ASSERTM(channels == channels_,
               "Input shape and kernel channels wont match: (%d vs %d).", channels, channels_);
    
    AT_ASSERTM(times == w_times,
     "Input shape and kernel times wont match: (%d vs %d).", times, w_times);    

    //const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    //const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int height_out = height;
    const int width_out = width;
    const int channel_out = kernel_h * kernel_w * defcor_group;


    auto grad_input = at::zeros_like(input);
    auto grad_weight = at::zeros_like(weight);
    auto grad_offset = at::zeros_like(offset);

    using scalar_t = float;

    for (int b = 0; b < batch; b++)
    {
        auto input_n = input.select(0, b);
        auto offset_n = offset.select(0, b);
        auto grad_output_n = grad_output.select(0, b);
        auto grad_input_n = grad_input.select(0, b);
        auto grad_offset_n = grad_offset.select(0, b);

        // gradient w.r.t. input coordinate data
        defcor_col2im_coord_cuda(c10::cuda::getCurrentCUDAStream(),
                                 grad_output_n.data_ptr<scalar_t>(),
                                 input_n.data_ptr<scalar_t>(),
                                 offset_n.data_ptr<scalar_t>(),
                                 weight.data_ptr<scalar_t>(),
                                 1, channels, times, height, width,
                                 height_out, width_out, channel_out, kernel_h, kernel_w,
                                 pad_h, pad_w, stride_h, stride_w,
                                 dilation_h, dilation_w, defcor_group,
                                 grad_offset_n.data_ptr<scalar_t>());

        // gradient w.r.t. input data
        defcor_col2im_cuda(c10::cuda::getCurrentCUDAStream(),
                           grad_output_n.data_ptr<scalar_t>(),
                           input_n.data_ptr<scalar_t>(),
                           offset_n.data_ptr<scalar_t>(),
                           weight.data_ptr<scalar_t>(),
                           1, channels, times, height, width,
                           height_out, width_out, channel_out, kernel_h, kernel_w,
                           pad_h, pad_w, stride_h, stride_w,
                           dilation_h, dilation_w, defcor_group,
                           grad_input_n.data_ptr<scalar_t>());

        // gradient w.r.t. weight, dWeight should accumulate across the batch and group
        defcor_col2im_weight_cuda(c10::cuda::getCurrentCUDAStream(),
                                  grad_output_n.data_ptr<scalar_t>(),
                                  input_n.data_ptr<scalar_t>(),
                                  offset_n.data_ptr<scalar_t>(),
                                  weight.data_ptr<scalar_t>(),
                                  1, channels, times, height, width,
                                  height_out, width_out, channel_out, kernel_h, kernel_w,
                                  pad_h, pad_w, stride_h, stride_w,
                                  dilation_h, dilation_w, defcor_group,
                                  grad_weight.data_ptr<scalar_t>());

    }

    return {
        grad_input, grad_offset, grad_weight
    };
}

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
                    const int defagg_group)
{
    using scalar_t = float;
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, weight, bias, offset, mask));
    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(offset.is_cuda(), "offset must be a CUDA tensor");
    
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int times = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);

    const int batch_w = weight.size(0);
    const int channels_ = weight.size(1);
    const int w_times = weight.size(2);
    const int w_height_ = weight.size(3);
    const int w_width_ = weight.size(4);

    const int weight_c = defagg_group*kernel_h*kernel_w;

    AT_ASSERTM(height == w_height_ && width == w_width_,
               "Input shape and weight shape wont match: (%d x %d vs %d x %d).", height, width, w_height_, w_width_);

    AT_ASSERTM(channels_ == weight_c,
               "Kernel shape and weight channels wont match: (%d vs %d).", channels_, weight_c);

    AT_ASSERTM(times == w_times,
                "Input shape and kernel times wont match: (%d vs %d).", times, w_times);

    //const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    //const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    const int time_out = times;
    const int height_out = height;
    const int width_out = width;
    const int channel_out = channels * defagg_group;


    auto output = at::zeros({batch, channel_out, time_out, height_out, width_out}, input.options());


    // compute the deformable correlation
    defagg_im2col_forward_cuda(c10::cuda::getCurrentCUDAStream(),
                               input.data_ptr<scalar_t>(),
                               offset.data_ptr<scalar_t>(),
                               weight.data_ptr<scalar_t>(),
                               batch, channels, times, height, width,
                               height_out, width_out, channel_out, kernel_h, kernel_w,
                               pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                               defagg_group, output.data_ptr<scalar_t>());

    return output;
}

std::vector<at::Tensor> defagg_cuda_backward(const at::Tensor &input,
                                             const at::Tensor &weight,
                                             const at::Tensor &offset,
                                             const at::Tensor &grad_output,
                                             int kernel_h, int kernel_w,
                                             int stride_h, int stride_w,
                                             int pad_h, int pad_w,
                                             int dilation_h, int dilation_w,
                                             int defagg_group)
{


    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(offset.is_cuda(), "offset must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int times = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);

    const int batch_w = weight.size(0);
    const int channels_ = weight.size(1);
    const int w_times = weight.size(2);
    const int w_height_ = weight.size(3);
    const int w_width_ = weight.size(4);

    const int weight_c = defagg_group * kernel_h * kernel_w;

    AT_ASSERTM(height == w_height_ && width == w_width_,
               "Input shape and weight shape wont match: (%d x %d vs %d x %d).", height, width, w_height_, w_width_);

    AT_ASSERTM(channels_ == weight_c,
               "Kernel shape and weight channels wont match: (%d vs %d).", channels_, weight_c);
    
    AT_ASSERTM(times == w_times,
     "Input shape and kernel times wont match: (%d vs %d).", times, w_times);    

    //const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    //const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int height_out = height;
    const int width_out = width;
    const int channel_out = channels * defagg_group;


    auto grad_input = at::zeros_like(input);
    auto grad_weight = at::zeros_like(weight);
    auto grad_offset = at::zeros_like(offset);

    using scalar_t = float;

    for (int b = 0; b < batch; b++)
    {
        auto input_n = input.select(0, b);
        auto offset_n = offset.select(0, b);
        auto weight_n = weight.select(0, b);
        auto grad_output_n = grad_output.select(0, b);
        auto grad_input_n = grad_input.select(0, b);
        auto grad_offset_n = grad_offset.select(0, b);
        auto grad_weight_n = grad_weight.select(0, b);

        // gradient w.r.t. input coordinate data
        defagg_col2im_coord_cuda(c10::cuda::getCurrentCUDAStream(),
                                 grad_output_n.data_ptr<scalar_t>(),
                                 input_n.data_ptr<scalar_t>(),
                                 offset_n.data_ptr<scalar_t>(),
                                 weight_n.data_ptr<scalar_t>(),
                                 1, channels, times, height, width,
                                 height_out, width_out, channel_out, kernel_h, kernel_w,
                                 pad_h, pad_w, stride_h, stride_w,
                                 dilation_h, dilation_w, defagg_group,
                                 grad_offset_n.data_ptr<scalar_t>());

        // gradient w.r.t. input data
        defagg_col2im_cuda(c10::cuda::getCurrentCUDAStream(),
                           grad_output_n.data_ptr<scalar_t>(),
                           input_n.data_ptr<scalar_t>(),
                           offset_n.data_ptr<scalar_t>(),
                           weight_n.data_ptr<scalar_t>(),
                           1, channels, times, height, width,
                           height_out, width_out, channel_out, kernel_h, kernel_w,
                           pad_h, pad_w, stride_h, stride_w,
                           dilation_h, dilation_w, defagg_group,
                           grad_input_n.data_ptr<scalar_t>());

        // gradient w.r.t. weight, dWeight should accumulate across the batch and group
        defagg_col2im_weight_cuda(c10::cuda::getCurrentCUDAStream(),
                                  grad_output_n.data_ptr<scalar_t>(),
                                  input_n.data_ptr<scalar_t>(),
                                  offset_n.data_ptr<scalar_t>(),
                                  weight_n.data_ptr<scalar_t>(),
                                  1, channels, times, height, width,
                                  height_out, width_out, channel_out, kernel_h, kernel_w,
                                  pad_h, pad_w, stride_h, stride_w,
                                  dilation_h, dilation_w, defagg_group,
                                  grad_weight_n.data_ptr<scalar_t>());

    }

    return {
        grad_input, grad_offset, grad_weight
    };
}