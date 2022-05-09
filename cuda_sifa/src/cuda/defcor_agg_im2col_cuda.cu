// auth: Fuchen Long
// date: 2021/7/18

#include "defcor_agg_im2col_cuda.h"
#include <cstdio>
#include <algorithm>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


__device__ float im2col_bilinear_cuda(const float *bottom_data, const int data_width,
                                      const int height, const int width, float h, float w)
{
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

__device__ float get_gradient_weight_cuda(float argmax_h, float argmax_w,
                                          const int h, const int w, const int height, const int width)
{
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  float weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

__device__ float get_coordinate_weight_cuda(float argmax_h, float argmax_w,
                                            const int height, const int width, const float *im_data,
                                            const int data_width, const int bp_dir)
{
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  float weight = 0;

  if (bp_dir == 0)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }
  else if (bp_dir == 1)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}



__global__ void defcor_im2col_gpu_kernel(const int n,
                                         const float *data_im, const float *data_offset, const float *data_weight,
                                         const int batch_size, const int channels, const int times, const int height, const int width,
                                         const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kernel_w,
                                         const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                         const int dilation_h, const int dilation_w, const int defcor_group,
                                         float *data_out)
{
  // launch N * C * T * H * W cores
  CUDA_KERNEL_LOOP(index, n)
  {

    // input index: N x C_col x T x H x W 
    const int w_out = index % width_col;
    const int h_out = (index / width_col) % height_col;
    const int t_out = (index / width_col / height_col) % times;
    const int c_out = (index / width_col / height_col / times) % channel_col; 
    const int n_out = (index / width_col / height_col / times / channel_col);

    const int group_id = c_out / (kernel_w * kernel_h);
    const int c_in_per_defcor =  channels / defcor_group;

    int t_former = t_out - 1;
    if (t_out <= 0){
      t_former = 0;
    }
    const float *data_offset_ptr = data_offset + n_out * 2 * kernel_h * kernel_w * times * height_col * width_col;

    int kw = (c_out % kernel_w);
    int kh = (c_out / kernel_w) % kernel_h;
    

    // the next frame position
    const int h_in = h_out * stride_h - pad_h + kh * dilation_h;
    const int w_in = w_out * stride_w - pad_w + kw * dilation_w;

    // the offset of the next frame
    const int data_offset_h_ptr = (((2 * (kh * kernel_w + kw)) * times + t_out) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = (((2 * (kh * kernel_w + kw) + 1) * times + t_out) * height_col + h_out) * width_col + w_out;
    const float offset_h = data_offset_ptr[data_offset_h_ptr];
    const float offset_w = data_offset_ptr[data_offset_w_ptr];    
    const float h_im = h_in + offset_h;
    const float w_im = w_in + offset_w;
    float val = static_cast<float>(0);
    const int c_in_start = group_id * c_in_per_defcor;
    const int c_in_end = (group_id + 1) * c_in_per_defcor;
    if (h_im > -1 && w_im > -1 && h_im < height && w_im < width){
      for (int c_in = c_in_start; c_in < c_in_end; ++c_in){
        const int former_ptr = (((n_out * channels + c_in) * times + t_former) * height + h_out) * width + w_out;
        const int weight_offset = ((c_in * times + t_out) * kernel_h + kh) * kernel_w + kw;
        const float* data_im_ptr = data_im + ((n_out * channels + c_in) * times + t_out) * height * width;
        float next_val = im2col_bilinear_cuda(data_im_ptr, width, height, width, h_im, w_im);
        val += (data_weight[weight_offset] * data_im[former_ptr] * next_val);
      }
    }
    data_out[index] = val;
  }
}

__global__ void defcor_col2im_gpu_kernel(const int n, const float *top_diff,
                                         const float *data_im, const float *data_offset, const float *data_weight,
                                         const int batch_size, const int channels, const int times, const int height, const int width,
                                         const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kernel_w,
                                         const int pad_h, const int pad_w,
                                         const int stride_h, const int stride_w,
                                         const int dilation_h, const int dilation_w, const int channel_coord, const int defcor_group,
                                         float *grad_im)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    const int c_in_per_defcor = channels / defcor_group;

    // The index: channels x times x kernel_h x kernel_w x batch_size x height_col x width_col 
    const int j = (index / width_col / height_col / batch_size ) % kernel_w; // the w of kernel position 
    const int i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h; // the h of kernel position
    const int c = (index / width_col / height_col / batch_size / kernel_w / kernel_h / times); // the channels of input data 
    const int group_id = c / c_in_per_defcor;
    const int c_out = (group_id * kernel_h + i) * kernel_w + j;

  
    int w_out = (index % width_col); // former
    int h_out = (index / width_col) % height_col; // former
    int t_out = (index / width_col / height_col / batch_size / kernel_w / kernel_h) % times;
    int n_out = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w; // current
    int h_in = h_out * stride_h - pad_h; // current
    int index_out =  (((n_out * channel_col + c_out) * times + t_out) * height_col + h_out) * width_col + w_out;
    
    int t_former = t_out - 1;
    if (t_out <= 0){
      t_former = 0;
    }    

    const float *data_offset_ptr = data_offset + n_out * 2 * kernel_h * kernel_w * times * height_col * width_col;
    const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * times + t_out) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * times + t_out) * height_col + h_out) * width_col + w_out; 
    const float offset_h = data_offset_ptr[data_offset_h_ptr];
    const float offset_w = data_offset_ptr[data_offset_w_ptr];
    const float cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const float cur_inv_w_data = w_in + j * dilation_w + offset_w;    
    const int weight_offset = ((c * times + t_out) * kernel_h + i) * kernel_w + j;
    const float weight_in = data_weight[weight_offset];
    const int former_ptr = (((n_out * channels + c) * times + t_former) * height_col + h_out) * width_col + w_out;
    const float bottom_former = data_im[former_ptr];

    const float cur_top_grad = top_diff[index_out];
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;

    // gradient to the former (t_former) frame
    if (cur_inv_h_data > -1 && cur_inv_w_data > -1 && cur_inv_h_data < height && cur_inv_w_data < width){
      const float* data_im_ptr = data_im + ((n_out * channels + c) * times + t_out) * height * width;
      float next_val = im2col_bilinear_cuda(data_im_ptr, width, height, width, cur_inv_h_data, cur_inv_w_data);
      atomicAdd(grad_im + former_ptr, weight_in * next_val * cur_top_grad);
    }
    
    // gradient to the current (t_out) frame
    for (int dy = -2; dy <= 2; dy++)
    {
      for (int dx = -2; dx <= 2; dx++)
      {
        if (cur_h + dy >= 0 && cur_h + dy < height &&
            cur_w + dx >= 0 && cur_w + dx < width &&
            abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1)
        {
          int cur_bottom_grad_pos = (((n_out * channels + c) * times + t_out) * height + cur_h + dy) * width + cur_w + dx;
          float weight = get_gradient_weight_cuda(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * weight_in * bottom_former * cur_top_grad);
        }
      }
    }
  }
}

__global__ void defcor_col2im_weight_gpu_kernel(const int n, const float *top_diff,
                                                const float *data_im, const float *data_offset, const float *data_weight,
                                                const int batch_size, const int channels, const int times, const int height, const int width,
                                                const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kernel_w,
                                                const int pad_h, const int pad_w,
                                                const int stride_h, const int stride_w,
                                                const int dilation_h, const int dilation_w, const int channel_coord, const int defcor_group,
                                                float *grad_we)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    // index: batch_size x channel_col x times x height_col x width_col
    const int w_out = index % width_col;
    const int h_out = (index / width_col) % height_col;
    const int t_out = (index / width_col / height_col) % times;
    const int c_out = (index / width_col / height_col / times) % channel_col; 
    const int n_out = (index / width_col / height_col / times / channel_col);

    const int group_id = c_out / (kernel_w * kernel_h);
    const int c_in_per_defcor =  channels / defcor_group;
  
    int t_former = t_out - 1;
    if (t_out <= 0){
      t_former = 0;
    }
    const float *data_offset_ptr = data_offset + n_out * 2 * kernel_h * kernel_w * times * height_col * width_col;

    int kh = (c_out / kernel_w) % kernel_h;
    int kw = (c_out % kernel_w);
    
    // the next frame position
    const int h_in = h_out * stride_h - pad_h + kh * dilation_h;
    const int w_in = w_out * stride_w - pad_w + kw * dilation_w;

    // the offset of the next frame
    const int data_offset_h_ptr = (((2 * (kh * kernel_w + kw)) * times + t_out) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = (((2 * (kh * kernel_w + kw) + 1) * times + t_out) * height_col + h_out) * width_col + w_out;
    const float offset_h = data_offset_ptr[data_offset_h_ptr];
    const float offset_w = data_offset_ptr[data_offset_w_ptr];    
    const float h_im = h_in + offset_h;
    const float w_im = w_in + offset_w;
    const float cur_top_grad = top_diff[index];

    if (h_im > -1 && w_im > -1 && h_im < height && w_im < width){
      const int c_in_start = group_id * c_in_per_defcor; 
      const int c_in_end = (group_id + 1) * c_in_per_defcor;
      for (int c_in = c_in_start; c_in < c_in_end; ++c_in){
        const int former_ptr = (((n_out * channels + c_in) * times + t_former) * height + h_out) * width + w_out;
        int weight_offset = ((c_in * times + t_out) * kernel_h + kh) * kernel_w + kw;
        const float* data_im_ptr = data_im + ((n_out * channels + c_in) * times + t_out) * height * width;
        float next_val = im2col_bilinear_cuda(data_im_ptr, width, height, width, h_im, w_im);
        atomicAdd(grad_we + weight_offset, next_val * cur_top_grad * data_im[former_ptr]);
      }
    }
  }
}


__global__ void defcor_col2im_coord_gpu_kernel(const int n, const float *top_diff,
                                              const float *data_im, const float *data_offset, const float *data_weight,
                                              const int batch_size, const int channels, const int times, const int height, const int width,
                                              const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kernel_w, 
                                              const int pad_h, const int pad_w,
                                              const int stride_h, const int stride_w,
                                              const int dilation_h, const int dilation_w, const int channel_coord, const int defcor_group,             
                                              float *grad_offset)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    // the kernel offest: N x (2*k*k) x T x H x W
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int t = (index / width_col / height_col) % times;
    int c = (index / width_col / height_col / times) % channel_coord;
    int n = (index / width_col / height_col / times) / channel_coord;

    int t_former = t - 1;
    if (t <= 0){
      t_former = 0;
    }

    // compute the start and end of the output
    const float *data_offset_ptr = data_offset + n * channel_coord * times * height_col * width_col;

    const int bp_dir = c % 2;
    int h_out = h;
    int w_out = w;
    int t_out = t;
    // offset: h x w x 2
    int fw = (c / 2 % kernel_w);
    int fh = (c / 2 / kernel_w) % kernel_h;
    const int data_offset_h_ptr = (((2 * (fh * kernel_w + fw)) * times + t_out) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = (((2 * (fh * kernel_w + fw) + 1) * times + t_out) * height_col + h_out) * width_col + w_out;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    const float offset_h = data_offset_ptr[data_offset_h_ptr];
    const float offset_w = data_offset_ptr[data_offset_w_ptr];
    float inv_h = h_in + fh * dilation_h + offset_h;
    float inv_w = w_in + fw * dilation_w + offset_w;
    if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
    {
      inv_h = inv_w = -2;
    }
    
    const int c_in_per_defcor = channels / defcor_group;
    for (int group_id = 0; group_id < defcor_group; ++group_id){
      int c_out = (group_id * kernel_h + fh) * kernel_w + fw;
      int index_out =  (((n * channel_col + c_out) * times + t_out) * height_col + h_out) * width_col + w_out;
      const float cur_top_grad = top_diff[index_out];
      
      // sum over the channle 
      float val = static_cast<float>(0);
      const int im_c_start = group_id * c_in_per_defcor;
      const int im_c_end = (group_id + 1) * c_in_per_defcor;
      for (int im_c = im_c_start; im_c < im_c_end; ++im_c)
      {
        // the former frame offset position
        const int former_offset = (((n * channels + im_c) * times + t_former) * height + h) * width + w;
        const int next_offset = ((n * channels + im_c) * times + t_out) * height * width;
        const int weight_offset = ((im_c * times + t_out) * kernel_h + fh) * kernel_w + fw;
        const float next_coord_weight = get_coordinate_weight_cuda(inv_h, inv_w, height, width, data_im + next_offset, width, bp_dir);
        val += (next_coord_weight * data_im[former_offset] * data_weight[weight_offset] * cur_top_grad);
      }
      grad_offset[index] += val;
    }
  }
}


__global__ void defagg_im2col_gpu_kernel(const int n,
                                         const float *data_im, const float *data_offset, const float *data_weight,
                                         const int batch_size, const int channels, const int times, const int height, const int width,
                                         const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kernel_w,
                                         const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                         const int dilation_h, const int dilation_w, const int defagg_group,
                                         float *data_out)
{
  // launch N * C * T * H * W cores
  CUDA_KERNEL_LOOP(index, n)
  {

    // input index: N x C_col x T x H x W 
    const int w_out = index % width_col;
    const int h_out = (index / width_col) % height_col;
    const int t_out = (index / width_col / height_col) % times;
    const int c_out = (index / width_col / height_col / times) % channel_col; 
    const int n_out = (index / width_col / height_col / times / channel_col);
  
    const int c_out_per_defagg =  channel_col / defagg_group;
    const int group_id = c_out / c_out_per_defagg;
    const int weight_channels = defagg_group * kernel_h * kernel_w;

    const float *data_offset_ptr = data_offset + n_out * 2 * kernel_h * kernel_w * times * height_col * width_col;


    // aggregation of all the deformable positions
    float val = static_cast<float>(0);
    for (int kh = 0; kh < kernel_h; ++kh){
      for (int kw = 0; kw < kernel_w; ++kw){
        const int h_in = h_out * stride_h - pad_h + kh * dilation_h;
        const int w_in = w_out * stride_w - pad_w + kw * dilation_w;
        const int data_offset_h_ptr = (((2 * (kh * kernel_w + kw)) * times + t_out) * height_col + h_out) * width_col + w_out;
        const int data_offset_w_ptr = (((2 * (kh * kernel_w + kw) + 1) * times + t_out) * height_col + h_out) * width_col + w_out;
        const float offset_h = data_offset_ptr[data_offset_h_ptr];
        const float offset_w = data_offset_ptr[data_offset_w_ptr];    
        const float h_im = h_in + offset_h;
        const float w_im = w_in + offset_w; 
        const int c_in_img = c_out % c_out_per_defagg;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width){
          const int w_channels = (group_id * kernel_h + kh) * kernel_w + kw; 
          const int weight_offset = (((n_out * weight_channels + w_channels) * times + t_out) * height + h_out) * width + w_out;
          const float* data_im_ptr = data_im + ((n_out*channels+c_in_img)*times + t_out)*height*width;
          float def_value = im2col_bilinear_cuda(data_im_ptr, width, height, width, h_im, w_im);
          val += (data_weight[weight_offset] * def_value);
        }
      }
    }
    data_out[index] = val;
  }
}

__global__ void defagg_col2im_gpu_kernel(const int n, const float *top_diff,
                                         const float *data_im, const float *data_offset, const float *data_weight,
                                         const int batch_size, const int channels, const int times, const int height, const int width,
                                         const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kernel_w,
                                         const int pad_h, const int pad_w,
                                         const int stride_h, const int stride_w,
                                         const int dilation_h, const int dilation_w, const int channel_coord, const int defagg_group,
                                         float *grad_im)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    const int weight_channels = defagg_group * kernel_h * kernel_w;

    // The index: channels x group_id x times x kernel_h x kernel_w x batch_size x height_col x width_col 
    const int j = (index / width_col / height_col / batch_size ) % kernel_w; // the w of kernel position 
    const int i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h; // the h of kernel position
    const int group_id = (index / width_col / height_col / batch_size / kernel_w / kernel_h / times) % defagg_group; 
    const int c = (index / width_col / height_col / batch_size / kernel_w / kernel_h / times / defagg_group);
    const int c_out = group_id * channels + c;

  
    int w_out = (index % width_col); 
    int h_out = (index / width_col) % height_col; 
    int t_out = (index / width_col / height_col / batch_size / kernel_w / kernel_h) % times;
    int n_out = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w; 
    int h_in = h_out * stride_h - pad_h; 
    int index_out =  (((n_out * channel_col + c_out) * times + t_out) * height_col + h_out) * width_col + w_out;
  

    const float *data_offset_ptr = data_offset + n_out * 2 * kernel_h * kernel_w * times * height_col * width_col;
    const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * times + t_out) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * times + t_out) * height_col + h_out) * width_col + w_out; 
    const float offset_h = data_offset_ptr[data_offset_h_ptr];
    const float offset_w = data_offset_ptr[data_offset_w_ptr];
    const float cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const float cur_inv_w_data = w_in + j * dilation_w + offset_w;
    const int w_channels = (group_id * kernel_h + i) * kernel_w + j; 
    const int weight_offset = (((n_out * weight_channels + w_channels) * times + t_out) * height + h_out) * width + w_out;
    const float weight_in = data_weight[weight_offset];
    //const int cur_ptr = (((n_out * channels + c) * times + t_out) * height_col + h_out) * width_col + w_out;
    //const float bottom_cur = data_im[cur_ptr];

    const float cur_top_grad = top_diff[index_out];
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;

    // gradient to the former (t_former) frame
    //if (cur_inv_h_data > -1 && cur_inv_w_data > -1 && cur_inv_h_data < height && cur_inv_w_data < width){
    //  const float* data_im_ptr = data_im + ((n_out * channels + c) * times + t_out) * height * width;
    //  float next_val = im2col_bilinear_cuda(data_im_ptr, width, height, width, cur_inv_h_data, cur_inv_w_data);
    //  atomicAdd(grad_im + former_ptr, weight_in * next_val * cur_top_grad);
    //}
    
    // gradient to the current (t_out) frame
    for (int dy = -2; dy <= 2; dy++)
    {
      for (int dx = -2; dx <= 2; dx++)
      {
        if (cur_h + dy >= 0 && cur_h + dy < height &&
            cur_w + dx >= 0 && cur_w + dx < width &&
            abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1)
        {
          int cur_bottom_grad_pos = (((n_out * channels + c) * times + t_out) * height + cur_h + dy) * width + cur_w + dx;
          float weight = get_gradient_weight_cuda(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * weight_in * cur_top_grad);
        }
      }
    }
  }
}

__global__ void defagg_col2im_weight_gpu_kernel(const int n, const float *top_diff,
                                                const float *data_im, const float *data_offset, const float *data_weight,
                                                const int batch_size, const int channels, const int times, const int height, const int width,
                                                const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kernel_w,
                                                const int pad_h, const int pad_w,
                                                const int stride_h, const int stride_w,
                                                const int dilation_h, const int dilation_w, const int channel_coord, const int defagg_group,
                                                float *grad_we)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    // index: batch_size x channel_col x times x height_col x width_col
    const int w_out = index % width_col;
    const int h_out = (index / width_col) % height_col;
    const int t_out = (index / width_col / height_col) % times;
    const int c_out = (index / width_col / height_col / times) % channel_col; 
    const int n_out = (index / width_col / height_col / times / channel_col);

    const int c_out_per_defagg =  channel_col / defagg_group;
    const int group_id = c_out / c_out_per_defagg;
    const int weight_channels = defagg_group * kernel_h * kernel_w;
  
    const float *data_offset_ptr = data_offset + n_out * 2 * kernel_h * kernel_w * times * height_col * width_col;
    
    const float cur_top_grad = top_diff[index];
    for (int kh = 0; kh < kernel_h; ++kh){
      for (int kw = 0; kw < kernel_w; ++kw){
        const int h_in = h_out * stride_h - pad_h + kh * dilation_h;
        const int w_in = w_out * stride_w - pad_w + kw * dilation_w;
        const int data_offset_h_ptr = (((2 * (kh * kernel_w + kw)) * times + t_out) * height_col + h_out) * width_col + w_out;
        const int data_offset_w_ptr = (((2 * (kh * kernel_w + kw) + 1) * times + t_out) * height_col + h_out) * width_col + w_out;
        const float offset_h = data_offset_ptr[data_offset_h_ptr];
        const float offset_w = data_offset_ptr[data_offset_w_ptr];
        const float h_im = h_in + offset_h;
        const float w_im = w_in + offset_w;
        const int c_in = c_out % c_out_per_defagg;
        const float* data_im_ptr = data_im + ((n_out * channels + c_in) * times + t_out) * height * width;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width){
            const int w_channels = (group_id * kernel_h + kh) * kernel_w + kw; 
            const int weight_offset = (((n_out * weight_channels + w_channels) * times + t_out) * height + h_out) * width + w_out;
            float next_val = im2col_bilinear_cuda(data_im_ptr, width, height, width, h_im, w_im);
            atomicAdd(grad_we + weight_offset, next_val * cur_top_grad);
        }
      }
    }
  }
}

__global__ void defagg_col2im_coord_gpu_kernel(const int n, const float *top_diff,
                                              const float *data_im, const float *data_offset, const float *data_weight,
                                              const int batch_size, const int channels, const int times, const int height, const int width,
                                              const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kernel_w, 
                                              const int pad_h, const int pad_w,
                                              const int stride_h, const int stride_w,
                                              const int dilation_h, const int dilation_w, const int channel_coord, const int defagg_group,             
                                              float *grad_offset)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    // the kernel offest: N x (2*k*k) x T x H x W
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int t = (index / width_col / height_col) % times;
    int c = (index / width_col / height_col / times) % channel_coord;
    int n = (index / width_col / height_col / times) / channel_coord;


    // compute the start and end of the output
    const float *data_offset_ptr = data_offset + n * channel_coord * times * height_col * width_col;
    const int weight_channels = defagg_group * kernel_h * kernel_w;

    const int bp_dir = c % 2;
    int h_out = h;
    int w_out = w;
    int t_out = t;
    // offset: h x w x 2
    int fw = (c / 2 % kernel_w);
    int fh = (c / 2 / kernel_w) % kernel_h;
    const int data_offset_h_ptr = (((2 * (fh * kernel_w + fw)) * times + t_out) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = (((2 * (fh * kernel_w + fw) + 1) * times + t_out) * height_col + h_out) * width_col + w_out;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    const float offset_h = data_offset_ptr[data_offset_h_ptr];
    const float offset_w = data_offset_ptr[data_offset_w_ptr];
    float inv_h = h_in + fh * dilation_h + offset_h;
    float inv_w = w_in + fw * dilation_w + offset_w;
    if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
    {
      inv_h = inv_w = -2;
    }
    
    const int c_out_per_defagg = channel_col / defagg_group; // equal channels
    float val = static_cast<float>(0);
    for (int group_id = 0; group_id < defagg_group; ++group_id){
      // sum over the channle 
      const int im_c_start = group_id * c_out_per_defagg;
      const int im_c_end = (group_id + 1) * c_out_per_defagg;
      for (int im_c = im_c_start; im_c < im_c_end; ++im_c){
        // the output channels is defagg_group * channels
        int index_out = (((n * channel_col + im_c) * times + t_out) * height_col + h_out) * width_col + w_out;
        const int im_c_in = im_c % c_out_per_defagg;
        const int cur_offset = ((n * channels + im_c_in) * times + t_out) * height * width;
        const int w_channels = (group_id * kernel_h + fh) * kernel_w + fw;
        const int weight_offset = (((n * weight_channels + w_channels) * times + t_out) * height + h_out) * width + w_out;
        const float cur_coord_weight = get_coordinate_weight_cuda(inv_h, inv_w, height, width, data_im + cur_offset, width, bp_dir);
        const float cur_top_grad = top_diff[index_out];
        val += (cur_coord_weight * data_weight[weight_offset] * cur_top_grad);
      }
    }
    grad_offset[index] = val;
  }
}


// main function

void defcor_im2col_forward_cuda(cudaStream_t stream,
  const float* data_im, const float* data_offset, const float* data_weight,
  const int batch_size, const int channels, const int times, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int defcor_group, float* data_out){

  const int num_kernels =  batch_size * channel_col * times * height_col * width_col;
  defcor_im2col_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
      num_kernels, data_im, data_offset, data_weight, 
      batch_size, channels, times, height_im, width_im,
      height_col, width_col, channel_col, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, 
      dilation_h, dilation_w, defcor_group, data_out);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

void defcor_col2im_cuda(cudaStream_t stream,
  const float* grad_out, const float* data_im, const float* data_offset, const float* data_weight,
  const int batch_size, const int channels, const int times, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w,
  const int defcor_group, float* grad_im){

  const int num_kernels = channels * times * kernel_h * kernel_w * batch_size  * height_col * width_col;
  const int channel_coord = 2 * kernel_h * kernel_w;
  defcor_col2im_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
        num_kernels, grad_out, data_im, data_offset, data_weight, 
        batch_size, channels, times, height_im, width_im,
        height_col, width_col, channel_col, kernel_h, kernel_w, 
        pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, channel_coord, defcor_group,  grad_im);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in defcor_col2im_cuda: %s\n", cudaGetErrorString(err));
  }

}

void defcor_col2im_weight_cuda(cudaStream_t stream,
  const float* grad_out, const float* data_im, const float* data_offset, const float* data_weight,
  const int batch_size, const int channels, const int times, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w, 
  const int defcor_group, float* grad_we){

  const int num_kernels = batch_size * channel_col * times * height_col * width_col;
  const int channel_coord = 2 * kernel_h * kernel_w;
  defcor_col2im_weight_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
        num_kernels, grad_out, data_im, data_offset, data_weight, 
        batch_size, channels, times, height_im, width_im,
        height_col, width_col, channel_col, kernel_h, kernel_w, 
        pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, channel_coord, defcor_group, grad_we);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in defcor_col2im_weight_cuda: %s\n", cudaGetErrorString(err));
  }
}

void defcor_col2im_coord_cuda(cudaStream_t stream,
  const float* grad_out, const float* data_im, const float* data_offset, const float* data_weight,
  const int batch_size, const int channels, const int times, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w, 
  const int defcor_group, float* grad_offset){
  const int channel_coord = 2 * kernel_h * kernel_w;
  const int num_kernels = batch_size * channel_coord * times * height_col * width_col;
  defcor_col2im_coord_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
        0, stream>>>(
        num_kernels, grad_out, data_im, data_offset, data_weight,
        batch_size, channels, times, height_im, width_im,
        height_col, width_col, channel_col, kernel_h, kernel_w,
        pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, channel_coord, defcor_group, grad_offset);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in defcor_col2im_coord_cuda: %s\n", cudaGetErrorString(err));
  }
}


void defagg_im2col_forward_cuda(cudaStream_t stream,
  const float* data_im, const float* data_offset, const float* data_weight,
  const int batch_size, const int channels, const int times, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int defagg_group, float* data_out){

  const int num_kernels =  batch_size * channel_col * times * height_col * width_col;
  defagg_im2col_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
      num_kernels, data_im, data_offset, data_weight, 
      batch_size, channels, times, height_im, width_im,
      height_col, width_col, channel_col, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, 
      dilation_h, dilation_w, defagg_group, data_out);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

void defagg_col2im_cuda(cudaStream_t stream,
  const float* grad_out, const float* data_im, const float* data_offset, const float* data_weight,
  const int batch_size, const int channels, const int times, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w,
  const int defagg_group, float* grad_im){

  const int num_kernels = channels * defagg_group * times * kernel_h * kernel_w * batch_size  * height_col * width_col;
  const int channel_coord = 2 * kernel_h * kernel_w;
  defagg_col2im_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
        num_kernels, grad_out, data_im, data_offset, data_weight, 
        batch_size, channels, times, height_im, width_im,
        height_col, width_col, channel_col, kernel_h, kernel_w, 
        pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, channel_coord, defagg_group,  grad_im);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in defagg_col2im_cuda: %s\n", cudaGetErrorString(err));
  }

}

void defagg_col2im_weight_cuda(cudaStream_t stream,
  const float* grad_out, const float* data_im, const float* data_offset, const float* data_weight,
  const int batch_size, const int channels, const int times, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w, 
  const int defagg_group, float* grad_we){

  const int num_kernels = batch_size * channel_col * times * height_col * width_col;
  const int channel_coord = 2 * kernel_h * kernel_w;
  defagg_col2im_weight_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
        num_kernels, grad_out, data_im, data_offset, data_weight, 
        batch_size, channels, times, height_im, width_im,
        height_col, width_col, channel_col, kernel_h, kernel_w, 
        pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, channel_coord, defagg_group, grad_we);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in defagg_col2im_weight_cuda: %s\n", cudaGetErrorString(err));
  }
}

void defagg_col2im_coord_cuda(cudaStream_t stream,
  const float* grad_out, const float* data_im, const float* data_offset, const float* data_weight,
  const int batch_size, const int channels, const int times, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w, 
  const int defagg_group, float* grad_offset){
  const int channel_coord = 2 * kernel_h * kernel_w;
  const int num_kernels = batch_size * channel_coord * times * height_col * width_col;
  defagg_col2im_coord_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
        0, stream>>>(
        num_kernels, grad_out, data_im, data_offset, data_weight,
        batch_size, channels, times, height_im, width_im,
        height_col, width_col, channel_col, kernel_h, kernel_w,
        pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, channel_coord, defagg_group, grad_offset);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in defagg_col2im_coord_cuda: %s\n", cudaGetErrorString(err));
  }
}