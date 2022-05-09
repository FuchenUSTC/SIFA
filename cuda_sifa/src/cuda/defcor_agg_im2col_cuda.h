// auth: Fuchen Long
// date: 2021/7/18 

#ifndef DEFCOR_AGG_IM2COL_CUDA
#define DEFCOR_AGG_IM2COL_CUDA

#ifdef __cplusplus
extern "C"
{
#endif

  void defcor_im2col_forward_cuda(cudaStream_t stream,
                          const float *data_im, const float *data_offset, const float *data_weight,
                          const int batch_size, const int channels, const int times, const int height_im, const int width_im,
                          const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kenerl_w,
                          const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                          const int dilation_h, const int dilation_w,
                          const int defcor_group, float *data_out);

  void defcor_col2im_cuda(cudaStream_t stream,
                          const float *grad_out, const float *data_im, const float *data_offset, const float *data_weight,
                          const int batch_size, const int channels, const int times, const int height_im, const int width_im,
                          const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kenerl_w,
                          const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                          const int dilation_h, const int dilation_w,
                          const int defcor_group, float *grad_im);

  void defcor_col2im_coord_cuda(cudaStream_t stream,
                                const float *grak_out, const float *data_im, const float *data_offset, const float *data_weight,
                                const int batch_size, const int channels, const int times, const int height_im, const int width_im,
                                const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kenerl_w,
                                const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                const int dilation_h, const int dilation_w,
                                const int defcor_group, float *grad_offset);

  void defcor_col2im_weight_cuda(cudaStream_t stream,
                                const float *grad_out, const float *data_im, const float *data_offset, const float *data_weight,
                                const int batch_size, const int channels, const int times, const int height_im, const int width_im,
                                const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kenerl_w,
                                const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                const int dilation_h, const int dilation_w,
                                const int defcor_group, float *grad_we);

  void defagg_im2col_forward_cuda(cudaStream_t stream,
                          const float *data_im, const float *data_offset, const float *data_weight,
                          const int batch_size, const int channels, const int times, const int height_im, const int width_im,
                          const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kenerl_w,
                          const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                          const int dilation_h, const int dilation_w,
                          const int defagg_group, float *data_out);

  void defagg_col2im_cuda(cudaStream_t stream,
                          const float *grad_out, const float *data_im, const float *data_offset, const float *data_weight,
                          const int batch_size, const int channels, const int times, const int height_im, const int width_im,
                          const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kenerl_w,
                          const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                          const int dilation_h, const int dilation_w,
                          const int defagg_group, float *grad_im);

  void defagg_col2im_coord_cuda(cudaStream_t stream,
                                const float *grak_out, const float *data_im, const float *data_offset, const float *data_weight,
                                const int batch_size, const int channels, const int times, const int height_im, const int width_im,
                                const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kenerl_w,
                                const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                const int dilation_h, const int dilation_w,
                                const int defagg_group, float *grad_offset);

  void defagg_col2im_weight_cuda(cudaStream_t stream,
                                const float *grad_out, const float *data_im, const float *data_offset, const float *data_weight,
                                const int batch_size, const int channels, const int times, const int height_im, const int width_im,
                                const int height_col, const int width_col, const int channel_col, const int kernel_h, const int kenerl_w,
                                const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                const int dilation_h, const int dilation_w,
                                const int defagg_group, float *grad_we);
#ifdef __cplusplus
}
#endif

#endif