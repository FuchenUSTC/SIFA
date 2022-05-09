# [CVPR 2022] Stand-Alone Inter-Frame Attention in Video Models

This repository includes the SIFA codes and related configurations. The architecture SIFA-Net and SIFA-Transformer is implemented with python in PyTorch framework. And the kernel of SIFA operation is programmed in C++ with CUDA Library.

The related pre-trained model weigths will be released soon.

# Update 
* 2022.5.8: Repository for SIFA and related training configurations

# Contents:

* [Paper Introduction](#paper-introduction)
* [Required Environment](#required-environment)
* [Compiling of SIFA](#compiling-of-sifa)
* [Training of SIFA](#training-of-sifa)
* [Citation](#citation)

# Paper Introduction

# Required Environment

- python 3.8.0
- Pytorch 1.7
- CUDA 10.1
- cuDNN 8.0
- GPU NVIDIA Tesla V100 (16GB x8)

To guarantee the success of compiling SIFA cuda kernel, the nvcc cuda compiler should be installed in the environment. We have integrated the complete running environment into a docker image and will release it on DockerHub in the future.

# Compiling of SIFA

```
cd ./cuda_sifa
```
Then check the compilation paramter `-gencode=arch` and `code` in the `setup.py` to match the GPU type, e.g., sm_70 for Tesla V100. Then, run
```
bash make.sh
```
If the compilation is successful, the C++ extention file `_ext.cpython-38-x86_64-linux-gnu.so` will be generated in the `./cuda_sifa` folder. Copy it to the main directory.
```
cp _ext.cpython-38-x86_64-linux-gnu.so ../
```

# Training of SIFA

If the frame data has been prepared, please run 
```
python -m torch.distributed.launch --nproc_per_node=4 train_val_3d.py --config_file=settings/c2d_sifa_resnet50-1x16x1.k400.yml
```
or 
```
python -m torch.distributed.launch --nproc_per_node=4 train_val_3d.py --config_file=settings/c2d_sifa_swin-b-1x64x2.k400.yml
```
for the training of SIFA-Net or SIFA-Transformer.

The related training configuration files can be checked in the `.yml` files in the folder of `./base_config/` and `./settings`.


# Citation
