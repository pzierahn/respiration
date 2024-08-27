// Source: https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/channelnorm_package/channelnorm_cuda.cc
#pragma once

#include <ATen/ATen.h>

void channelnorm_kernel_forward(
    at::Tensor& input1,
    at::Tensor& output, 
    int norm_deg);


void channelnorm_kernel_backward(
    at::Tensor& input1,
    at::Tensor& output,
    at::Tensor& gradOutput,
    at::Tensor& gradInput1,
    int norm_deg);
