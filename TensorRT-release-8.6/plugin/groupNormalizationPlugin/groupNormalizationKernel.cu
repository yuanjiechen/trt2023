/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "groupNormalizationPlugin.h"
#include "common/bertCommon.h"
#include "common/common.cuh"
#include "common/serialize.hpp"
#include <cuda.h>
namespace nvinfer1
{
namespace plugin
{

template <typename T, unsigned TPB>
__global__ void scaleShiftChannelsInplaceKernel(T* inOut, const int ld, const int skips, const T* beta, const T* gamma)
{
    // grid is blocks x C x B
    // ld should be H*W
    // blockIdx.z = batch
    // blockIdx.y = channel
    // blockIdx.x = block per col
    // ld = 3840

    // const T b = beta[blockIdx.y * skips];
    // const T g = gamma[blockIdx.y * skips];
    const int skip_offset = blockIdx.y * ld;
    const int offset = blockIdx.z * gridDim.y * ld + skip_offset; //(blockIdx.z * gridDim.y + blockIdx.y) * ld;
    const int data_offset = (skip_offset + tx) / skips
    const int tx = blockIdx.x * TPB + threadIdx.x;

    const T b = beta[data_offset];
    const T g = gamma[data_offset];

    if (tx < ld)
    {
        inOut[offset + tx] = g * inOut[offset + tx] + b;
    }
}

template <typename T>
cudaError_t scaleShiftChannelsInplace(T* inOut, const int B, const int C, const int channelVolume, const int skips, const T* beta,
    const T* gamma, cudaStream_t stream)
{

    constexpr int TPB = 256;
    const int colBlocks = (channelVolume + TPB - 1) / TPB;
    const dim3 grid(colBlocks, C, B); // 16, 32, 2
    const int skips_ = channelVolume * C / skips;
    // std::cout << "before cuda kernel\n";
    // std::cout << sizeof(T) << endl;
    scaleShiftChannelsInplaceKernel<T, TPB><<<grid, TPB, 0, stream>>>(inOut, channelVolume, skips_, beta, gamma); // (16, 32, 2) 256
    // cudaStreamSynchronize(stream);
    // cudaError_t err = cudaPeekAtLastError();
    // std::cout  << err << "    after cuda kernel\n";

    return cudaPeekAtLastError();
}

template cudaError_t scaleShiftChannelsInplace<float>(float* inOut, const int B, const int C, const int channelVolume, const int skips, const float* beta,
    const float* gamma, cudaStream_t stream);

template cudaError_t scaleShiftChannelsInplace<half>(half* inOut, const int B, const int C, const int channelVolume, const int skips, const half* beta,
    const half* gamma, cudaStream_t stream);
} /* plugin */
} /* nvinfer1 */
