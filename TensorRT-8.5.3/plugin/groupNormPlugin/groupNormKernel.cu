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

#include "groupNormKernel.h"

#include <cub/cub.cuh>
#include <cuda_fp16.h>

static inline __device__ __host__ float sigmoid(float x)
{
    return 1.F / (1.F + expf(-x));
}

static inline __device__  half sigmoid(half x)
{
    return half(1.) / (half(1.) + hexp(-x));
}

struct GroupSums
{
    // Is it the 1st element of the group?
    int32_t flag;
    // The sum.
    float sum;
    // The sum of squares.
    float sumSq;
};

struct GroupSumsOp
{
    inline __device__ GroupSums operator()(GroupSums const& a, GroupSums const& b)
    {
        GroupSums dst;
        dst.sum = b.flag ? b.sum : (a.sum + b.sum);
        dst.sumSq = b.flag ? b.sumSq : (a.sumSq + b.sumSq);
        dst.flag = a.flag + b.flag;
        return dst;
    }
};

template <int32_t tTHREADS_PER_BLOCK>
__global__ void groupNormNHWCSumKernel(GroupNormNHWCParams params)
{
    // The object in charge of doing the sums for the different blocks.
    typedef cub::BlockScan<GroupSums, tTHREADS_PER_BLOCK> BlockScan;

    // Allocate shared memory for BlockScan.
    __shared__ typename BlockScan::TempStorage tempStorage;
    // Allocate shared memory for the groups. We could reduce the amount of shared
    // memory reserved.
    __shared__ float2 smem[tTHREADS_PER_BLOCK];

    // The instance in the batch.
    int32_t ni = blockIdx.z;
    // The channel loaded by that thread (2 channels per thread for F16x2).
    int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * 2;

    // The first activation loaded by that block.
    int32_t hwBegin = blockIdx.y * params.hwPerBlock;
    // The last activation loaded by that block.
    int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);

    // The sums.
    float sum = 0.F;
    float sumSq = 0.F;

    // Iterate over the activations to compute the sums.
    for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi)
    {
        // The offset.
        int64_t offset = static_cast<int64_t>(ni) * params.hwc + static_cast<int64_t>(hwi) * params.c + ci;

        // Fetch two channels per thread.
        __half2 h2(0, 0);
        if (ci < params.c)
        {
            h2 = *reinterpret_cast<__half2 const*>(&params.src[offset]);
        }

        // Extract the two half values.
        // float2 f2 = __half22float2(h2);

        // Update the sum.
        sum += __half2float(h2.x + h2.y);
        // Update the sum of squares.
        sumSq += __half2float(h2.x * h2.x + h2.y * h2.y);
    }

    // The group that thread works on and the channel in the group (modulus).
    int32_t gi = threadIdx.x * 2 / params.cPerGroup;
    int32_t cj = threadIdx.x * 2 - params.cPerGroup * gi;

    // The data for the summations.
    GroupSums inp{cj == 0 ? 1 : 0, sum, sumSq};

    // Do the segmented scan.
    GroupSums out;
    BlockScan(tempStorage).InclusiveScan(inp, out, GroupSumsOp());

    // Store the results for the groups in shared memory (to produce coalesced
    // stores later).
    if (cj == params.cPerGroup - 2 /* 2 channels per thread */)
    {
        smem[gi] = make_float2(out.sum, out.sumSq);
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The global group index.
    int32_t gj = blockIdx.x * params.groupsPerBlock + threadIdx.x;

    // Threads that have nothing left to do, exit.
    if (threadIdx.x >= params.groupsPerBlock || gj >= params.groups)
    {
        return;
    }

    // The first threads (those storing to global memory, load the values).
    float2 sums = smem[threadIdx.x];

    // Store to global memory.
    atomicAdd(&params.redBuffer[(2 * ni + 0) * params.groups + gj], sums.x);
    atomicAdd(&params.redBuffer[(2 * ni + 1) * params.groups + gj], sums.y);
}

void groupNormNHWCSum(GroupNormNHWCParams const& params, cudaStream_t stream)
{
    // Make sure the values are as we expect.
    PLUGIN_ASSERT(params.c % params.cPerBlock == 0 && params.hw % params.hwPerBlock == 0);
    // Make sure a group does not span multiple blocks.
    PLUGIN_ASSERT(params.cPerBlock % params.cPerGroup == 0);

    dim3 grid;

    // The number of blocks to compute all the channels.
    grid.x = params.c / params.cPerBlock;
    // The number of blocks to compute all the activations in a given instance.
    grid.y = divUp(params.hw, params.hwPerBlock);
    // The number of instances.
    grid.z = params.n;

    switch (params.cPerBlock)
    {
    case 320: groupNormNHWCSumKernel<160><<<grid, 160, 0, stream>>>(params); break;
    case 480: groupNormNHWCSumKernel<256><<<grid, 256, 0, stream>>>(params); break;
    case 256: groupNormNHWCSumKernel<128><<<grid, 128, 0, stream>>>(params); break;
    case 128: groupNormNHWCSumKernel<64><<<grid, 64, 0, stream>>>(params); break;
    default: PLUGIN_FAIL("Not implemented");
    }

    PLUGIN_CUASSERT(cudaGetLastError());
}

template <int32_t tTHREADS_PER_BLOCK>
__global__ void groupNormNHWCScaleKernel(GroupNormNHWCParams params)
{
    __shared__ __half2 gammaH2[tTHREADS_PER_BLOCK], betaH2[tTHREADS_PER_BLOCK];
    // The instance in the batch.
    int32_t ni = blockIdx.z;
    // The channel loaded by that thread (2 channels per thread for F16x2).
    int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * 2;
    // The group that thread works on and the channel in the group (modulus).
    int32_t gi = ci / params.cPerGroup;

    // Load the sum and sum of squares for the group.
    float sum = 0.F, sumSq = 0.F;
    if (gi < params.groups)
    {
        sum = params.redBuffer[(2 * ni + 0) * params.groups + gi];
        sumSq = params.redBuffer[(2 * ni + 1) * params.groups + gi];
    }

    // Load gamma/beta.
    // __half2 gammaF2, betaF2;
    // __half2 gammaH2(0, 0), betaH2(0, 0);
    if (ci < params.c)
    {
        gammaH2[threadIdx.x] = *reinterpret_cast<half2 const*>(&params.gamma[ci]);
        betaH2[threadIdx.x] = *reinterpret_cast<half2 const*>(&params.beta[ci]);

        // gammaH2 = __float22half2_rn(gammaF2);
        // betaH2 = __float22half2_rn(betaF2);

    }

    // Compute the mean.
    float mean = sum * params.invHWC;
    // Compute the variance.
    float var = sumSq * params.invHWC - (mean * mean);
    // Compute the inverse of the stddev.
    float invStdDev = var <= 0.F ? 1.F : rsqrtf(var);

    // The first activation loaded by that block.
    int32_t hwBegin = blockIdx.y * params.hwPerBlock;
    // The last activation loaded by that block.
    int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);
    __half2 h2(0, 0);
    // float2 f2 = __half22float2(h2);
    half mean_ = __float2half(mean);
    half invStdDev_ = __float2half(invStdDev);
    // Iterate over the activations to compute the sums.
    for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi)
    {
        // The src/dst offset.
        int64_t offset = (int64_t) ni * params.hwc + hwi * params.c + ci;

        // Fetch two channels per thread.
        // __half2 h2(0, 0);
        if (ci < params.c)
        {
            h2 = *reinterpret_cast<__half2 const*>(&params.src[offset]);
        }

        // Extract the two half values.
        // f2 = __half22float2(h2);

        // Normalize the channels.
        h2.x = (h2.x - mean_) * invStdDev_;
        h2.y = (h2.y - mean_) * invStdDev_;

        // Scale by gamma and add beta.
        h2.x = gammaH2[threadIdx.x].x * h2.x + betaH2[threadIdx.x].x;
        h2.y = gammaH2[threadIdx.x].y * h2.y + betaH2[threadIdx.x].y;

        // Apply Swish if needed.
        if (params.withSwish)
        {
            h2.x = h2.x * sigmoid(h2.x);
            h2.y = h2.y * sigmoid(h2.y);
        }

        // Store the scaled values.
        if (ci < params.c)
        {
            *reinterpret_cast<__half2*>(&params.dst[offset]) = h2; //__float22half2_rn(f2);
        }
    }
}

void groupNormNHWCScale(GroupNormNHWCParams const& params, cudaStream_t stream)
{
    // Make sure the dimensions are aligned with what we expect.
    PLUGIN_ASSERT(params.c % params.cPerBlock == 0);
    // Make sure a group does not span multiple blocks.
    PLUGIN_ASSERT(params.cPerBlock % params.cPerGroup == 0);

    dim3 grid;

    // The number of blocks to compute all the channels.
    grid.x = params.c / params.cPerBlock;
    // The number of blocks to compute all the activations in a given instance.
    grid.y = divUp(params.hw, params.hwPerBlock);
    // The number of instances.
    grid.z = params.n;

    switch (params.cPerBlock)
    {
    case 320: groupNormNHWCScaleKernel<160><<<grid, 160, 0, stream>>>(params); break;
    case 480: groupNormNHWCScaleKernel<256><<<grid, 256, 0, stream>>>(params); break;
    case 256: groupNormNHWCScaleKernel<128><<<grid, 128, 0, stream>>>(params); break;
    case 128: groupNormNHWCScaleKernel<64><<<grid, 64, 0, stream>>>(params); break;
    default: PLUGIN_FAIL("Not implemented");
    }

    PLUGIN_CUASSERT(cudaGetLastError());
}
