#
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import numpy as np
from cuda import cudart
import tensorrt as trt
from common import allocate_buffers, memcpy_device_to_host, memcpy_host_to_device, memcopy_device_to_device
from pathlib import Path
import pickle
import torch

class MyCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, nCalibration, path, cacheFile):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.nCalibration = nCalibration
        self.file_path = path
        self.file_len = len(list(Path(self.file_path).glob("*.pkl")))
        assert self.nCalibration <= self.file_len, "Too large calib number"
        self.allocate_mem()

        self.cacheFile = cacheFile
        self.count = 0

    def __del__(self):
        for item in self.dIn:
            cudart.cudaFree(item)

    def allocate_mem(self):
        for fl in Path(self.file_path).glob("*.pkl"):
            with open(fl, "rb") as f:
                sample_inputs = pickle.load(f)
            break
        
        self.dIn = []
        for inp in sample_inputs:
            inp = inp.cuda()
            allocate_size = trt.volume(inp.size()) * trt.float32.itemsize
            _, pointer = cudart.cudaMalloc(allocate_size)
            self.dIn.append(int(pointer))


    def get_batch_size(self):  # do NOT change name
        return 1

    def get_batch(self, nameList=None, inputNodeName=None):  # do NOT change name
        if self.count < self.nCalibration:
            files = list(Path(self.file_path).glob("*.pkl"))[self.count]
            self.count += 1
            with open(files, "rb") as f:
                inputs = pickle.load(f)
            
            for i, inp in enumerate(inputs):
                # torch.Tensor.contiguous
                inp = inp.contiguous().cuda()
                inp_size = trt.volume(inp.size()) * trt.float32.itemsize
                cudart.cudaMemcpy(self.dIn[i], inp.data_ptr(), inp_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
            # data = np.ascontiguousarray(np.random.rand(np.prod(self.shape)).astype(np.float32).reshape(*self.shape) * 200 - 100)
            # cudart.cudaMemcpy(self.dIn, data.ctypes.data, self.buffeSize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            return self.dIn
        else:
            return None

    def read_calibration_cache(self):  # do NOT change name
        if os.path.exists(self.cacheFile):
            print("Succeed finding cahce file: %s" % (self.cacheFile))
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return

    def write_calibration_cache(self, cache):  # do NOT change name
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")

if __name__ == "__main__":
    cudart.cudaDeviceSynchronize()
    m = MyCalibrator(5, (1, 1, 28, 28), "./int8.cache")

