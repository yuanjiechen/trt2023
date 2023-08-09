import ctypes
from cuda import cudart
import numpy as np
import os
import tensorrt as trt
from calibrator import MyCalibrator

cachefile = "./int8.cache"
n_calib = 1
path = "./looped_data"

logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(logger, '')
builder = trt.Builder(logger)
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 20 * (2 ** 30))
network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
network = builder.create_network(network_flags)
parser = trt.OnnxParser(network, logger)

onnx_path = os.path.realpath("./onnxs/controlnet_one_loop_folded.onnx")
engine_path = os.path.realpath("./controlnet_one_loop_fp16.engine")
calib_count = 280

with open(onnx_path, "rb") as f:
    print(parser.parse_from_file(onnx_path))

if network.num_outputs == 0:
    last_layer = network.get_layer(network.num_layers - 1)
    network.mark_output(last_layer.get_output(0))

inputs = [network.get_input(i) for i in range(network.num_inputs)]
outputs = [network.get_output(i) for i in range(network.num_outputs)]


print("Network Description")
for input in inputs:
    batch_size = input.shape[0]
    print("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
for output in outputs:
    print("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))
assert batch_size > 0

config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.INT8)
config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
config.int8_calibrator = MyCalibrator(calib_count, "../fix_seed", "int8.cache")
config.max_aux_streams = 10

with builder.build_serialized_network(network, config) as engine, open(engine_path, "wb+") as f:
    f.write(engine)