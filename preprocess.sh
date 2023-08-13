echo "preprocess"
mkdir onnxs
/usr/bin/python3 export_controlnet_onnx.py
cd onnxs
polygraphy surgeon sanitize --fold-constants controlnet_one_loop.onnx -o controlnet_one_loop_folded.onnx --save-external-data 
polygraphy surgeon sanitize --fold-constants controlnet_vae.onnx -o controlnet_vae_folded.onnx --save-external-data 
polygraphy surgeon sanitize --fold-constants hint_block.onnx -o hint_block_folded.onnx --save-external-data 

cd ..
/usr/bin/python3 edit_onnx.py
cd TensorRT-8.5.3
mkdir build
cd build
cmake .. -DTRT_LIB_DIR=/home/player/TensorRT-8.6.1.6/lib
make clean
make -j
cp libnvinfer* ../../lib/
cd ../..
cd onnxs

trtexec --onnx=./controlnet_vae_folded.onnx --workspace=16384 --fp16 --saveEngine=./controlnet_vae_fp16.engine --infStreams=4 --maxAuxStreams=10 --useCudaGraph --precisionConstraints=prefer --layerPrecisions=*:fp16 --sparsity=enable #--builderOptimizationLevel=5
trtexec --onnx=./controlnet_one_loop_folded.onnx --exportProfile=./profile_one.json --exportLayerInfo=./layerinfo_one.json --profilingVerbosity=detailed --workspace=22222 --fp16 --saveEngine=./controlnet_one_loop_fp16.engine --infStreams=4 --maxAuxStreams=10  --useSpinWait --noDataTransfers --sparsity=enable  --staticPlugins=../lib/libnvinfer_plugin.so --builderOptimizationLevel=5  # --layerOutputTypes=*:fp16 --precisionConstraints=obey --layerPrecisions=*:fp16  
trtexec --onnx=./hint_block_folded.onnx --workspace=16384 --fp16 --saveEngine=./hint_block_fp16.engine --infStreams=4 --maxAuxStreams=10 --precisionConstraints=prefer --layerPrecisions=*:fp16 --sparsity=enable #--builderOptimizationLevel=5

# int8
# trtexec --onnx=./controlnet_vae_folded.onnx --workspace=16384 --fp16 --int8 --saveEngine=./controlnet_vae_fp16.engine --infStreams=4 --maxAuxStreams=10 --useCudaGraph --sparsity=enable
# trtexec --onnx=./controlnet_one_loop_folded.onnx --exportProfile=./profile_one.json --exportLayerInfo=./layerinfo_one.json --profilingVerbosity=detailed --workspace=20000 --best --saveEngine=./controlnet_one_loop_fp16.engine --infStreams=4 --maxAuxStreams=10  --calib=../int8.cache  --builderOptimizationLevel=5 --staticPlugins=../lib/libnvinfer_plugin.so #--timingCacheFile=../cache.engine --sparsity=enable 
# trtexec --onnx=./hint_block_folded.onnx --workspace=16384 --fp16 --int8 --saveEngine=./hint_block_fp16.engine --infStreams=4 --maxAuxStreams=10 --useCudaGraph 

mv controlnet_vae_fp16.engine ../
mv controlnet_one_loop_fp16.engine ../
mv hint_block_fp16.engine ../
cd ..

# trtexec --loadEngine=./controlnet_one_loop_fp16.engine --exportProfile=./onnxs/profile_one.json --exportLayerInfo=./onnxs/layerinfo_one.json --infStreams=4 --maxAuxStreams=10
# python3 setup.py install --user
# ninja package

# cmake .. -DTRT_LIB_DIR=/home/player/TensorRT-8.6.1.6/lib