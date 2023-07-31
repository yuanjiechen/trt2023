echo "preprocess"
mkdir onnxs
# /usr/bin/python3 export_controlnet_onnx.py
cd onnxs
# polygraphy surgeon sanitize --fold-constants controlnet_one_loop.onnx -o controlnet_one_loop_folded.onnx --save-external-data 
# polygraphy surgeon sanitize --fold-constants controlnet_vae.onnx -o controlnet_vae_folded.onnx --save-external-data 
# polygraphy surgeon sanitize --fold-constants hint_block.onnx -o hint_block_folded.onnx --save-external-data 

cd ..
# /usr/bin/python3 edit_onnx.py
cd onnxs

# trtexec --onnx=./controlnet_full.onnx --exportProfile=./profile.json --exportLayerInfo=./layerinfo.json --profilingVerbosity=detailed --workspace=16384 --fp16 --saveEngine=./controlnet_full_fp16.engine --infStreams=4 --maxAuxStreams=10 --useCudaGraph
# trtexec --onnx=./controlnet_vae_folded.onnx --workspace=16384 --fp16 --saveEngine=./controlnet_vae_fp16.engine --infStreams=4 --maxAuxStreams=10 --useCudaGraph
trtexec --onnx=./controlnet_one_loop_folded_I.onnx --exportProfile=./profile_one.json --exportLayerInfo=./layerinfo_one.json --profilingVerbosity=detailed --workspace=16384 --fp16 --saveEngine=./controlnet_one_loop_fp16.engine --infStreams=4 --maxAuxStreams=10 --useCudaGraph --heuristic --precisionConstraints=prefer --layerPrecisions=*:fp16 --staticPlugins=../lib/libnvinfer_plugin.so #--layerOutputTypes=*:fp16 
# trtexec --onnx=./hint_block_folded.onnx --workspace=16384 --fp16 --saveEngine=./hint_block_fp16.engine --infStreams=4 --maxAuxStreams=10 --useCudaGraph

# /home/player/.local/bin/polygraphy
# mv controlnet_full_fp16.engine ../
mv controlnet_vae_fp16.engine ../
mv controlnet_one_loop_fp16.engine ../
mv hint_block_fp16.engine ../
cd ..
# trtexec --onnx=./controlnet.onnx --exportProfile=./profile.json --exportLayerInfo=./layerinfo.json --profilingVerbosity=detailed --workspace=16384 --saveEngine=./controlnet.engine --infStreams=4

# python3 setup.py install --user
# ninja package

# cmake .. -DTRT_LIB_DIR=/home/player/TensorRT-8.6.1.6/lib