echo "preprocess"
mkdir onnxs
/usr/bin/python3 export_controlnet_onnx.py
cd onnxs
trtexec --onnx=./controlnet_full.onnx --exportProfile=./profile.json --exportLayerInfo=./layerinfo.json --profilingVerbosity=detailed --workspace=16384 --fp16 --saveEngine=./controlnet_fp16.engine --infStreams=4
mv controlnet_fp16.engine ../
cd ..
# trtexec --onnx=./controlnet.onnx --exportProfile=./profile.json --exportLayerInfo=./layerinfo.json --profilingVerbosity=detailed --workspace=16384 --saveEngine=./controlnet.engine --infStreams=4