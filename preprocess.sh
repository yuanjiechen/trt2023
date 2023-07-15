echo "preprocess"
/usr/bin/python3 export_controlnet_onnx.py
trtexec --onnx=./controlnet.onnx --exportProfile=./profile.json --exportLayerInfo=./layerinfo.json --profilingVerbosity=detailed --workspace=16384 --fp16 --saveEngine=./controlnet_fp16.engine --infStreams=4
# trtexec --onnx=./controlnet.onnx --exportProfile=./profile.json --exportLayerInfo=./layerinfo.json --profilingVerbosity=detailed --workspace=16384 --saveEngine=./controlnet.engine --infStreams=4