echo "preprocess"
mkdir onnxs
/usr/bin/python3 export_controlnet_onnx.py
cd onnxs
polygraphy surgeon sanitize --fold-constants controlnet_full.onnx -o controlnet_full_folded.onnx
polygraphy surgeon sanitize --fold-constants controlnet_vae.onnx -o controlnet_vae_folded.onnx

trtexec --onnx=./controlnet_full_folded.onnx --exportProfile=./profile.json --exportLayerInfo=./layerinfo.json --profilingVerbosity=detailed --workspace=16384 --fp16 --saveEngine=./controlnet_full_fp16.engine --infStreams=4 --maxAuxStreams=10 --useCudaGraph
trtexec --onnx=./controlnet_vae_folded.onnx --exportProfile=./profile.json --exportLayerInfo=./layerinfo.json --profilingVerbosity=detailed --workspace=16384 --fp16 --saveEngine=./controlnet_vae_fp16.engine --infStreams=4 --maxAuxStreams=10 --useCudaGraph
# trtexec --onnx=./controlnet_with_vae.onnx --exportProfile=./profile.json --exportLayerInfo=./layerinfo.json --profilingVerbosity=detailed --workspace=16384 --fp16 --saveEngine=./controlnet_with_vae_fp16.engine --infStreams=4 --maxAuxStreams=10 --useCudaGraph
# /home/player/.local/bin/polygraphy


mv controlnet_full_fp16.engine ../
mv controlnet_vae_fp16.engine ../
# mv controlnet_with_vae_fp16.engine ../
cd ..
# trtexec --onnx=./controlnet.onnx --exportProfile=./profile.json --exportLayerInfo=./layerinfo.json --profilingVerbosity=detailed --workspace=16384 --saveEngine=./controlnet.engine --infStreams=4