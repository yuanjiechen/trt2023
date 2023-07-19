from canny2image_TRT import hackathon
import onnx
import torch
import pickle

if __name__ == '__main__':
    hk = hackathon()
    hk.initialize()
    # inputs = (torch.randn((1, 4, 32, 48)).cuda(), torch.randn((1, 3, 256, 384)).cuda(), torch.randn((1,)).cuda(), torch.randn((1, 77, 768)).cuda())
    with open("./controlnet_full.pkl", "rb") as f:
        inputs = pickle.load(f)
    
    with open("./controlnet_vae.pkl", "rb") as f:
        inputs_vae = pickle.load(f)

    # with open("./controlnet_with_vae.pkl", "rb") as f:
        # inputs_full = pickle.load(f)

    cuda_inputs = []
    cuda_vae = []
    # cuda_full = []

    for inp in inputs: cuda_inputs.append( inp.cuda())
    for inp in inputs_vae: cuda_vae.append(inp.cuda())
    # for inp in inputs_full: cuda_full.append(inp.cuda())

    try:
        torch.onnx.export(hk.model.eval(), tuple(cuda_inputs), "./onnxs/controlnet_full.onnx", opset_version=17, do_constant_folding=True)
        torch.onnx.export(hk.model.first_stage_model, tuple(cuda_vae), "./onnxs/controlnet_vae.onnx", opset_version=17, do_constant_folding=True)
        # torch.onnx.export(hk.model.full_model, tuple(cuda_full), "./onnxs/controlnet_with_vae.onnx", opset_version=17, do_constant_folding=True)
    except BaseException as e:
        print(e)
        raise