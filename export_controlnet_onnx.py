from canny2image_TRT import hackathon
import onnx
import torch
import pickle

if __name__ == '__main__':
    hk = hackathon()
    hk.initialize()
    # inputs = (torch.randn((1, 4, 32, 48)).cuda(), torch.randn((1, 3, 256, 384)).cuda(), torch.randn((1,)).cuda(), torch.randn((1, 77, 768)).cuda())
    with open("./controlnet.pkl", "rb") as f:
        inputs = pickle.load(f)
    
    cuda_inputs = []
    for inp in inputs:
        cuda_inputs.append( inp.cuda())
    try:
        torch.onnx.export(hk.model.control_model.eval(), tuple(cuda_inputs), "./controlnet.onnx", opset_version=17, do_constant_folding=True)
    except BaseException as e:
        print(e)
        raise