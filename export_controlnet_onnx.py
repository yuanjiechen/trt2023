from canny2image_TRT import hackathon
import onnx
import torch
import pickle
import onnxruntime as ort
if __name__ == '__main__':
    hk = hackathon()
    hk.initialize()
    # inputs = (torch.randn((1, 4, 32, 48)).cuda(), torch.randn((1, 3, 256, 384)).cuda(), torch.randn((1,)).cuda(), torch.randn((1, 77, 768)).cuda())
    # with open("./controlnet_full.pkl", "rb") as f:
    #     inputs = pickle.load(f)
    
    with open("./controlnet_vae.pkl", "rb") as f:
        inputs_vae = pickle.load(f)

    with open("./controlnet_one_loop.pkl", "rb") as f:
        inputs_full = pickle.load(f)

    with open("./hint_block.pkl", "rb") as f:
        inputs_hint = pickle.load(f)

    # cuda_inputs = []
    cuda_vae = []
    cuda_full = []
    cuda_hint = []

    # for inp in inputs: cuda_inputs.append( inp.cuda())
    for inp in inputs_vae: cuda_vae.append(inp.cuda())
    for inp in inputs_full: cuda_full.append(inp.cuda())
    for inp in inputs_hint: cuda_hint.append(inp.cuda())

    try:
        # torch.onnx.export(hk.model.eval(), tuple(cuda_inputs), "./onnxs/controlnet_full.onnx", opset_version=17, do_constant_folding=True)
        torch.onnx.export(hk.model.first_stage_model, tuple(cuda_vae), "./onnxs/controlnet_vae.onnx", opset_version=17, do_constant_folding=True)
        torch.onnx.export(hk.ddim_sampler.full_model, tuple(cuda_full), "./onnxs/controlnet_one_loop.onnx", opset_version=17, do_constant_folding=True)
        torch.onnx.export(hk.ddim_sampler.control_input_block, tuple(cuda_hint), "./onnxs/hint_block.onnx", opset_version=17, do_constant_folding=True)
        # sess_options = ort.SessionOptions()
        # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # sess_options.optimized_model_filepath = "./onnxs/controlnet_one_loop.onnx"
        # print("1")
        # ort_session = ort.InferenceSession("./onnxs/controlnet_one_loop.onnx", sess_options, providers=['CUDAExecutionProvider'])
        # print("2")
    except BaseException as e:
        print(e)
        raise