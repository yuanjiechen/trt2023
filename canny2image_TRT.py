import os

from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

import time
import pickle



quant = False
if quant:
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import quant_modules, tensor_quant, calib
    from util import set_quantizer_by_name
    quant_modules.initialize()
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
class hackathon():

    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        # self.model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
        self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        if quant:

            # for name, module in self.model.named_modules():
            #     if hasattr(module, "quant_add"): 
            #         setattr(module, "quant_add", True)
            #         print(f"Set {name} quant_add as True")
            
            self.quantize()
        self.ddim_sampler = DDIMSampler(self.model)
    
    def quantize(self):
        set_quantizer_by_name(self.model, ['transformer_blocks'], _disabled=True)
        with open("amax_300.pkl", "rb") as f:
            amax_list = pickle.load(f)
            i = 0
            for name, module in self.model.named_modules():
                if isinstance(module, quant_nn.TensorQuantizer) and module._disabled != True:
                    module.register_buffer("_amax", torch.from_numpy(amax_list[i]).float().cuda())
                    i += 1


    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map).float().cuda() / 255.0
            control = control.unsqueeze(0)
            control = einops.rearrange(control, 'b h w c -> b c h w')
            # seed = 1935962553
            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            os.environ["PL_SEED_WORKERS"] = f"{int(False)}"


            # if config.save_memory:
            #     self.model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": control, "c_crossattn": self.model.get_learned_conditioning([prompt + ', ' + a_prompt])}# * num_samples
            un_cond = {"c_concat": control, "c_crossattn": self.model.get_learned_conditioning([n_prompt])}# * num_samples
            shape = (4, H // 8, W // 8)
            # print(shape)
            # if config.save_memory:
            #     self.model.low_vram_shift(is_diffusing=True)

            # self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            # print(type(self.model))
            samples = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            # if config.save_memory:
            #     self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = x_samples #[x_samples[i] for i in range(num_samples)]
        return results