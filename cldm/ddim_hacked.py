"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor

import tensorrt as trt
from cuda import cuda, cudart
import numpy as np
from pathlib import Path
from common import allocate_buffers, memcpy_device_to_host, memcpy_host_to_device, memcopy_device_to_device
import pickle
import time

from transformers import CLIPTextModel
from ldm.modules.diffusionmodules.openaimodel import ResBlock
import ctypes
class Control_Diff_VAE(torch.nn.Module):
    def __init__(self, control_model):
        super().__init__()
        self.control_model = control_model
    
    def forward(self, img, ts, ts_df, 
                conds_all, c_hint, 
                a_t, a_prev, sqrt_one_minus_at):

        model_t, model_uncond = self.control_model(img, ts, ts_df, conds_all, c_hint)
        model_output = model_uncond + 9 * (model_t - model_uncond)

        e_t = model_output

        pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t

        dir_xt = (1. - a_prev).sqrt() * e_t

        img = a_prev.sqrt() * pred_x0 + dir_xt #+ 0.006 * noise

        return img #, intermediates

class Control_input_block(torch.nn.Module):
    def __init__(self, input_block, transformer) -> None:
        super().__init__()
        self.input_block = input_block
        self.transformer = transformer #CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval()

    def forward(self, c_hint, c_cond_txt, u_cond_txt):
        c_hint_out = self.input_block(c_hint, None)
        conds = torch.cat([c_cond_txt, u_cond_txt], dim=0)
        conds_out = self.transformer(input_ids=conds, output_hidden_states=False, return_dict=False)[0]#.last_hidden_state
        return c_hint_out, conds_out #c_cond_out, u_cond_out
class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

        self.device = torch.device("cuda")
        self.timesteps = 12
        self.model.control_model.init_steps(self.timesteps)
        self.model.model.diffusion_model.init_steps(self.timesteps)
        
        self.make_schedule(ddim_num_steps=self.timesteps, ddim_eta=0.0, verbose=False)
        self.full_model = Control_Diff_VAE(self.model)
        self.control_input_block = Control_input_block(self.model.control_model.input_hint_block, self.model.cond_stage_model.transformer)
        ########## Create tensor only once not 20 times !!!!!   
        self.alphas =  torch.reshape(self.ddim_alphas, [len(self.ddim_alphas), 1, 1, 1, 1]).to(torch.float).sqrt().contiguous()
        self.alphas_prev = torch.from_numpy(self.ddim_alphas_prev).to(self.device, torch.float).reshape([len(self.ddim_alphas_prev), 1, 1, 1, 1]).contiguous() 
        self.sqrt_one_minus_alphas = torch.reshape(self.ddim_sqrt_one_minus_alphas, [len(self.ddim_sqrt_one_minus_alphas), 1, 1, 1, 1]).to(torch.float).contiguous() 
        #sigmas = torch.reshape(self.ddim_sigmas, [len(self.ddim_sigmas), 1, 1, 1, 1]).to(torch.float) #[torch.full((1, 1, 1, 1), sigmas[idx], device=self.device) for idx in range(len(sigmas))]
        ##########


        if not Path("controlnet_one_loop_fp16.engine").exists(): self.control_net_use_trt = False
        else: self.control_net_use_trt = True

        if not Path("hint_block_fp16.engine").exists(): self.input_block_use_trt = False
        else: self.input_block_use_trt = True

        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, '')

        if self.control_net_use_trt:
            ctypes.cdll.LoadLibrary("./lib/libnvinfer_plugin.so")
            with open("controlnet_one_loop_fp16.engine", 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
                self.context = model.create_execution_context()
                self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(model)
                self.out_tensor = torch.zeros([1, 4, 32, 48], dtype=torch.float32, device=self.device)
                self.input_name, self.output_name = [], []
                for i in range(model.num_bindings):
                    if model.binding_is_input(i): self.input_name.append(model.get_binding_name(i))
                    else: self.output_name.append(model.get_binding_name(i))
                self.graph_exec = self.get_cuda_graph()

        if self.input_block_use_trt:
            with open("hint_block_fp16.engine", 'rb') as f, trt.Runtime(logger) as runtime:
                model_hint = runtime.deserialize_cuda_engine(f.read())
                self.context_hint = model_hint.create_execution_context()
                self.inputs_hint, self.outputs_hint, self.bindings_hint, self.stream_hint = allocate_buffers(model_hint)
                self.c_hint_trt = torch.zeros([1, 320, 32, 48], dtype=torch.float32, device=self.device)
                self.conds_all_trt = torch.zeros([2, 77, 768], dtype=torch.float32, device=self.device)
                # self.c_cond_txt_trt = torch.zeros([1, 77, 768], dtype=torch.float32, device=self.device)
                # self.u_cond_txt_trt = torch.zeros([1, 77, 768], dtype=torch.float32, device=self.device)

    def get_cuda_graph(self):
        with open("controlnet_one_loop.pkl", "rb") as f:
            inputs = pickle.load(f)

        for i, inp in enumerate(inputs):
            memcopy_device_to_device(self.bindings[i], inp.cuda().data_ptr(), self.inputs[i].nbytes)
            # cuda_inputs.append(inp.cuda())
            self.context.set_tensor_address(self.input_name[i], self.bindings[i])

        self.context.set_tensor_address(self.output_name[0], self.outputs[0].device)

        self.context.execute_async_v3(self.stream) # important
        cudart.cudaStreamBeginCapture(self.stream, cudart.cudaStreamCaptureMode(0))
        self.context.execute_async_v3(self.stream)

        err, graph = cudart.cudaStreamEndCapture(self.stream)
        err, graph_exec = cudart.cudaGraphInstantiate(graph, 0)
        print(err)
        return graph_exec

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.to(torch.float32).to(self.model.device) #.clone().detach()

        # self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        # self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        # self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        # self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))


    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               **kwargs
               ):

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        # print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule
                                                    )
        return samples #, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None):
        device = self.model.betas.device
        # b = shape[0]


        img = torch.randn(shape, device=device)


        total_steps = self.timesteps #timesteps if ddim_use_original_steps else timesteps.shape[0]

        #steps = torch.from_numpy(np.ascontiguousarray(time_range)).to(device=device, dtype=torch.long).reshape([len(time_range), -1])

        ############ RUN ONLY ONE TIME NOT 20 TIMES !!!
        c_cond_txt =  torch.cat([cond['c_crossattn']], 1).to(torch.int32) #cond['c_crossattn'].to(torch.int32) #
        c_hint = torch.cat([cond['c_concat']], 1) #cond['c_concat'] #

        u_cond_txt = torch.cat([unconditional_conditioning['c_crossattn']], 1).to(torch.int32) #unconditional_conditioning['c_crossattn'].to(torch.int32) #

        ############
        # with open("hint_block.pkl", "wb+") as f:
        #     pickle.dump([c_hint, c_cond_txt, u_cond_txt], f)
            # raise
        if self.input_block_use_trt:
            self.bindings_hint[0] = int(c_hint.data_ptr())
            self.bindings_hint[1] = int(c_cond_txt.data_ptr())
            self.bindings_hint[2] = int(u_cond_txt.data_ptr())
            self.context_hint.execute_async_v2(
                bindings=self.bindings_hint,
                stream_handle=self.stream_hint)
            cudart.cudaStreamSynchronize(self.stream_hint)
            memcopy_device_to_device(self.c_hint_trt.data_ptr(), self.outputs_hint[0].device, self.outputs_hint[0].nbytes)
            memcopy_device_to_device(self.conds_all_trt.data_ptr(), self.outputs_hint[1].device, self.outputs_hint[1].nbytes)

            c_hint = self.c_hint_trt.contiguous()
            conds_all = self.conds_all_trt.contiguous()
        else:
            # c_hint, c_cond_txt, u_cond_txt = self.control_input_block(c_hint, c_cond_txt, u_cond_txt)
            c_hint, conds_all = self.control_input_block(c_hint, c_cond_txt, u_cond_txt)

        for i in range(total_steps - 1):

            index = total_steps - i - 1
            # t = steps[i].item()
            ts = self.model.control_model.step_dict[i] #steps[i].item()
            ts_df = self.model.model.diffusion_model.step_dict[i] #steps[i].item()

            if self.control_net_use_trt:
                memcopy_device_to_device(self.bindings[0], img.data_ptr(), self.inputs[0].nbytes)
                memcopy_device_to_device(self.bindings[1], ts.data_ptr(), self.inputs[1].nbytes)
                memcopy_device_to_device(self.bindings[2], ts_df.data_ptr(), self.inputs[2].nbytes)
                if i == 0:
                    memcopy_device_to_device(self.bindings[3], conds_all.data_ptr(), self.inputs[3].nbytes)
                    memcopy_device_to_device(self.bindings[4], c_hint.data_ptr(), self.inputs[4].nbytes)
                memcopy_device_to_device(self.bindings[5], self.alphas[index].data_ptr(), self.inputs[5].nbytes)
                memcopy_device_to_device(self.bindings[6], self.alphas_prev[index].data_ptr(), self.inputs[6].nbytes)
                memcopy_device_to_device(self.bindings[7], self.sqrt_one_minus_alphas[index].data_ptr(), self.inputs[7].nbytes)

                cudart.cudaGraphLaunch(self.graph_exec, self.stream)
                cudart.cudaStreamSynchronize(self.stream)
                memcopy_device_to_device(self.out_tensor.data_ptr(), self.outputs[0].device, self.outputs[0].nbytes)
                img = self.out_tensor  

            else:
                img = self.full_model(img, ts, ts_df, conds_all, c_hint, self.alphas[index], self.alphas_prev[index], self.sqrt_one_minus_alphas[index])#, sigmas[index])


        return img 
