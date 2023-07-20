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

class Control_Diff_VAE(torch.nn.Module):
    def __init__(self, control_model):
        super().__init__()
        self.control_model = control_model
        # self.vae_model = vae_model

        self.device = torch.device("cuda")
        # self.noise = torch.randn()
    
    def forward(self, img, ts, ts_df, 
                c_cond_txt, c_hint, u_cond_txt, u_hint, 
                a_t, a_prev, sqrt_one_minus_at):


        model_t = self.control_model(img, ts, ts_df, c_cond_txt, c_hint)
        model_uncond = self.control_model(img, ts, ts_df, u_cond_txt, u_hint)
        model_output = model_uncond + 9 * (model_t - model_uncond)
        #########################
        e_t = model_output

        pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()

        dir_xt = (1. - a_prev).sqrt() * e_t

        img = a_prev.sqrt() * pred_x0 + dir_xt

        # img = 1. / 0.18215 * img
        # img = self.vae_model(img)

        return img #, intermediates

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

        self.device = torch.device("cuda")
        self.model.control_model.init_steps()
        self.model.model.diffusion_model.init_steps()

        self.make_schedule(ddim_num_steps=20, ddim_eta=0.0, verbose=False)
        self.full_model = Control_Diff_VAE(self.model)

        if not Path("controlnet_one_loop_fp16.engine").exists(): self.control_net_use_trt = False
        else: self.control_net_use_trt = True
        if self.control_net_use_trt:

            logger = trt.Logger(trt.Logger.INFO)
            trt.init_libnvinfer_plugins(logger, '')
            with open("controlnet_one_loop_fp16.engine", 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
                self.context = model.create_execution_context()
                self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(model)
                # self.inputs_stage2, self.outputs_stage2, self.bindings_stage2, self.stream_stage2 = allocate_buffers(model)
                self.out_tensor, self.out_tensor2 = torch.zeros([1, 4, 32, 48], dtype=torch.float32, device=self.device), torch.zeros([1, 4, 32, 48], dtype=torch.float32, device=self.device)


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
        # sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
        #     (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
        #                 1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        # self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

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
        # if conditioning is not None:
        #     if isinstance(conditioning, dict):
        #         ctmp = conditioning[list(conditioning.keys())[0]]
        #         while isinstance(ctmp, list): ctmp = ctmp[0]
        #         cbs = ctmp.shape[0]
        #         if cbs != batch_size:
        #             print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

        #     elif isinstance(conditioning, list):
        #         for ctmp in conditioning:
        #             if ctmp.shape[0] != batch_size:
        #                 print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

        #     else:
        #         if conditioning.shape[0] != batch_size:
        #             print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose) # 0.7 ms -> 0.4 ms

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
        ########## Create tensor only once not 20 times !!!!!   
        alphas =  torch.reshape(self.ddim_alphas, [len(self.ddim_alphas), 1, 1, 1, 1]).to(torch.float) #[torch.full((1, 1, 1, 1), alphas[idx], device=self.device) for idx in range(len(alphas))]
        alphas_prev = torch.from_numpy(self.ddim_alphas_prev).to(self.device, torch.float).reshape([len(self.ddim_alphas_prev), 1, 1, 1, 1]) #[torch.full((1, 1, 1, 1), alphas_prev[idx], device=self.device) for idx in range(len(alphas_prev))]
        sqrt_one_minus_alphas = torch.reshape(self.ddim_sqrt_one_minus_alphas, [len(self.ddim_sqrt_one_minus_alphas), 1, 1, 1, 1]).to(torch.float) #[torch.full((1, 1, 1, 1), sqrt_one_minus_alphas[idx], device=self.device) for idx in range(len(sqrt_one_minus_alphas))]
        #sigmas = torch.reshape(self.ddim_sigmas, [len(self.ddim_sigmas), 1, 1, 1, 1]).to(torch.float) #[torch.full((1, 1, 1, 1), sigmas[idx], device=self.device) for idx in range(len(sigmas))]
        ##########

        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        #rand_noise = torch.randn(img.shape, device=device)
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        #intermediates = {'x_inter': [img], 'pred_x0': [img]}
        #time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]

        #steps = torch.from_numpy(np.ascontiguousarray(time_range)).to(device=device, dtype=torch.long).reshape([len(time_range), -1])

        ############ RUN ONLY ONE TIME NOT 20 TIMES !!!
        c_cond_txt = torch.cat(cond['c_crossattn'], 1)
        c_hint = torch.cat(cond['c_concat'], 1)
        # conds = [c_cond_txt, c_hint]

        u_cond_txt = torch.cat(unconditional_conditioning['c_crossattn'], 1)
        u_hint = torch.cat(unconditional_conditioning['c_concat'], 1)    
        # u_conds = [u_cond_txt, u_hint]
        ############

        for i in range(total_steps - 1):

            index = total_steps - i - 1
            # t = steps[i].item()
            ts = self.model.control_model.step_dict[i] #steps[i].item()
            ts_df = self.model.model.diffusion_model.step_dict[i] #steps[i].item()
            # ts_all = [t, ts, ts_df]

            if self.control_net_use_trt:

                self.bindings[0] = int(img.data_ptr())
                self.bindings[1] = int(ts.data_ptr())
                self.bindings[2] = int(ts_df.data_ptr())
                self.bindings[3] = int(c_cond_txt.data_ptr())
                self.bindings[4] = int(c_hint.data_ptr())
                self.bindings[5] = int(u_cond_txt.data_ptr())
                self.bindings[6] = int(u_hint.data_ptr())
                self.bindings[7] = int(alphas[index].data_ptr())
                self.bindings[8] = int(alphas_prev[index].data_ptr())
                self.bindings[9] = int(sqrt_one_minus_alphas[index].data_ptr())             
                self.context.execute_async_v2(
                    bindings=self.bindings,
                    stream_handle=self.stream)
                cudart.cudaStreamSynchronize(self.stream)
                
                memcopy_device_to_device(self.out_tensor.data_ptr(), self.outputs[0].device, self.outputs[0].nbytes)

                img = self.out_tensor   
            else:
                # with open("controlnet_one_loop.pkl", "wb+") as f:
                #     pickle.dump([img, ts, ts_df, c_cond_txt, c_hint, u_cond_txt, u_hint, alphas[index], alphas_prev[index], sqrt_one_minus_alphas[index]], f)
                # torch.onnx.export(self.full_model, (img, ts, ts_df, c_cond_txt, c_hint, u_cond_txt, u_hint, alphas[index], alphas_prev[index], sqrt_one_minus_alphas[index], sigmas[index]), "./onnxs/controlnet_one_loop.onnx", opset_version=17, do_constant_folding=True)
                # print("end")
                # input()
                img = self.full_model(img, ts, ts_df, c_cond_txt, c_hint, u_cond_txt, u_hint, alphas[index], alphas_prev[index], sqrt_one_minus_alphas[index])#, sigmas[index])
            #########################


            ################ orig func call
                # outs = self.p_sample_ddim(img, conds, ts_all, index=index, use_original_steps=ddim_use_original_steps,
                #                           quantize_denoised=quantize_denoised, temperature=temperature,
                #                           noise_dropout=noise_dropout, score_corrector=score_corrector,
                #                           corrector_kwargs=corrector_kwargs,
                #                           unconditional_guidance_scale=unconditional_guidance_scale,
                #                           unconditional_conditioning=u_conds,
                #                           dynamic_threshold=dynamic_threshold)
                # img, pred_x0 = outs
            ################ orig func call

        return img #, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        # b, *_, device = *x.shape, x.device
        # t_orig, 
        ts, ts_df = t
        # cond_txt = c[0]
        # hint = c[1]
        # with open("controlnet_full.pkl", "wb+") as f:
        #     pickle.dump([x, ts, ts_df, cond_txt, hint], f)
        # raise
        # torch.onnx.export(self.model, (x, t, cond_txt, hint), "./onnxs/controlnet_full.onnx", opset_version=17, do_constant_folding=True)
        # raise
        ###########################
        
        if self.control_net_use_trt:
            cond_txt = c[0] #torch.cat(c['c_crossattn'], 1)
            hint = c[1] #torch.cat(c['c_concat'], 1)

            self.bindings[0] = int(x.data_ptr())
            self.bindings[1] = int(ts.data_ptr())
            self.bindings[2] = int(ts_df.data_ptr())
            self.bindings[3] = int(cond_txt.data_ptr())
            self.bindings[4] = int(hint.data_ptr())
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream)
            cudart.cudaStreamSynchronize(self.stream)
            
            memcopy_device_to_device(self.out_tensor.data_ptr(), self.outputs[0].device, self.outputs[0].nbytes)

            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                model_output = self.out_tensor

            else:
                model_t = self.out_tensor

                cond_txt = unconditional_conditioning[0] #torch.cat(unconditional_conditioning['c_crossattn'], 1)
                hint = unconditional_conditioning[1] #torch.cat(unconditional_conditioning['c_concat'], 1)      

                self.bindings[0] = int(x.data_ptr())
                self.bindings[1] = int(ts.data_ptr())
                self.bindings[2] = int(ts_df.data_ptr())
                self.bindings[3] = int(cond_txt.data_ptr())
                self.bindings[4] = int(hint.data_ptr())
                self.context.execute_async_v2(
                    bindings=self.bindings,
                    stream_handle=self.stream)
                cudart.cudaStreamSynchronize(self.stream)

                memcopy_device_to_device(self.out_tensor2.data_ptr(), self.outputs[0].device, self.outputs[0].nbytes)
                model_uncond = self.out_tensor2
                # model_uncond = self.model(x, t, cond_txt, hint)
                model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        #########################

        else:
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                cond_txt = c[0] #torch.cat(c['c_crossattn'], 1)
                hint = c[1] #torch.cat(c['c_concat'], 1)
                model_output = self.model(x, ts, ts_df, cond_txt, hint)
            else:
                cond_txt = c[0] #torch.cat(c['c_crossattn'], 1)
                hint = c[1] #torch.cat(c['c_concat'], 1)
                model_t = self.model(x, ts, ts_df, cond_txt, hint)
                
                cond_txt = unconditional_conditioning[0] # torch.cat(unconditional_conditioning['c_crossattn'], 1)
                hint = unconditional_conditioning[1] #torch.cat(unconditional_conditioning['c_concat'], 1)
                model_uncond = self.model(x, ts, ts_df, cond_txt, hint)
                model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)
        # eps
        # if self.model.parameterization == "v":
            # e_t = self.model.predict_eps_from_z_and_v(x, t_orig, model_output)
        # else:
        e_t = model_output

        # if score_corrector is not None:
        #     assert self.model.parameterization == "eps", 'not implemented'
        #     e_t = score_corrector.modify_score(self.model, e_t, x, t_orig, c, **corrector_kwargs)

        # alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        # alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        # sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        # sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = self.alphas[index] #torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = self.alphas_prev[index] #torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = self.sigmas[index] #torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = self.sqrt_one_minus_alphas[index] #torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        # if self.model.parameterization != "v":
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # else:
        #     pred_x0 = self.model.predict_start_from_z_and_v(x, t_orig, model_output)

        # if quantize_denoised:
        #     pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        # if dynamic_threshold is not None:
        #     raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * self.noise #noise_like(x.shape, device, repeat_noise) * temperature
        # if noise_dropout > 0.:
        #     noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        num_reference_steps = timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), timesteps[i], device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec
