import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from common import allocate_buffers, memcpy_device_to_host, memcpy_host_to_device, memcopy_device_to_device
import tensorrt as trt
from cuda import cuda, cudart
import numpy as np
from pathlib import Path
import pickle
import time
import torch.nn.functional as F
from ldm.modules.diffusionmodules.openaimodel import ResBlock, Downsample
class ControlledUnetModel(UNetModel):
    def init_steps(self):
        step_keys = [951, 901, 851, 801, 751, 701, 651, 601, 551, 501, 451, 401, 351, 301, 251, 201, 151, 101, 51, 1]
        step_values = [timestep_embedding(torch.tensor([key], dtype=torch.long, device=torch.device("cuda"), requires_grad=False), self.model_channels, repeat_only=False) for key in step_keys]
        with torch.no_grad():
            step_embed = [self.time_embed(val) for val in step_values] # 20
            # self.step_dict = step_embed 
            res_modules = [] # emb layers
            emb_results = []

            for i, module in enumerate(self.modules()):
                if isinstance(module, ResBlock):
                    res_modules.append(module.emb_layers)
            
            emb_size = torch.zeros([len(step_keys), len(res_modules)]).cuda() # 20, 24
            for key_idx in range(len(step_keys)):
                temp_res = []
                for i, module in enumerate(res_modules):
                    temp_res.append(module(step_embed[key_idx]).reshape(1, -1, 1, 1))
                emb_results.append(torch.cat(temp_res, dim=1).cuda().contiguous())

            self.step_dict = emb_results

    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, memory_x=None, x_in=None, **kwargs):
        hs = []
        j = len(control) - 1

        emb_indexs = [320, 320, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
        emb_idx = 0 # data_start_idx
        emb_start = 0 # list_idx

        with torch.no_grad():

            emb = timesteps #self.step_dict[timesteps.item()]
            h = x
            # for i, module in enumerate(self.input_blocks):
            #     if memory_x is None:
            #         if i == 1:
            #             h, return_memory_x, x_in = module(h, emb, context)
            #         else:
            #             h, _, _ = module(h, emb, context)
            #         hs.append(h)

            #     else:
            #         if i == 1: 
            #             h = x_in
            #             h, return_memory_x, x_in = module(h, emb, context, memory_x)
            #             memory_x = None
            #         elif i > 1:
            #             h, _, _ = module(h, emb, context)
            #         hs.append(h)
            #####################


            for i, module in enumerate(self.input_blocks):
                emb = timesteps[:, emb_idx:emb_idx + emb_indexs[emb_start], ...]
                if i == 0:
                    if memory_x is None: h, _, _ = module(h, emb, context)
                elif i == 1:
                    if memory_x is not None: h = x_in
                    h, return_memory_x, x_in = module(h, emb, context, memory_x)
                    emb_idx += emb_indexs[emb_start]
                    emb_start += 1
                    memory_x = None
                else:
                    h, _, _ = module(h, emb, context)
                    if i != 3 and i != 6 and i != 9:
                        emb_idx += emb_indexs[emb_start]
                        emb_start += 1
                hs.append(h)


            h0 = hs[0]

            emb = timesteps[:, emb_idx:emb_idx + emb_indexs[emb_start], ...]
            h = self.middle_block[0](h, emb)
            emb_idx += emb_indexs[emb_start]
            emb_start += 1
            
            h, _, _ = self.middle_block[1](h, context)
            emb = timesteps[:, emb_idx:emb_idx + emb_indexs[emb_start], ...]
            h = self.middle_block[2](h, emb)
            emb_idx += emb_indexs[emb_start]
            emb_start += 1

            # h, _, _ = self.middle_block(h, emb, context)
        if control is not None:
            h += control[j]#.pop()
            j -= 1
        k = len(hs) - 1
        for i, module in enumerate(self.output_blocks):
            emb = timesteps[:, emb_idx:emb_idx + emb_indexs[emb_start], ...]
            # if only_mid_control or control is None:
            #     h = torch.cat([h, hs.pop()], dim=1)
            # else:
 
            h = torch.cat([h, hs[k] + control[j]], dim=1)
            k -= 1
            j -= 1
            h, _, _ = module(h, emb, context)
            emb_idx += emb_indexs[emb_start]
            emb_start += 1

        return self.out(h), return_memory_x, x_in, h0


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def init_steps(self):
        step_keys = [951, 901, 851, 801, 751, 701, 651, 601, 551, 501, 451, 401, 351, 301, 251, 201, 151, 101, 51, 1]
        step_values = [timestep_embedding(torch.tensor([key], dtype=torch.long, device=torch.device("cuda"), requires_grad=False), self.model_channels, repeat_only=False) for key in step_keys]
        with torch.no_grad():
            step_embed = [self.time_embed(val) for val in step_values]
            # self.step_dict = step_embed #dict(zip(step_keys, step_embed))
            res_modules = [] # emb layers
            emb_results = []

            for i, module in enumerate(self.modules()):
                if isinstance(module, ResBlock):
                    res_modules.append(module.emb_layers)
            # print(len(res_modules))
            # raise
            emb_size = torch.zeros([len(step_keys), len(res_modules)]).cuda() # 20, 24
            for key_idx in range(len(step_keys)):
                temp_res = []
                for i, module in enumerate(res_modules):
                    temp_res.append(module(step_embed[key_idx]).reshape(1, -1, 1, 1))
                emb_results.append(torch.cat(temp_res, dim=1).cuda().contiguous())
            self.step_dict = emb_results


    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, memory_x=None, x_in=None, **kwargs): # context 1, 77, 768
        # t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        # emb = self.time_embed(t_emb)
        emb_indexs = [320, 320, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280]
        emb_idx = 0 # data_start_idx
        emb_start = 0 # list_idx
        # emb = timesteps #self.step_dict[timesteps.item()]

        guided_hint = hint#self.input_hint_block(hint, None, None)

        outs = []

        h = x#.type(self.dtype)
        for i, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
            emb = timesteps[:, emb_idx:emb_idx + emb_indexs[emb_start], ...]
            if i == 0: # conv, no emb used
                if memory_x is not None:
                    outs.append(h)
                    guided_hint = None
                    continue
                else:
                    h, _, _ = module(h, emb, context)
                    h += guided_hint
                    guided_hint = None
            
            elif i == 1:
                if memory_x is not None: h = x_in
                h, return_memory_x, x_in = module(h, emb, context, memory_x)
                emb_idx += emb_indexs[emb_start]
                emb_start += 1

                memory_x = None
            else:
                h, _, _ = module(h, emb, context)
                
                if i != 3 and i != 6 and i != 9:
                    emb_idx += emb_indexs[emb_start]
                    emb_start += 1
            
            outs.append(zero_conv(h, emb, context)[0])

        emb = timesteps[:, emb_idx:emb_idx + emb_indexs[emb_start], ...]
        h = self.middle_block[0](h, emb)
        emb_idx += emb_indexs[emb_start]
        emb_start += 1

        h, _, _ = self.middle_block[1](h, context)
        emb = timesteps[:, emb_idx:emb_idx + emb_indexs[emb_start], ...]
        h = self.middle_block[2](h, emb)
        # h, _, _ = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context)[0])

        return outs, return_memory_x, x_in


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

        # if not Path("controlnet_fp16.engine").exists(): self.control_net_use_trt = False
        # else: self.control_net_use_trt = False
        # if self.control_net_use_trt:

        #     device = torch.device("cuda")
        #     logger = trt.Logger(trt.Logger.INFO)
        #     trt.init_libnvinfer_plugins(logger, '')
        #     with open("controlnet_fp16.engine", 'rb') as f, trt.Runtime(logger) as runtime:
        #         model = runtime.deserialize_cuda_engine(f.read())
        #         self.context = model.create_execution_context()
        #         self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(model)
        #         self.shape_list = [[[1, 320, 32, 48], 3], [[1, 320, 16, 24], 1], [[1, 640, 16, 24], 2], [[1, 640, 8, 12], 1], [[1, 1280, 8, 12], 2], [[1, 1280, 4, 6], 4]]

        #         self.mid_tensors = [torch.zeros(box[0], device=device, dtype=torch.float32) for box in self.shape_list for _ in range(box[1]) ]

            
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], c_concat=[control])

    def forward(self, x_noisy, ts, ts_df, cond_txt, u_cond_txt, hint):#, *args, **kwargs

        control, return_memory_x, x_in = self.control_model(x=x_noisy, hint=hint, timesteps=ts, context=cond_txt)
        eps, return_memory_x_df, x_in_df, h0 = self.model.diffusion_model(x=x_noisy, timesteps=ts_df, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        control, _, _ = self.control_model(x=control[0], hint=hint, timesteps=ts, context=u_cond_txt, memory_x=return_memory_x, x_in=x_in)
        eps_uncond, _, _, _ = self.model.diffusion_model(x=h0, timesteps=ts_df, context=u_cond_txt, control=control, only_mid_control=self.only_mid_control, memory_x=return_memory_x_df, x_in=x_in_df)

        return eps, eps_uncond

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
