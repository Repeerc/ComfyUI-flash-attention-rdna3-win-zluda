import logging
import math
from einops import rearrange
from matplotlib.style import available
from torch import Tensor, nn
import numpy as np
from comfy import model_management
import torch

import comfy.ldm.flux.layers


torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

torch.backends.cudnn.enabled = False

import comfy.ldm.flux 
import comfy.ldm.modules.attention
import comfy.ldm.modules.diffusionmodules.model

from comfy.ldm.modules.diffusionmodules.model import (
    Normalize,
    ResnetBlock,
    Upsample,
    nonlinearity,
)
from comfy.ldm.modules.attention import (
    attention_sub_quad,
    attention_pytorch,
    attention_split,
    attention_xformers,
)
from comfy.ldm.modules.diffusionmodules.model import (
    normal_attention,
    pytorch_attention,
    xformers_attention,
)

import comfy.ops

ops = comfy.ops.disable_weight_init

import torch
import os

os.add_dll_directory(os.path.join(os.environ["HIP_PATH"], "bin"))

HIP_VER = 57

if '6.1' in os.environ["HIP_PATH"]:
    from .hip61 import flash_attn_wmma
    from .hip61 import ck_fttn_pyb
    HIP_VER = 61
    print("Detected HIP Version:6.1")
else:
    from .hip57 import flash_attn_wmma
    from .hip57 import ck_fttn_pyb
    HIP_VER = 57
    print("Detected HIP Version:5.7")
    


select_attention_algorithm = None
select_attention_vae_algorithm = None


def rocm_fttn(
    q_in, k_in, v_in, heads, mask=None, attn_precision=None, skip_reshape=False
):  # (q, k, v, causal=False, sm_scale=None):
    
    if skip_reshape: 
        b, _, _, dim_head = q_in.shape
        q = q_in
        k = k_in
        v = v_in
    else: 
        b, _, dim_head = q_in.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q_in, k_in, v_in),
        )    
        del q_in, k_in, v_in
    
    if dim_head <= 64 and HIP_VER == 61:
       q, k, v = map(lambda t: t.transpose(1, 2),(q, k, v)) 
       ret = ck_fttn_pyb.fwd(
           q, k, v, None, 0, q.shape[-1] ** -0.5, False, False, None
       )[0]
       return ret.reshape(b, -1, heads * dim_head)
    
    # print(q.shape, k.shape, v.shape) # [1, 24, 6400, 128] flux
    
    dtype = q.dtype
    q, k, v = map(
        lambda t: t.to(torch.float16),
        (q, k, v),
    ) 

    Br = 64
    Bc = 256
    if dim_head >= 272:
        Br = 32
        Bc = 128
       

    ret = flash_attn_wmma.forward(q, k, v, Br, Bc, False, dim_head**-0.5)[0]
    #o = rearrange(o, "b h n d -> b n (h d)", h=heads) 
    o = (
        o.transpose(1, 2).reshape(b, -1, heads * dim_head)
    )
    
    return o.to(dtype)


def rocm_fttn_vae(q, k, v, *args, **kwargs):

    B, C, H, W = q.shape
    
    
    
    q, k, v = map(
        lambda t: t.view(B, 1, C, -1).transpose(2, 3),
        (q, k, v),
    )
    
    d_q = q.shape[-1]     

    dtype = q.dtype
    if dtype != torch.float16:
        q = q.clip(-40000.0, 40000.0)
        k = k.clip(-40000.0, 40000.0)
        v = v.clip(-40000.0, 40000.0)
    
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
    # print("VAE",q.shape, k.shape, v.shape)
    
    Br = 64
    Bc = 256
    if d_q >= 272:
        Br = 32
        Bc = 128

    o = flash_attn_wmma.forward(q, k, v, Br, Bc, False, d_q**-0.5)[0]

    o = o.transpose(2, 3).reshape(B, C, H, W)
    return o.to(dtype)


class AttnBlock_hijack(nn.Module):
    def __init__(self, in_channels):
        global select_attention_vae_algorithm
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = ops.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = ops.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = ops.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = ops.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

        self.optimized_attention = select_attention_vae_algorithm

    def forward(self, x):
        global select_attention_vae_algorithm
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        if self.optimized_attention != select_attention_vae_algorithm:
            self.optimized_attention = select_attention_vae_algorithm

        h_ = self.optimized_attention(q, k, v)

        h_ = self.proj_out(h_)

        return x + h_


class Decoder_hijack(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        conv_out_op=ops.Conv2d,
        resnet_op=ResnetBlock,
        attn_op=AttnBlock_hijack,
        **ignorekwargs
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        logging.debug(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = ops.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = resnet_op(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = attn_op(block_in)
        self.mid.block_2 = resnet_op(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    resnet_op(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(attn_op(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_out_op(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z, **kwargs):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, **kwargs)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, **kwargs)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h, **kwargs)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    print("  AttnBlock_hijack")
    return AttnBlock_hijack(in_channels)

# comfy.ldm.modules.diffusionmodules.model.make_attn = make_attn
# comfy.ldm.modules.diffusionmodules.model.AttnBlock = AttnBlock_hijack
# comfy.ldm.modules.diffusionmodules.model.Decoder = Decoder_hijack
# comfy.ldm.modules.attention.optimized_attention = select_attention_algorithm
# comfy.ldm.flux.math.optimized_attention = select_attention_vae_algorithm
 
# setattr(comfy.ldm.modules.attention,"optimized_attention",select_attention_algorithm) 
# setattr(comfy.ldm.modules.diffusionmodules.model, "make_attn", make_attn)
# setattr(comfy.ldm.modules.diffusionmodules.model, "AttnBlock", AttnBlock_hijack)
# setattr(comfy.ldm.modules.diffusionmodules.model, "Decoder", Decoder_hijack) 


class AttnOptSelector:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        available_attns = []
        available_attns.append("Flash-Attention-v2")
        # if model_management.xformers_enabled():
        available_attns.append("xformers")
        available_attns.append("pytorch")
        available_attns.append("split")
        available_attns.append("sub-quad")

        return {
            "required": {
                "sampling_attention": (available_attns,),
                "vae_attention": (available_attns,),
                "Model": ("MODEL", )
            },
        }

    RETURN_TYPES = ("MODEL", )

    FUNCTION = "test"
    OUTPUT_NODE = True

    CATEGORY = "_for_testing"

    def test(self, sampling_attention, vae_attention, Model):
        global select_attention_algorithm, select_attention_vae_algorithm
        print("  Select optimized attention:", sampling_attention, vae_attention)
        if sampling_attention == "xformers":
            select_attention_algorithm = attention_xformers
        elif sampling_attention == "pytorch":
            select_attention_algorithm = attention_pytorch
        elif sampling_attention == "split":
            select_attention_algorithm = attention_split
        elif sampling_attention == "sub-quad":
            select_attention_algorithm = attention_sub_quad
        elif sampling_attention == "Flash-Attention-v2":
            select_attention_algorithm = rocm_fttn

        if vae_attention == "xformers":
            select_attention_vae_algorithm = xformers_attention
        elif vae_attention == "pytorch":
            select_attention_vae_algorithm = pytorch_attention
        elif vae_attention == "split":
            select_attention_vae_algorithm = normal_attention
        elif vae_attention == "sub-quad":
            select_attention_vae_algorithm = normal_attention
        elif vae_attention == "Flash-Attention-v2":
            select_attention_vae_algorithm = rocm_fttn_vae

        comfy.ldm.modules.diffusionmodules.model.make_attn = make_attn
        comfy.ldm.modules.diffusionmodules.model.AttnBlock = AttnBlock_hijack
        comfy.ldm.modules.diffusionmodules.model.Decoder = Decoder_hijack
        comfy.ldm.modules.attention.optimized_attention = select_attention_algorithm
        
        comfy.ldm.flux.math.optimized_attention = select_attention_algorithm 
        setattr(comfy.ldm.flux.math, "optimized_attention", select_attention_algorithm)
        
        setattr(comfy.ldm.modules.attention,"optimized_attention",select_attention_algorithm) 
        setattr(comfy.ldm.modules.diffusionmodules.model, "make_attn", make_attn)
        setattr(comfy.ldm.modules.diffusionmodules.model, "AttnBlock", AttnBlock_hijack)
        setattr(comfy.ldm.modules.diffusionmodules.model, "Decoder", Decoder_hijack)
        
        return (Model, sampling_attention + vae_attention)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {"AttnOptSelector": AttnOptSelector}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"AttnOptSelector": "optimized attention selector"}
