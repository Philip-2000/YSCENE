import math
from random import random
from functools import partial
from collections import namedtuple
from tkinter.messagebox import NO
from tkinter.tix import Tree

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
        
class ExtResidual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, y, *args, **kwargs):
        return self.fn(x, y, *args, **kwargs) + x

def Upsample(dim, dim_out = None, pool=False):

    if pool:
        return nn.Sequential(
            #nn.Upsample(scale_factor = 2, mode = 'nearest'),
        
            nn.Conv1d(dim, default(dim_out, dim), 1)
        )
    else:
        return nn.Identity()

def Downsample(dim, dim_out = None, pool=False):
    
    if pool:
        return nn.Sequential(
            #return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)
        
            nn.Conv1d(dim, default(dim_out, dim), 1) 
        )
    else:
        return nn.Identity() 

class WeightStandardizedConv2d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class ExtPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, y):
        x = self.norm(x)
        return self.fn(x, y)
        
class PreNormCross(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, context):
        x = self.norm(x)
        return self.fn(x, context)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 1, padding = 0) # 3-->1
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            if len(time_emb.shape) ==2:
                time_emb = rearrange(time_emb, 'b c -> b c 1')
            else:
                # BxNxC --> BxCxN
                time_emb = torch.permute(time_emb, (0, 2, 1))
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class ExtLinearAttention(nn.Module):
    def __init__(self, dim, ind_dim=5, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.ind_dim = ind_dim
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.ind_qk = nn.Conv1d(ind_dim, heads, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x, indicator):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)
        qk = self.ind_qk(indicator.reshape((b, self.ind_dim, n*n))).reshape((b, self.heads, n, n))
        
        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim + qk
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        
        out = rearrange(out, 'b h n d -> b (h d) n', h = self.heads)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

class ExtAttention(nn.Module):
    def __init__(self, dim, ind_dim=5, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.ind_dim = ind_dim
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.ind_qk = nn.Conv1d(ind_dim, heads, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x, indicator):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)
        qk = self.ind_qk(indicator.reshape((b, self.ind_dim, n*n))).reshape((b, self.heads, n, n))

        #q = q.softmax(dim = -2)
        #k = k.softmax(dim = -1)

        q = q * self.scale        

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = (sim + qk).softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

class RPEAttention(nn.Module):
    def __init__(self, dim, ind_dim=5, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.ind_dim = ind_dim
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.ind_qk = nn.Conv1d(ind_dim, hidden_dim * 2, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x, indicator):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)
        ind_qk = self.ind_qk(indicator.reshape((b, self.ind_dim, n*n))).chunk(2, dim = 1)
        ind_q, ind_k = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), ind_qk)
        #.reshape((b, self.heads, n, n))

        qq = q.reshape((b,self.heads,-1,n,1)) + ind_q.reshape((b,self.heads,-1,n,n))
        qq = qq * self.scale        
        kk = k.reshape((b,self.heads,-1,1,n)) + ind_k.reshape((b,self.heads,-1,n,n))

        sim = einsum('b h d i j, b h d i j -> b h i j', qq, kk)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, dimx, dimx_out, dimy, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkvx = nn.Conv1d(dimx, hidden_dim * 3, 1, bias = False)
        self.to_outx = nn.Conv1d(hidden_dim, dimx_out, 1)
        self.to_qkvy = nn.Conv1d(dimy, hidden_dim * 3, 1, bias = False)

    def forward(self, x, y):
        bx, cx, nx = x.shape
        qkvx = self.to_qkvx(x).chunk(3, dim = 1)
        qx, kx, vx = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkvx)

        by, cy, ny = y.shape
        qkvy = self.to_qkvy(x).chunk(3, dim = 1)
        qy, ky, vy = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkvy)

        qx = qx * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', qx, ky)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, vy)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_outx(out)

class Unet1DYL(nn.Module):
    def __init__(
        self,
        dim=256, #
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        seperate_all=False,
        merge_bbox = False,
        objectness_dim=1,
        class_dim=21,
        use_weight=False,
        weight_dim=21,
        translation_dim = 3,
        size_dim = 3,
        angle_dim = 1,
        objfeat_dim=0,
        context_dim = 256,
        instanclass_dim = 0,
        modulate_time_context_instanclass = False,
        text_condition=False,
        text_dim=256,
        ind_dim=0,
        rel_wall_dim=0,
        rel_windoor_dim=0,
        rpe=False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        self.seperate_all = seperate_all
        self.objectness_dim = objectness_dim
        self.class_dim = class_dim
        self.use_weight = use_weight
        self.weight_dim = weight_dim
        self.translation_dim = translation_dim
        self.size_dim = size_dim
        self.angle_dim = angle_dim
        self.bbox_dim  = translation_dim + size_dim + angle_dim
        self.objfeat_dim = objfeat_dim
        self.modulate_time_context_instanclass =  modulate_time_context_instanclass
        self.text_condition = text_condition
        self.text_dim = text_dim
        self.ind_dim = ind_dim
        self.rel_wall_dim = rel_wall_dim
        self.rel_windoor_dim = rel_windoor_dim
        self.rpe=rpe

        if self.seperate_all:
            if self.objectness_dim >0:
                self.objectness_embedf = Unet1DYL._encoder_mlp(dim, self.objectness_dim)

            if self.objfeat_dim  >0:
                self.objfeat_embedf = Unet1DYL._encoder_mlp(dim, self.objfeat_dim)
            
            self.weight_embedf = Unet1DYL._encoder_mlp(dim, self.weight_dim)
            self.class_embedf = Unet1DYL._encoder_mlp(dim, self.class_dim)
            self.bbox_embedf = Unet1DYL._encoder_mlp(dim, self.translation_dim+self.size_dim+self.angle_dim)
            if self.rel_wall_dim > 0:
                self.wall_embedf = Unet1DYL._encoder_mlp(dim, self.rel_wall_dim)
            if self.rel_windoor_dim > 0:
                self.windoor_embedf = Unet1DYL._encoder_mlp(dim, self.rel_windoor_dim)
            
            input_channels = dim 
            print('separate unet1dyl encoder of objectness/class/translation/size/angle')

        else:
            input_channels = channels
            print('unet1dyl encoder of all object properties')

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 1) #nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)
        
        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )


        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)



        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = context_dim + instanclass_dim), 
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                CrossAttention(dim_in, dim_in, text_dim) if self.text_condition else nn.Identity(),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                ExtResidual(ExtPreNorm(dim_in, RPEAttention(dim_in, ind_dim))) if self.rpe else (ExtResidual(ExtPreNorm(dim_in, ExtLinearAttention(dim_in, ind_dim))) if self.ind_dim > 0 else Residual(PreNorm(dim_in, LinearAttention(dim_in))) ),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 1) #3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block0 = block_klass(mid_dim, mid_dim, time_emb_dim = context_dim + instanclass_dim) 
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn_cross = CrossAttention(mid_dim, mid_dim, text_dim) if self.text_condition else nn.Identity()
        self.mid_attn = ExtResidual(ExtPreNorm(mid_dim, RPEAttention(mid_dim, ind_dim))) if self.rpe else (ExtResidual(ExtPreNorm(mid_dim, ExtAttention(mid_dim, ind_dim))) if self.ind_dim > 0 else Residual(PreNorm(mid_dim, Attention(mid_dim))))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1) 

            self.ups.append(nn.ModuleList([
                block_klass(dim_out, dim_in, time_emb_dim = context_dim + instanclass_dim), 
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                CrossAttention(dim_out, dim_out, text_dim) if self.text_condition else nn.Identity(),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                ExtResidual(ExtPreNorm(dim_out, RPEAttention(dim_out, ind_dim))) if self.rpe else (ExtResidual(ExtPreNorm(dim_out, ExtLinearAttention(dim_out, ind_dim))) if self.ind_dim > 0 else Residual(PreNorm(dim_out, LinearAttention(dim_out)))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 1) #3, padding = 1)
            ]))


        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        
        if self.seperate_all:
            if self.objectness_dim >0:
                self.objectness_hidden2output = Unet1DYL._decoder_mlp(dim, self.objectness_dim)

            if self.objfeat_dim >0:
                self.objfeat_hidden2output = Unet1DYL._decoder_mlp(dim, self.objfeat_dim)

            self.class_hidden2output = Unet1DYL._decoder_mlp(dim, self.class_dim)
            self.weight_hidden2output = Unet1DYL._decoder_mlp(dim, self.weight_dim)
            
            self.bbox_hidden2output = Unet1DYL._decoder_mlp(dim, self.translation_dim+self.size_dim+self.angle_dim)
            print('separate unet1dyl decoder of objectness/class/translation/size/angle')

        else:
            self.final_conv = nn.Conv1d(dim, self.out_dim, 1)
            print('unet1d decoderyl of all object properties')
        
        if self.use_weight:
            for name, module in self._modules.items():
                #print(name)
                module.trainable = False
            self.weight_embedf.trainable = True
            self.weight_hidden2output.trainable = True
            #raise NotImplementedError

    @staticmethod
    def _encoder_mlp(hidden_size, input_size):
        mlp_layers = [
                nn.Conv1d(input_size, hidden_size, 1),
                nn.GELU(),
                nn.Conv1d(hidden_size, hidden_size*2, 1),
                nn.GELU(),
                nn.Conv1d(hidden_size*2, hidden_size, 1),
            ]
        return nn.Sequential(*mlp_layers)
    
    @staticmethod
    def _decoder_mlp(hidden_size, output_size):
        mlp_layers = [
            nn.Conv1d(hidden_size, hidden_size*2, 1),
            nn.GELU(),
            nn.Conv1d(hidden_size*2, hidden_size, 1),
            nn.GELU(),
            nn.Conv1d(hidden_size, output_size, 1),
        ]
        return nn.Sequential(*mlp_layers)
    
    def forward(self, x, beta, context=None, context_cross=None, indicator=None): 
        # batchsz = 128 : maxObj = 12 : env_dim = 69         absolutionTensor.dim = -1:
        # (tr_dim=3)+(sz_dim=3)+(ag_dim=2)
        # (  bbox_dim = 8     cosA, sinA ) + (cl_dim = 22) + (ft_dim = 32) + (dis=1)+(ori=2) + (loc=2)+(ori=2)
        # (                          point_dim     =     62              ) + (wall_dim = 3 ) + (windoor_dim=4)
        
        # (B, N, C) --> (B, C, N)
        batch_size, num_points, rela_dim = x.size()
        x = torch.permute(x, (0, 2, 1)).contiguous()
        point_dim = self.bbox_dim+self.class_dim+self.objectness_dim+self.objfeat_dim
        
        if self.seperate_all:
            if self.use_weight:
                self.class_dim = self.weight_dim
                x_class = self.weight_embedf(x[:, self.bbox_dim:self.bbox_dim+self.class_dim, :])
            else:
                x_class = self.class_embedf(x[:, self.bbox_dim:self.bbox_dim+self.class_dim, :])
            
            if self.objectness_dim >0:
                x_object = self.objectness_embedf(x[:, self.bbox_dim+self.class_dim:self.bbox_dim+self.class_dim+self.objectness_dim, :])
            else:
                x_object = 0
            
            if self.objfeat_dim > 0:
                x_objfeat = self.objfeat_embedf(x[:, self.bbox_dim+self.class_dim+self.objectness_dim:point_dim, :])
            else:
                x_objfeat = 0
                
            x_bbox = self.bbox_embedf(x[:, 0:self.bbox_dim, :])
            if self.rel_wall_dim > 0:
                x_wall = self.wall_embedf(x[:,point_dim:point_dim+self.rel_wall_dim, :])
            else:
                x_wall = 0
            
            if self.rel_windoor_dim > 0:
                x_windoor = self.windoor_embedf(x[:,point_dim+self.rel_wall_dim:point_dim+self.rel_wall_dim+self.rel_windoor_dim])
                #x[:, 65:69, :].shape = (self.batchsz=128 : self.windoor=4 : maxObj=12)
                #x_windoor.shape = (self.batchsz=128 : ???=512 : maxObj=12) 

                x_windoor0 = torch.zeros_like(x_windoor)
                dis = (x[:, point_dim+self.rel_wall_dim:point_dim+self.rel_wall_dim+2,:]**2).sum(axis=1).reshape((batch_size,1,-1))
                cond = dis > torch.ones_like(dis)*1000
                #print("dis")
                #print(dis.shape)
                #print("cond")
                #print(cond.shape)
                cond = cond.repeat((1,x_windoor.shape[1],1))
                x_windoor[cond] = x_windoor0[cond]
            else:
                x_windoor = 0

            x = x_class + x_bbox + x_object + x_objfeat + x_wall + x_windoor
        else:
            raise NotImplementedError()
        
        # denosing 
        if context_cross is not None:
            # [B, N, C] --> [B, C, N]
            context_cross = torch.permute(context_cross, (0, 2, 1)).contiguous()
        if indicator is not None:
            # [B, N, N, C] --> [B, C, N, N]
            indicator = torch.permute(indicator, (0, 3, 1, 2)).contiguous()
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(beta) 
        h = []

        # unet-1D
        for block0, block1, attncross, block2, attn, downsample in self.downs:
            x = block0(x, context) 
            x = block1(x, t)
            h.append(x)
            x = attncross(x, context_cross) if self.text_condition else attncross(x)
            x = block2(x, t)
            x = attn(x, indicator) if self.ind_dim > 0 else attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block0(x, context)
        x = self.mid_block1(x, t)
        x = self.mid_attn_cross(x, context_cross) if self.text_condition else self.mid_attn_cross(x)
        x = self.mid_attn(x,indicator) if self.ind_dim > 0 else self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block0, block1, attncross, block2, attn, upsample in self.ups:
            x = block0(x, context) 
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = attncross(x, context_cross) if self.text_condition else attncross(x)
            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x, indicator) if self.ind_dim > 0 else attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x, t)
        
        if self.seperate_all:
            out_bbox  = self.bbox_hidden2output(x)
            if self.use_weight:
                out_class = self.weight_hidden2output(x)
            else:
                out_class = self.class_hidden2output(x)
            out = torch.cat([out_bbox, out_class], dim=1).contiguous()
            if self.objectness_dim >0:
                out_object = self.objectness_hidden2output(x)
                out = torch.cat([out, out_object], dim=1).contiguous()

            if self.objfeat_dim >0:
                out_objfeat = self.objfeat_hidden2output(x)
                out = torch.cat([out, out_objfeat], dim=1).contiguous()
        else:
            out = self.final_conv(x)
        
        # (B, N, C) <-- (B, C, N)
        out = torch.permute(out, (0, 2, 1)).contiguous()
        return out