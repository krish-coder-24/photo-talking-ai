# Reference: 
# 1. DiT https://github.com/facebookresearch/DiT
# 2. TIMM https://github.com/rwightman/pytorch-image-models

import torch
import torch.nn as nn
import numpy as np
import math
import time
from .blocks import FinalLayer
from .blocks import MMDoubleStreamBlock as DiTBlock2
from .blocks import MMSingleStreamBlock as DiTBlock
from .blocks import CrossDiTBlock as DiTBlock3
from .blocks import MMfourStreamBlock as DiTBlock4
# from .positional_embedding import get_1d_sincos_pos_embed
from .posemb_layers import apply_rotary_emb, get_1d_rotary_pos_embed
from .embedders import TimestepEmbedder, MotionEmbedder, AudioEmbedder, ConditionAudioEmbedder, SimpleAudioEmbedder, LabelEmbedder
from einops import rearrange, repeat
audio_embedder_map = {
    "normal": AudioEmbedder,
    "cond": ConditionAudioEmbedder,
    "simple": SimpleAudioEmbedder
}
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
class TalkingHeadDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_dim=265,
        output_dim =265,
        seq_len=80,
        audio_unit_len=5,
        audio_blocks=12,
        audio_dim=768,
        audio_tokens = 1,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        audio_embedder_type="normal",
        audio_cond_dim = 63,
        norm_type="rms_norm",
        qk_norm="rms_norm",
        **kwargs
    ):
        super().__init__()
        
        self.num_emo_class = 8
        self.emo_drop_prob = 0.1

        self.num_heads = num_heads
        self.out_channels = output_dim

        self.motion_embedder = MotionEmbedder(input_dim, hidden_size)
        self.identity_embedder=MotionEmbedder(audio_cond_dim, hidden_size)
        self.time_embedder = TimestepEmbedder(hidden_size)       
        self.audio_embedder = audio_embedder_map['normal'](
            seq_len          = audio_unit_len, 
            blocks           = audio_blocks,
            channels         = audio_dim,
            intermediate_dim = hidden_size,
            output_dim       = hidden_size,
            context_tokens   = audio_tokens, 
            input_len        = seq_len,
            condition_dim    = audio_cond_dim, 
            norm_type        = norm_type, 
            # qk_norm          = qk_norm,
            # n_heads          =num_heads
        )
        self.dim=hidden_size//num_heads
        
        self.emo_embedder = LabelEmbedder(num_classes=self.num_emo_class, hidden_size=hidden_size, dropout_prob=self.emo_drop_prob)
        
        # Will use fixed sin-cos embedding:
        # self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_size), requires_grad=False)
        self.blocks4 = nn.ModuleList([
            DiTBlock4(
                hidden_size, num_heads, 
                mlp_ratio=mlp_ratio, 
                norm_type=norm_type, 
                qk_norm=qk_norm
            ) for _ in range(3)
        ])
        self.blocks2 = nn.ModuleList([
            DiTBlock2(
                hidden_size, num_heads, 
                mlp_ratio=mlp_ratio, 
                norm_type=norm_type, 
                qk_norm=qk_norm
            ) for _ in range(6)
        ])
        self.blocks=nn.ModuleList([
            DiTBlock(
                hidden_size, num_heads, 
                mlp_ratio=mlp_ratio, 
                norm_type=norm_type, 
                qk_norm=qk_norm
            ) for _ in range(12)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels, norm_type=norm_type)
        self.initialize_weights()
        self.bank=[]
    def initialize_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2])
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize input layers nn.Linear
        self.motion_embedder.initialize_weights()
        self.identity_embedder.initialize_weights()
        # Initialize audio embedding 
        self.audio_embedder.initialize_weights()

        # Initialize emotion embedding
        self.emo_embedder.initialize_weights()

        # Initialize timestep embedding MLP
        self.time_embedder.initialize_weights()
        
        # Initialize DiT blocks:
        for block in self.blocks:
            block.initialize_weights()
        for block in self.blocks2:
            block.initialize_weights()
        for block in self.blocks4:
            block.initialize_weights()
        # Initialize output layers:
        # self.final_layer.initialize_weights()
    def cal_sync_loss(self, audio_embedding, mouth_embedding, label):
        if isinstance(label, torch.Tensor):
            gt_d = label.float().view(-1,1).to(audio_embedding.device)
        else:
            gt_d = (torch.ones([audio_embedding.shape[0],1]) * label).float().to(audio_embedding.device) # int
        d = nn.functional.cosine_similarity(audio_embedding, mouth_embedding)
        loss = self.logloss(d.unsqueeze(1), gt_d)
        return loss, d

    def forward(self, motion, times, audio, emo, audio_cond,mask=None):
        """
        Forward pass of Talking Head DiT.
        motion: (B, N, xD) tensor of moton features inputs (head motion, emotion, etc.)
        time: (B,) tensor of diffusion timesteps
        audio: (B, N, M, yD) tensor of audio features, (batch_size, video_length, blocks, channels).
        cond: (B, N, cD) tensor of conditional features
        audio_cond: (B, N, zD) or (B, zD) tensor of audio conditional features
        """
        # bianma=time.time()                     # (B, D)
        motion_embeds = self.motion_embedder(motion) # (B, N, D), N: seq length
        _,seq_len,_=motion.shape
        time_embeds = self.time_embedder(times)    
        cache=True
        if cache:
            # emotion embedding
            emo_embeds = self.emo_embedder(emo, self.training)# (B, D)
            audio_cond=audio_cond.mean(1)
            audio_cond_embeds = self.identity_embedder(audio_cond)
    
            # audio embedding
            freqs_cos, freqs_sin = get_1d_rotary_pos_embed(self.dim, seq_len,theta=256, use_real=True, theta_rescale_factor=1)
            freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
            audio_embeds = self.audio_embedder(audio)  # (B, N, M, D)
            # self.bank.append(audio_embeds)
            M=audio_embeds.shape[2]
            audio_embeds = rearrange(audio_embeds, "b n m d -> b (n m) d")
            # print(audio_embeds.shape)
            c = time_embeds+emo_embeds
            # motion embedding

            freqs_cos2=rearrange(freqs_cos.unsqueeze(0).repeat(M,1,1), "n m d -> (n m) d")
            freqs_sin2=rearrange(freqs_sin.unsqueeze(0).repeat(M,1,1),"n m d -> (n m) d")
            freqs_cis2 = (freqs_cos2, freqs_sin2) if freqs_cos2 is not None else None

            freqs_cos3=rearrange(freqs_cos.unsqueeze(0).repeat(3*M,1,1), "n m d -> (n m) d")
            freqs_sin3=rearrange(freqs_sin.unsqueeze(0).repeat(3*M,1,1),"n m d -> (n m) d")
            freqs_cis3 = (freqs_cos3, freqs_sin3) if freqs_cos2 is not None else None
            
            # self.bank.append(emo_embeds)
            # self.bank.append(audio_cond_embeds)
            emo_embeds=emo_embeds.unsqueeze(1).repeat(1,seq_len,1)
            audio_cond_embeds=audio_cond_embeds.unsqueeze(1).repeat(1,seq_len,1)
        for block in (self.blocks4):
            motion_embeds,audio_embeds,emo_embeds,audio_cond_embeds = block(motion_embeds, c, audio_embeds,emo_embeds,audio_cond_embeds,mask,freqs_cis,freqs_cis2,causal=False)  
        audio_embeds=torch.cat((audio_embeds,emo_embeds,audio_cond_embeds), 1)
        for block in self.blocks2:
            motion_embeds,audio_embeds= block(seq_len,motion_embeds, c, audio_embeds,mask,freqs_cis,freqs_cis3,causal=False)
        motion_embeds=torch.cat((motion_embeds, audio_embeds), 1)
        for block in self.blocks:
            motion_embeds = block(seq_len,motion_embeds, c,mask,freqs_cis,freqs_cis3,causal=False)
        motion_embeds=motion_embeds[:,:seq_len,:]
        out = self.final_layer(motion_embeds, c)                          # (B, N, out_channels)
        # print("dit",time.time()-b)
        return out

    def forward_with_cfg(self, motion, time, audio, cfg_scale, emo=None, audio_cond=None):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        pass
        # # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # half = x[: len(x) // 2]
        # combined = torch.cat([half, half], dim=0)
        # model_out = self.forward(combined, t, y)
        # # For exact reproducibility reasons, we apply classifier-free guidance on only
        # # three channels by default. The standard approach to cfg applies it to all channels.
        # # This can be done by uncommenting the following line and commenting-out the line following that.
        # # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        # cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        # half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        # eps = torch.cat([half_eps, half_eps], dim=0)
        # return torch.cat([eps, rest], dim=1)



def TalkingHeadDiT_XL(**kwargs):
    return TalkingHeadDiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def TalkingHeadDiT_L(**kwargs):
    return TalkingHeadDiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def TalkingHeadDiT_B(**kwargs):
    return TalkingHeadDiT(depth=12, hidden_size=768, num_heads=12, **kwargs)
def TalkingHeadDiT_MM(**kwargs):
    return TalkingHeadDiT(depth=6, hidden_size=768, num_heads=12, **kwargs)
def TalkingHeadDiT_S(**kwargs):
    return TalkingHeadDiT(depth=12, hidden_size=384, num_heads=6, **kwargs)

def TalkingHeadDiT_T(**kwargs):
    return TalkingHeadDiT(depth=6, hidden_size=256, num_heads=4, **kwargs)




TalkingHeadDiT_models = {
    'TalkingHeadDiT-XL': TalkingHeadDiT_XL, 
    'TalkingHeadDiT-L':  TalkingHeadDiT_L, 
    'TalkingHeadDiT-MM': TalkingHeadDiT_MM, 
    'TalkingHeadDiT-B':  TalkingHeadDiT_B, 
    'TalkingHeadDiT-S':  TalkingHeadDiT_S, 
    'TalkingHeadDiT-T':  TalkingHeadDiT_T,
}