import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from diffusers.models.modeling_utils import ModelMixin

from .blocks import _basic_init, DiTBlock
from .modules import RMSNorm
from .positional_embedding import get_1d_sincos_pos_embed

#################################################################################
#          Embedding Layers for Timesteps, Emotion Labels and Motions           #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int=256, dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def initialize_weights(self):
        self.apply(_basic_init)
        # Initialize timestep embedding MLP:
        for l in [0, 2]:
            nn.init.normal_(self.mlp[l].weight, std=0.02)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        if torch.is_floating_point(t):
            embedding = embedding.to(dtype=t.dtype)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float, dtype=None, device=None):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size, dtype=None, device=None)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
    
    def initialize_weights(self):
        # Initialize label embedding table:
        nn.init.normal_(self.embedding_table.weight, std=0.02)

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class MotionEmbedder(nn.Module):
    """
    Embeds motion into vector representations, Motion shape B x L x D
    """
    def __init__(self, motion_dim: int, hidden_size: int, dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(motion_dim, hidden_size, bias=True, dtype=None, device=None),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=None, device=None),
        )
    
    def initialize_weights(self):
        self.apply(_basic_init)
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        for l in [0, 2]:
            w = self.mlp[l].weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.mlp[l].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AudioEmbedder(ModelMixin):
    """Audio Projection Model

    This class defines an audio projection model that takes audio embeddings as input
    and produces context tokens as output. The model is based on the ModelMixin class
    and consists of multiple linear layers and activation functions. It can be used
    for various audio processing tasks.

    Attributes:
        seq_len (int): The length of the audio sequence.
        blocks (int): The number of blocks in the audio projection model.
        channels (int): The number of channels in the audio projection model.
        intermediate_dim (int): The intermediate dimension of the model.
        context_tokens (int): The number of context tokens in the output.
        output_dim (int): The output dimension of the context tokens.

    Methods:
        __init__(self, seq_len=5, blocks=12, channels=768, intermediate_dim=512, context_tokens=32, output_dim=768):
            Initializes the AudioProjModel with the given parameters.
        forward(self, audio_embeds):
            Defines the forward pass for the AudioProjModel.
            Parameters:
            audio_embeds (torch.Tensor): The input audio embeddings with shape (batch_size, video_length, blocks, channels).
            Returns:
            context_tokens (torch.Tensor): The output context tokens with shape (batch_size, video_length, context_tokens, output_dim).

    """

    def __init__(
        self,
        seq_len=5,
        blocks=12,  # add a new parameter blocks
        channels=768,  # add a new parameter channels
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        input_len = 80,
        condition_dim = 63,
        norm_type="rms_norm",
        qk_norm="rms_norm"
    ):
        super().__init__()
        input_dim = (
            seq_len * blocks * channels
        )  # update input_dim to be the product of blocks and channels.
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(input_dim, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)

        self.norm = nn.LayerNorm(output_dim) if norm_type == "layer_norm" else RMSNorm(output_dim)

    def initialize_weights(self):
        self.apply(_basic_init)
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.proj1.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj1.bias, 0)

        w = self.proj2.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj2.bias, 0)
        
        w = self.proj3.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj3.bias, 0)

    def forward(self, audio_embeds, conditions=None, emo=None):
        """
        Defines the forward pass for the AudioProjModel.

        Parameters:
            audio_embeds (torch.Tensor): The input audio embeddings with shape (batch_size, video_length, blocks, channels).
            conditions (torch.Tensor): optional other conditions with shape (batch_size, video_length, channels) or (batch_size, channels)
            emo (torch.Tensor): optional emotion embedding with shape (batch_size, channels)
        Returns:
            context_tokens (torch.Tensor): The output context tokens with shape (batch_size, video_length, context_tokens, output_dim).
        """
        # merge
        video_length = audio_embeds.shape[1]
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.reshape(batch_size, window_size * blocks * channels)

        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(
            batch_size, self.context_tokens, self.output_dim
        )

        context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(
            context_tokens, "(bz f) m c -> bz f m c", f=video_length
        )

        return context_tokens


class ConditionAudioEmbedder(ModelMixin):
    """Audio Projection Model with conditions

    This class defines an audio projection model that takes audio embeddings as input
    and produces context tokens as output. The model is based on the ModelMixin class
    and consists of multiple linear layers and activation functions. It can be used
    for various audio processing tasks.

    Attributes:
        seq_len (int): The length of the audio sequence.
        blocks (int): The number of blocks in the audio projection model.
        channels (int): The number of channels in the audio projection model.
        intermediate_dim (int): The intermediate dimension of the model.
        context_tokens (int): The number of context tokens in the output.
        output_dim (int): The output dimension of the context tokens.

    Methods:
        __init__(self, seq_len=5, blocks=12, channels=768, intermediate_dim=512, context_tokens=32, output_dim=768):
            Initializes the AudioProjModel with the given parameters.
        forward(self, audio_embeds):
            Defines the forward pass for the AudioProjModel.
            Parameters:
            audio_embeds (torch.Tensor): The input audio embeddings with shape (batch_size, video_length, blocks, channels).
            Returns:
            context_tokens (torch.Tensor): The output context tokens with shape (batch_size, video_length, context_tokens, output_dim).

    """

    def __init__(
        self,
        seq_len=5,
        blocks=12,  # add a new parameter blocks
        channels=768,  # add a new parameter channels
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        input_len = 80,
        condition_dim=63,
        norm_type="rms_norm",
        qk_norm="rms_norm"
    ):
        super().__init__()
        self.input_dim = (
            seq_len * blocks * channels + condition_dim
        )  # update input_dim to be the product of blocks and channels.
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)

        self.norm = nn.LayerNorm(output_dim) if norm_type == "layer_norm" else RMSNorm(output_dim)

    def initialize_weights(self):
        self.apply(_basic_init)
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.proj1.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj1.bias, 0)

        w = self.proj2.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj2.bias, 0)
        
        w = self.proj3.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj3.bias, 0)

    def forward(self, audio_embeds, conditions, emo=None):
        """
        Defines the forward pass for the AudioProjModel.

        Parameters:
            audio_embeds (torch.Tensor): The input audio embeddings with shape (batch_size, video_length, blocks, channels).
            conditions (torch.Tensor): other conditions with shape (batch_size, video_length, channels)
            emo (torch.Tensor): optional emotion embedding with shape (batch_size, channels)
        Returns:
            context_tokens (torch.Tensor): The output context tokens with shape (batch_size, video_length, context_tokens, output_dim).
        """
        # merge
        video_length = audio_embeds.shape[1]
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.reshape(batch_size, window_size * blocks * channels)  # bz*f, C
        # concat conditions
        conditions = rearrange(conditions, "bz f c -> (bz f) c")                          # bz*f, c
        audio_embeds = torch.cat([audio_embeds, conditions], dim=1)                       # bz*f, C+c

        # forward
        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(
            batch_size, self.context_tokens, self.output_dim
        )

        context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(
            context_tokens, "(bz f) m c -> bz f m c", f=video_length
        )

        return context_tokens


class SimpleAudioEmbedder(ModelMixin):
    """Simplfied Audio Projection Model

    This class defines an audio projection model that takes audio embeddings as input
    and produces context tokens as output. The model is based on the ModelMixin class
    and consists of multiple linear layers and activation functions. It can be used
    for various audio processing tasks.

    Attributes:
        seq_len (int): The length of the audio sequence.
        blocks (int): The number of blocks in the audio projection model.
        channels (int): The number of channels in the audio projection model.
        intermediate_dim (int): The intermediate dimension of the model.
        context_tokens (int): The number of context tokens in the output.
        output_dim (int): The output dimension of the context tokens.

    Methods:
        __init__(self, seq_len=5, blocks=12, channels=768, intermediate_dim=512, context_tokens=32, output_dim=768):
            Initializes the AudioProjModel with the given parameters.
        forward(self, audio_embeds):
            Defines the forward pass for the AudioProjModel.
            Parameters:
            audio_embeds (torch.Tensor): The input audio embeddings with shape (batch_size, video_length, blocks, channels).
            Returns:
            context_tokens (torch.Tensor): The output context tokens with shape (batch_size, video_length, context_tokens, output_dim).

    """

    def __init__(
        self,
        seq_len=5,
        blocks=12,  # add a new parameter blocks
        channels=768,  # add a new parameter channels
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        input_len = 80,
        condition_dim = 63,
        norm_type="rms_norm",
        qk_norm="rms_norm",
        n_blocks = 4,
        n_heads = 4,
        mlp_ratio = 4
    ):
        super().__init__()
        self.input_dim = (
            seq_len * blocks * channels
        )  # update input_dim to be the product of blocks and channels.
        self.context_tokens = context_tokens
        self.output_dim = output_dim
        self.condition_dim=condition_dim
        # define input layer
        
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, intermediate_dim, bias=True, dtype=None, device=None),
            nn.SiLU(),
            nn.Linear(intermediate_dim, condition_dim+2*intermediate_dim, bias=True, dtype=None, device=None),
        )

        self.condition2_layer = nn.Linear(condition_dim, condition_dim)
        self.emo_layer =nn.Linear(intermediate_dim, intermediate_dim)
        # fuse layer for fusion additonal conditions, like ref_kp
        self.use_condition = True
        self.condition_layer = nn.Linear(condition_dim+intermediate_dim, intermediate_dim)
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, input_len, intermediate_dim), requires_grad=False)

        # # mid blocks
        self.mid_blocks = nn.ModuleList([
            DiTBlock(
                intermediate_dim, n_heads, 
                mlp_ratio=mlp_ratio, 
                norm_type=norm_type, 
                qk_norm=qk_norm
            ) for _ in range(n_blocks)
        ])
        # output layer
        self.output_layer = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.output_layer2 = nn.Linear(condition_dim+condition_dim, context_tokens * output_dim)
        self.output_layer3 = nn.Linear(intermediate_dim+intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_type == "layer_norm" else RMSNorm(output_dim)
        self.norm2= nn.LayerNorm(output_dim) if norm_type == "layer_norm" else RMSNorm(output_dim)
        self.norm3= nn.LayerNorm(output_dim) if norm_type == "layer_norm" else RMSNorm(output_dim)
    def initialize_weights(self):
        # 1. Initialize input layer
        for l in [0, 2]:
            w = self.input_layer[l].weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.input_layer[l].bias, 0)
        w = self.emo_layer.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.emo_layer.bias, 0)
        #w = self.input_layer.weight.data
        #nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        #nn.init.constant_(self.input_layer.bias, 0)
        # 2. Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2])
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # 3. Initialize condition layer
        nn.init.normal_(self.condition_layer.weight, std=0.02)
        nn.init.constant_(self.condition_layer.bias, 0)
        nn.init.normal_(self.condition2_layer.weight, std=0.02)
        nn.init.constant_(self.condition2_layer.bias, 0)
        # 4. Initialize mid blocks
        # for block in self.mid_blocks:
        #     block.initialize_weights()
        # 5. Initialize output layer
        w = self.output_layer.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.output_layer.bias, 0)

        w = self.output_layer2.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.output_layer2.bias, 0)

        w = self.output_layer3.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.output_layer3.bias, 0)

    def forward(self, audio_embeds, conditions, emo_embeds,mask=None,freqs_cis=None):
        """
        Defines the forward pass for the AudioProjModel.

        Parameters:
            audio_embeds (torch.Tensor): The input audio embeddings with shape (batch_size, video_length, blocks, channels).
            conditions (torch.Tensor): other conditions with shape (batch_size, video_length, channels) or (batch_size, channels)
            emo_embeds (torch.Tensor): optional emotion embedding with shape (batch_size, channels)
        Returns:
            context_tokens (torch.Tensor): The output context tokens with shape (batch_size, video_length, context_tokens, output_dim).
        """
        # preprare inputs
        condition2=self.condition2_layer(conditions)
        emo2=self.emo_layer(emo_embeds)

        video_length = audio_embeds.shape[1]
        emo_embeds=emo_embeds.unsqueeze(1).repeat(1,video_length,1)
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")

        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.reshape(batch_size, window_size * blocks * channels)
        
        # input layer
        audio_embeds = self.input_layer(audio_embeds)
        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", f=video_length)
        # audio_embeds=audio_embeds+self.pos_embed[:,:,:-1]
        audio_kp=audio_embeds[:,:,:self.condition_dim]
        audio_xs,audio_emo=audio_embeds[:,:,self.condition_dim:].chunk(2, dim=-1)
        #enhance
        audio_enc_kp=torch.cat([audio_kp,conditions], dim=-1)
        audio_enc_emo=torch.cat([audio_emo,emo_embeds], dim=-1)
        audio_enc_kp=rearrange(audio_enc_kp, "bz f c -> (bz f) c")
        audio_enc_emo=rearrange(audio_enc_emo, "bz f c -> (bz f) c")
        kp_context = self.output_layer2(audio_enc_kp).reshape(
            batch_size, self.context_tokens, self.output_dim
        )
        kp_context=kp_context
        kp_context=self.norm2(kp_context)
        emo_context = self.output_layer3(audio_enc_emo).reshape(
            batch_size, self.context_tokens, self.output_dim
        )
        emo_context=self.norm3(emo_context)
        # condition layer
        if self.use_condition:
            audio_xs = self.condition_layer(torch.cat([audio_xs, condition2], dim=-1))
        # positional embeddings
              # add positional embedding
        audio_xs=audio_xs+self.pos_embed
        # mid blocks
        for block in self.mid_blocks:
            audio_xs = block(audio_xs, emo2,mask=mask,freqs_cis=None)
        # output layer
        audio_xs = rearrange(audio_xs, "bz f c -> (bz f) c")
        audio_xs = self.output_layer(audio_xs).reshape(
            batch_size, self.context_tokens, self.output_dim
        )
        audio_xs = self.norm(audio_xs)

        kp_context=rearrange(kp_context, "(bz f) m c -> bz f m c", f=video_length)
        emo_context=rearrange(emo_context, "(bz f) m c -> bz f m c", f=video_length)
        audio_xs=rearrange(audio_xs, "(bz f) m c -> bz f m c", f=video_length)
        # context_tokens=torch.cat([audio_xs, kp_context,emo_context], dim=1)
        # context_tokens = self.output_layer(audio_embeds).reshape(
        #     batch_size, self.context_tokens, self.output_dim
        # )
        # # context_tokens = self.norm(context_tokens)
        # context_tokens = rearrange(
        #     context_tokens, "(bz f) m c -> bz f m c", f=video_length
        # )

        return kp_context,emo_context,audio_xs,audio_kp,audio_emo,conditions,emo_embeds


class ConditionEmbedder(nn.Module):
    def __init__(
        self,
        input_dim=768,  # add a new parameter channels
        intermediate_dim=1024,
        output_dim=2048,
        input_len = 80,
        norm_type="rms_norm",
        qk_norm="rms_norm",
        n_blocks = 4,
        n_heads = 4,
        mlp_ratio = 4
    ):
        super().__init__()
        self.input_dim = input_dim 
        self.output_dim = output_dim

        # define input layer
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, intermediate_dim, bias=True, dtype=None, device=None),
            nn.SiLU(),
            nn.Linear(intermediate_dim, intermediate_dim, bias=True, dtype=None, device=None),
        )
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, input_len, intermediate_dim), requires_grad=False)

        # mid blocks
        self.mid_blocks = nn.ModuleList([
            DiTBlock(
                intermediate_dim, n_heads, 
                mlp_ratio=mlp_ratio, 
                norm_type=norm_type, 
                qk_norm=qk_norm
            ) for _ in range(n_blocks)
        ])
        # output layer
        self.output_layer = nn.Linear(intermediate_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_type == "layer_norm" else RMSNorm(output_dim)

    def initialize_weights(self):
        # 1. Initialize input layer
        for l in [0, 2]:
            w = self.input_layer[l].weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.input_layer[l].bias, 0)

        # 2. Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2])
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # 3. Initialize mid blocks
        for block in self.mid_blocks:
            block.initialize_weights()
        # 4. Initialize output layer
        w = self.output_layer.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, cond_embeds, emo_embeds):
        # cond_embeds, B, L, D; emo_embeds, B, D
        # input layer
        #batch_size, length, channels = cond_embeds.shape
        #cond_embeds = rearrange(cond_embeds, "bz f c -> (bz f) c")
        cond_embeds = self.input_layer(cond_embeds)
        # positional embeddings
        #cond_embeds = rearrange(cond_embeds, "bz (f c) -> bz f c")
        cond_embeds = cond_embeds + self.pos_embed
        # mid blocks
        for block in self.mid_blocks:
            cond_embeds = block(cond_embeds, emo_embeds)
        # output layer
        #cond_embeds = rearrange(cond_embeds, "bz f c -> (bz f) c")
        context_tokens = self.output_layer(cond_embeds)
        context_tokens = self.norm(context_tokens)

        return context_tokens


class VectorEmbedder(nn.Module):
    """Embeds a flat vector of dimension input_dim"""

    def __init__(self, input_dim: int, hidden_size: int, dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        flatten: bool = True,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        if img_size is not None:
            self.img_size = (img_size, img_size)
            self.grid_size = tuple(
                [s // p for s, p in zip(self.img_size, self.patch_size)]
            )
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        # flatten spatial dim and transpose to channels last, kept for bwd compat
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            dtype=dtype,
            device=device,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return x

