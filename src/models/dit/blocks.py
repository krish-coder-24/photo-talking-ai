import torch
import torch.nn as nn
import numbers

from .modules import RMSNorm, SelfAttention, CrossAttention, Mlp,MMdual_attention,MMsingle_attention,MMfour_attention

from einops import rearrange, repeat
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _basic_init(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning, contains CrossAttention.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        norm_type = block_kwargs.get("norm_type", "rms_norm")

        assert norm_type in ["layer_norm", "rms_norm"]

        make_norm_layer = (
            nn.LayerNorm if norm_type == "layer_norm" else RMSNorm
        )

        self.norm1 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.norm2 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def initialize_weights(self):
        self.apply(_basic_init)
        # Zero-out adaLN modulation layers in DiT blocks:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c,mask=None,freqs_cis=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn1(modulate(self.norm1(x), shift_msa, scale_msa),mask,freqs_cis)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class MMSingleStreamBlock(nn.Module):
    ''' A multimodal dit block with seperate modulation '''
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        norm_type = block_kwargs.get("norm_type", "rms_norm")

        assert norm_type in ["layer_norm", "rms_norm"]

        make_norm_layer = (
            nn.LayerNorm if norm_type == "layer_norm" else RMSNorm
        )

        self.norm1 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = MMsingle_attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.attn2 = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
        self.norm3 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.qkv_xs = nn.Linear(hidden_size, hidden_size * 3+mlp_hidden_dim, bias=True)
        # self.xs_mlp = Mlp(in_features=hidden_size+mlp_hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.linear2 = nn.Linear(
            hidden_size + mlp_hidden_dim, hidden_size,
        )
        self.mlp_act = approx_gelu()
        self.adaLN_modulation_xs = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3* hidden_size, bias=True)
        )
        self.hidden_size=hidden_size
        self.mlp_hidden_dim=mlp_hidden_dim
    def initialize_weights(self):
        self.apply(_basic_init)
        # Zero-out adaLN modulation layers in DiT blocks:
        nn.init.constant_(self.adaLN_modulation_xs[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_xs[-1].bias, 0)

    def forward(self,seq_len, x, c,mask=None,freqs_cis=None,freqs_cis2=None,causal=False):
        shift_msa_xs, scale_msa_xs, gate_msa_xs = self.adaLN_modulation_xs(c).chunk(3, dim=1)
        # Prepare for attention
        x_mod=modulate(self.norm1(x), shift_msa_xs, scale_msa_xs)
        qkv, mlp = torch.split(
            self.qkv_xs(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )
        att1= self.attn1(seq_len,qkv,mask,causal=causal,freqs_cis=freqs_cis,freqs_cis2=freqs_cis2)
        output=self.linear2(torch.cat((att1, self.mlp_act(mlp)), 2))
        x=x+gate_msa_xs.unsqueeze(1)*output
        return x
class MMfourStreamBlock(nn.Module):
    ''' A multimodal dit block with seperate modulation '''
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        norm_type = block_kwargs.get("norm_type", "rms_norm")

        assert norm_type in ["layer_norm", "rms_norm"]

        make_norm_layer = (
            nn.LayerNorm if norm_type == "layer_norm" else RMSNorm
        )

        self.norm1 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = MMfour_attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
        self.norm2 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.attn2 = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
        self.norm3 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)

        self.norm4 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm5 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm6 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm7 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm8 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.xs_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.audio_mlp1 = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.audio_mlp2 = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.audio_mlp3 = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation_xs = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.adaLN_modulation_audio1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        
        )
        self.adaLN_modulation_audio2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_modulation_audio3 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True))
    def initialize_weights(self):
        self.apply(_basic_init)
        # Zero-out adaLN modulation layers in DiT blocks:
        nn.init.constant_(self.adaLN_modulation_xs[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_xs[-1].bias, 0)

        nn.init.constant_(self.adaLN_modulation_audio1[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_audio1[-1].bias, 0)

        nn.init.constant_(self.adaLN_modulation_audio2[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_audio2[-1].bias, 0)

        nn.init.constant_(self.adaLN_modulation_audio3[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_audio3[-1].bias, 0)


    def forward(self, x, c, y1,y2,y3,mask=None,freqs_cis=None,freqs_cis2=None,causal=False):
        shift_msa_xs, scale_msa_xs, gate_msa_xs, shift_mlp_xs, scale_mlp_xs, gate_mlp_xs = self.adaLN_modulation_xs(c).chunk(6, dim=1)
        shift_mca_audio1, scale_mca_audio1, gate_mca_audio1, shift_mlp_audio1, scale_mlp_audio1, gate_mlp_audio1 = self.adaLN_modulation_audio1(c).chunk(6, dim=1)
        shift_mca_audio2, scale_mca_audio2, gate_mca_audio2, shift_mlp_audio2, scale_mlp_audio2, gate_mlp_audio2 = self.adaLN_modulation_audio2(c).chunk(6, dim=1)
        shift_mca_audio3, scale_mca_audio3, gate_mca_audio3, shift_mlp_audio3, scale_mlp_audio3, gate_mlp_audio3= self.adaLN_modulation_audio3(c).chunk(6, dim=1)
        # Prepare for attention
        att1,att2,att3,att4= self.attn1( modulate(self.norm1(x), shift_msa_xs, scale_msa_xs),
                                modulate(self.norm2(y1), shift_mca_audio1, scale_mca_audio1),
                                modulate(self.norm3(y2), shift_mca_audio2, scale_mca_audio2),
                                modulate(self.norm4(y3), shift_mca_audio3, scale_mca_audio3),
                                mask,causal=causal,freqs_cis=freqs_cis,freqs_cis2=freqs_cis2)
        x=x+gate_msa_xs.unsqueeze(1)*att1
        y1=y1+gate_mca_audio1.unsqueeze(1)*att2
        y2=y2+gate_mca_audio2.unsqueeze(1)*att3
        y3=y3+gate_mca_audio3.unsqueeze(1)*att4

        x = x + gate_mlp_xs.unsqueeze(1) * self.xs_mlp(modulate(self.norm5(x), shift_mlp_xs, scale_mlp_xs))
        y1 = y1 + gate_mlp_audio1.unsqueeze(1) * self.audio_mlp1(modulate(self.norm6(y1), shift_mlp_audio1, scale_mlp_audio1))
        y2 = y2 + gate_mlp_audio2.unsqueeze(1) * self.audio_mlp2(modulate(self.norm7(y2), shift_mlp_audio2, scale_mlp_audio2))
        y3 = y3 + gate_mlp_audio3.unsqueeze(1) * self.audio_mlp3(modulate(self.norm8(y3), shift_mlp_audio3, scale_mlp_audio3))
        return x,y1,y2,y3
class MMDoubleStreamBlock(nn.Module):
    ''' A multimodal dit block with seperate modulation '''
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        norm_type = block_kwargs.get("norm_type", "rms_norm")

        assert norm_type in ["layer_norm", "rms_norm"]

        make_norm_layer = (
            nn.LayerNorm if norm_type == "layer_norm" else RMSNorm
        )

        self.norm1 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = MMdual_attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
        self.norm2 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.attn2 = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
        self.norm3 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.xs_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.audio_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation_xs = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.adaLN_modulation_audio = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
    def initialize_weights(self):
        self.apply(_basic_init)
        # Zero-out adaLN modulation layers in DiT blocks:
        nn.init.constant_(self.adaLN_modulation_xs[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_xs[-1].bias, 0)

        nn.init.constant_(self.adaLN_modulation_audio[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_audio[-1].bias, 0)

    def forward(self, seq_len,x, c, y,mask=None,freqs_cis=None,freqs_cis2=None,causal=False):
        shift_msa_xs, scale_msa_xs, gate_msa_xs, shift_mlp_xs, scale_mlp_xs, gate_mlp_xs = self.adaLN_modulation_xs(c).chunk(6, dim=1)
        shift_mca_audio, scale_mca_audio, gate_mca_audio, shift_mlp_audio, scale_mlp_audio, gate_mlp_audio = self.adaLN_modulation_audio(c).chunk(6, dim=1)
        # Prepare for attention
        att1,att2 = self.attn1(seq_len,modulate(self.norm1(x), shift_msa_xs, scale_msa_xs),modulate(self.norm2(y), shift_mca_audio, scale_mca_audio),mask,causal=causal,freqs_cis=freqs_cis,freqs_cis2=freqs_cis2)
        x=x+gate_msa_xs.unsqueeze(1)*att1
        y=y+gate_mca_audio.unsqueeze(1)*att2
        x = x + gate_mlp_xs.unsqueeze(1) * self.xs_mlp(modulate(self.norm3(x), shift_mlp_xs, scale_mlp_xs))
        y = y + gate_mlp_audio.unsqueeze(1) * self.audio_mlp(modulate(self.norm4(y), shift_mlp_audio, scale_mlp_audio))
        return x,y
class CrossDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning, contains CrossAttention.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        norm_type = block_kwargs.get("norm_type", "rms_norm")

        assert norm_type in ["layer_norm", "rms_norm"]

        make_norm_layer = (
            nn.LayerNorm if norm_type == "layer_norm" else RMSNorm
        )

        self.norm1 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
        self.norm2 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn2 = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
        self.norm3 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
    
    def initialize_weights(self):
        self.apply(_basic_init)
        # Zero-out adaLN modulation layers in DiT blocks:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, y,mask=None):
        shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn1(modulate(self.norm1(x), shift_msa, scale_msa),mask)
        x = x + gate_mca.unsqueeze(1) * self.attn2(modulate(self.norm2(x), shift_mca, scale_mca), y,mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x
class SelfBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning, contains CrossAttention.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        norm_type = block_kwargs.get("norm_type", "rms_norm")

        assert norm_type in ["layer_norm", "rms_norm"]

        make_norm_layer = (
            nn.LayerNorm if norm_type == "layer_norm" else RMSNorm
        )
        
        self.norm2 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn2 = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
    
    def initialize_weights(self):
        self.apply(_basic_init)
        # Zero-out adaLN modulation layers in DiT blocks:

    def forward(self, x, y,mask=None):
        x = x + self.attn2(self.norm2(x),mask)
        return x
class CrossBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning, contains CrossAttention.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        norm_type = block_kwargs.get("norm_type", "rms_norm")

        assert norm_type in ["layer_norm", "rms_norm"]

        make_norm_layer = (
            nn.LayerNorm if norm_type == "layer_norm" else RMSNorm
        )
        
        self.norm2 = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn2 = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
    
    def initialize_weights(self):
        self.apply(_basic_init)
        # Zero-out adaLN modulation layers in DiT blocks:

    def forward(self, x, y,mask=None):
        x = x + self.attn2(self.norm2(x), y,mask)
        return x
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels, norm_type="rms_norm"):
        super().__init__()
        assert norm_type in ["layer_norm", "rms_norm"]

        make_norm_layer = (
            nn.LayerNorm if norm_type == "layer_norm" else RMSNorm
        )

        self.norm_final = make_norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def initialize_weights(self):
        self.apply(_basic_init)
        # Zero-out output layers:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x