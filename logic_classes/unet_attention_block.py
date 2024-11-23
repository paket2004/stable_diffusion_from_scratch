import torch
from torch import nn
from .self_attention import SelfAttention
from .cross_attention import CrossAttention
from torch.nn import functional as F
class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)

        residue_long = x
        x = self.groupnorm(x)
        
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        x = x.view((n, c, h * w))
        
        x = x.transpose(-1, -2)
        
        # Normalization + Self-Attention with skip connection

       
        residue_short = x
        
        x = self.layernorm_1(x)
        
        x = self.attention_1(x)
        
        x += residue_short
    
        residue_short = x

        # Normalization + Cross-Attention with skip connection
        
        x = self.layernorm_2(x)
        
        x = self.attention_2(x, context)
        
        x += residue_short

        residue_short = x

        # Normalization + FFN with GeGLU and skip connection
        x = self.layernorm_3(x)
        
        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        
        x = x * F.gelu(gate)
     
        x = self.linear_geglu_2(x)
        
        x += residue_short
        
        x = x.transpose(-1, -2)
        
        x = x.view((n, c, h, w))

        # Final skip connection between initial input and output of the block
        return self.conv_output(x) + residue_long
