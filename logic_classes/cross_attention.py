import torch
from torch import nn
import math
from torch.nn import functional as F
class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        input_shape = x.shape
        batch_size, _, _ = input_shape
        
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
    
        q = self.q_proj(x)
        
        k = self.k_proj(y)
        
        v = self.v_proj(y)
        q = q.view(interim_shape).transpose(1, 2) 
      
        k = k.view(interim_shape).transpose(1, 2) 
      
        v = v.view(interim_shape).transpose(1, 2) 
   
        weight = q @ k.transpose(-1, -2)
  
        weight /= math.sqrt(self.d_head)
   
        weight = F.softmax(weight, dim=-1)
        
        output = weight @ v
    
        output = output.transpose(1, 2).contiguous()
   
        output = output.view(input_shape)
        
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output