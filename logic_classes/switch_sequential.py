import torch
from torch import nn
from .unet_attention_block import UNET_AttentionBlock
from .unet_residual_block import UNET_ResidualBlock
class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x