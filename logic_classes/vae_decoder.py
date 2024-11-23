import torch
from torch import nn
from .vae_residual_block import VAE_ResidualBlock
from .vae_attention_block import VAE_AttentionBlock
class VAE_Decoder(nn.Sequential):
  def __init__(self):
    super().__init__(
        nn.Conv2d(4, 4, kernel_size=1, padding=0),

        nn.Conv2d(4, 512, kernel_size=3, padding=1),

        VAE_ResidualBlock(512, 512),

        VAE_AttentionBlock(512),

        VAE_ResidualBlock(512, 512),

        VAE_ResidualBlock(512, 512),

        VAE_ResidualBlock(512, 512),
    
        VAE_ResidualBlock(512, 512),

        nn.Upsample(scale_factor=2),

        nn.Conv2d(512, 512, kernel_size=3, padding=1),

        VAE_ResidualBlock(512, 512),

        VAE_ResidualBlock(512, 512),

        VAE_ResidualBlock(512, 512),

        nn.Upsample(scale_factor=2),

        nn.Conv2d(512, 512, kernel_size=3, padding=1),

        VAE_ResidualBlock(512, 256),

        VAE_ResidualBlock(256, 256),

        VAE_ResidualBlock(256, 256),
        nn.Upsample(scale_factor=2),

        nn.Conv2d(256, 256, kernel_size=3, padding=1),

        VAE_ResidualBlock(256, 128),

        VAE_ResidualBlock(128, 128),

        VAE_ResidualBlock(128, 128),

        nn.GroupNorm(32, 128),

        nn.SiLU(),
        # (batch_Size, 128, h, w) -> (batch_Size, 3, h, w)
        nn.Conv2d(128, 3, kernel_size=3, padding=1)
    )
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (batch_Size, 4, height / 8, width / 8)

    x /= 0.18215

    for module in self:
      x = module(x)
    return x