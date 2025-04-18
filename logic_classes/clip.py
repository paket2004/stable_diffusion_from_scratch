import torch
from torch import nn
from .clip_embedding import CLIPEmbedding
from .clip_layer import CLIPLayer
class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # num of words, embedding vector len, max_context_length
        self.embedding = CLIPEmbedding(49408, 768, 77)
        # clip layers with number of heads for attention, emb. size and number of layers
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])
        
        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # 64 bit signed int
        tokens = tokens.type(torch.long)

        # batch, seq_len -> batch, seq_len, dimension
        state = self.embedding(tokens)

        # Apply encoder layers similar to the Transformer's encoder.
        for layer in self.layers: 
            state = layer(state)
        output = self.layernorm(state)
        # batch size, seq_len, dim
        return output
