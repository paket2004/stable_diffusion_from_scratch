import torch
from torch import nn
class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))
    
    def forward(self, tokens):
        # batch_size, seq_len -> batch size, seq_len, dimension
        x = self.token_embedding(tokens)
        # just add positional embedding
        x += self.position_embedding
        
        return x
