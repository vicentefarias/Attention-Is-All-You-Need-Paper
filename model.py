import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Embedding layer: maps each token index (0 to vocab_size-1) to a d_model-dimensional vector
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Scale embeddings by sqrt(d_model) for more stable training
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create a matrix for positional encodings
        pe = torch.zeros(seq_len, d_model)

        # Compute positional encodings
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin to even indices in the embedding dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices in the embedding dimensions
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add an extra dimension for batch size (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as a buffer (no gradient updates needed)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input tensor
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps =  eps 
        self.alpha = nn.Parameter(torch.ones(1)) # Scale parameter
        self.bias = nn.Parameter(torch.zeros(1)) # Shift parameter

    def forward(self, x):
        mean =  x.mean(dim = -1, keepdim=True) # Mean along the last dimension
        std =   x.std(dim = -1, keepdim=True)  # Std along the last dimension
        # Normalize and apply scale and bias
        return self.alpha * (x - mean) / (std + self.eps) + self.bias 
