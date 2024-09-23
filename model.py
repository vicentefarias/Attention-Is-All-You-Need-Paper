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

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h 
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h 
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o = nn.Linear(d_model, d_model) # Wo 
        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # Calculate Attention Scores
        # (Batch, h, Seq_len, d_k) --> (Batch, h, Seq_len, Seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Apply mask to attention scores
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        # (Batch, h, Seq_len, Seq_len)
        attention_scores = attention_scores.softmax(dim = -1)

        # Apply dropout
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Return the weighted values and attention scores
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # Linear Projections
        query = self.w_q(q) # (Batch, Seq_len, d_model) 
        key = self.w_k(k)   # (Batch, Seq_len, d_model) 
        value = self.w_v(v) # (Batch, Seq_len, d_model) 

        # Reshape and transpose for multi-head attention
        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, h, d_k) --> (Batch, h, Seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        # Compute Attention Scores
        x, attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # Reshape to (Batch, Seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

        