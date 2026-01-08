import torch
import torch.nn as nn
import torch.optim as optim
import math

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias




class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout: nn.Dropout=None):
        d_k = query.shape[-1]
        # (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask: expected (batch, seq_len) or (batch, 1, seq_len)
            # expand to (batch, 1, 1, seq_len) so it can broadcast to (batch, h, seq_len, seq_len)
            # ensure same device & dtype
            mask_ = mask.to(device=attention_scores.device)
            # if mask is 1/0 float, convert to bool
            if mask_.dtype != torch.bool:
                mask_bool = (mask_ != 0)
            else:
                mask_bool = mask_
            mask_bool = mask_bool.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            # use non-inplace masked_fill to be safer with autograd
            attention_scores = attention_scores.masked_fill(~mask_bool, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)



class encoder(nn.Module) : 
    def __init__(self, d_latent ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, d_latent)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(8)
        self.layer_norm2 = nn.LayerNorm(16)
        
    def forward(self , x ) : 
        # x : (b , 1)
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

class decoder(nn.Module) : 
    def __init__(self , d_latent) : 
        super().__init__()
        self.fc1 = nn.Linear(d_latent, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(16)
        self.layer_norm2 = nn.LayerNorm(8)
        
    def forward(self , x ) : 
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

class Auto_encoder(nn.Module) :
    def __init__(self , d_latent) : 
        super().__init__()
        self.encoder = encoder(d_latent)
        self.decoder = decoder(d_latent)
    def forward(self , x )  :
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return encoder_out , decoder_out

class Multi_auto_encoder(nn.Module)  :
    def __init__(self , d_latent , N) : 
        super().__init__()
        self.auto_encoders = nn.ModuleList([
            Auto_encoder(d_latent) for _ in range(N)
        ])
    def forward(self , x ) : 
        out_encoder = []
        out_decoder = [] 
        b , seq_len , N = x.shape
        for i in range(len(self.auto_encoders)) : 
            out_enc , out_dec = self.auto_encoders[i](x[: , : , i].reshape(-1 , 1))
            out_encoder.append(out_enc.reshape(b , seq_len , -1)) #(b  , seqLen , d_latent)
            out_decoder.append(out_dec.reshape(b , seq_len , -1)) #(b  , seqLen , 1)
        return  torch.concat(out_encoder , dim=-1) , torch.concat(out_decoder , dim=-1)
                #(b  , seqLen , d_latent * N)                   #(b  , seqLen , N)


class compressor(nn.Module):
    # this module compresses information and sends it to the capsnet 
    def __init__(self, d_in, d_out): 
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        
        # Improved encoder with multiple layers
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_out * 2),
            nn.LayerNorm(d_out * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_out * 2, d_out),
            nn.LayerNorm(d_out),
            nn.ReLU()
        )
        
        # Attention-based pooling for better aggregation
        self.attention_pool = nn.Sequential(
            nn.Linear(d_in, d_in // 2),
            nn.Tanh(),
            nn.Linear(d_in // 2, 1)
        )
        
        # Decoder with improved architecture
        self.decoder = nn.Sequential(
            nn.Linear(d_out, d_out * 2),
            nn.LayerNorm(d_out * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_out * 2, d_in)
        )
        
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_in)

    def forward(self, x, mask): 
        # x: (b, seq_len, d_in)
        # mask: (b, seq_len)
        b, seq_len, _ = x.shape
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Compute attention weights for each timestep
        # x_flat: (b * seq_len, d_in)
        x_flat = x.reshape(-1, self.d_in)
        attn_scores = self.attention_pool(x_flat).reshape(b, seq_len, 1)  # (b, seq_len, 1)
        
        # Apply mask to attention scores
        mask_expanded = mask.unsqueeze(-1)  # (b, seq_len, 1)
        attn_scores = attn_scores.masked_fill(mask_expanded == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)  # (b, seq_len, 1)
        
        # Encode all timesteps
        x_reshaped = x.reshape(-1, self.d_in)  # (b * seq_len, d_in)
        encoded_all = self.encoder(x_reshaped)  # (b * seq_len, d_out)
        encoded_features = encoded_all.reshape(b, seq_len, self.d_out)  # (b, seq_len, d_out)
        
        # Attention-weighted pooling
        v = (encoded_features * attn_weights).sum(dim=1)  # (b, d_out)
        
        # Decode for reconstruction (expand v to match sequence length)
        v_expanded = v.unsqueeze(1).expand(b, seq_len, self.d_out)  # (b, seq_len, d_out)
        v_flat = v_expanded.reshape(-1, self.d_out)  # (b * seq_len, d_out)
        decoded_flat = self.decoder(v_flat)  # (b * seq_len, d_in)
        out_enc = decoded_flat.reshape(b, seq_len, self.d_in)  # (b, seq_len, d_in)
        
        return v, out_enc


        
class ClientNetwork(nn.Module): 
    def __init__(self, N, d_latent, h, dropout, seq_len, cap_in_dim , lr): 
        super().__init__()
        self.multi_autoEncoder = Multi_auto_encoder(d_latent, N)
        self.compressor = compressor(d_in=N * d_latent, d_out=cap_in_dim)
        self.attention = MultiHeadAttentionBlock(d_latent * N, h, dropout)
        self.PE = PositionalEncoding(d_latent * N, seq_len, dropout)
        self.resudal_connection = ResidualConnection(d_latent * N, dropout)
        self.loss_fn = nn.L1Loss()
        self.optimizer = optim.Adam(self.parameters() , lr=lr)

    def forward(self, x, mask , train=True):
        # ensure mask is on same device
        mask = mask.to(x.device)
        out_encoder, out_decoder1 = self.multi_autoEncoder(x)
        out_encoder = self.PE(out_encoder)

        out_attention = self.resudal_connection(
            out_encoder,
            lambda x: self.attention(x, x, x, mask) # attention returns (out, scores)
        )

        v, out_decoder2 = self.compressor(out_attention, mask)
        if train:
            # Apply mask to losses for better training
            mask_expanded = mask.unsqueeze(-1)  # (b, seq_len, 1)
            
            # Masked reconstruction loss for autoencoder
            masked_x = x * mask_expanded
            masked_dec1 = out_decoder1 * mask_expanded
            loss1 = self.loss_fn(masked_x, masked_dec1)
            
            # Masked reconstruction loss for compressor
            masked_attn = out_attention * mask_expanded
            masked_dec2 = out_decoder2 * mask_expanded
            loss2 = self.loss_fn(masked_attn, masked_dec2)
            
            # Weighted combination (autoencoder loss is more important)
            loss_client = 0.6 * loss1 + 0.4 * loss2
            return v, loss_client
        else:
            return v

    def train_one_batch(self, loss, v, grad_back):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        grad_back = grad_back.to(v.device)
        v.backward(grad_back)
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.optimizer.zero_grad()

        

