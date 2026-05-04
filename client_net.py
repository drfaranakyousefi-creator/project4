import torch
import torch.nn as nn
import torch.optim as optim
import math


# ─────────────────────────────────────────────
#  Layer Normalization
# ─────────────────────────────────────────────
class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))   # learnable scale
        self.bias  = nn.Parameter(torch.zeros(features))  # learnable shift

    def forward(self, x):
        # x: (batch, seq_len, features)
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1,  keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


# ─────────────────────────────────────────────
#  Positional Encoding
# ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                  # (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


# ─────────────────────────────────────────────
#  Residual Connection  (Pre-LN style)
# ─────────────────────────────────────────────
class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm    = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# ─────────────────────────────────────────────
#  Multi-Head Attention
# ─────────────────────────────────────────────
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_model = d_model
        self.h       = h
        self.d_k     = d_model // h

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout: nn.Dropout = None):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask_ = mask.to(device=attention_scores.device)
            mask_bool = (mask_ != 0) if mask_.dtype != torch.bool else mask_
            mask_bool = mask_bool.unsqueeze(1).unsqueeze(2)   # (b,1,1,seq)
            attention_scores = attention_scores.masked_fill(~mask_bool, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key   = self.w_k(k)
        value = self.w_v(v)

        # (b, seq, d_model) → (b, h, seq, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key   = key.view(  key.shape[0],   key.shape[1],   self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # (b, h, seq, d_k) → (b, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)


# ─────────────────────────────────────────────
#  ✅ Feed-Forward Block  (اضافه شد)
# ─────────────────────────────────────────────
class FeedForwardBlock(nn.Module):
    """
    Position-wise FFN استاندارد ترانسفورمر:
        Linear(d_model → d_ff) → ReLU → Dropout → Linear(d_ff → d_model)
    معمولاً d_ff = 4 * d_model انتخاب میشه.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.net(x)


# ─────────────────────────────────────────────
#  ✅ Transformer Encoder Block  (اضافه شد)
# ─────────────────────────────────────────────
class TransformerEncoderBlock(nn.Module):
    """
    یه بلاک کامل Encoder:
        x → [MHA + Residual & LN] → [FFN + Residual & LN] → output
    """

    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attention       = MultiHeadAttentionBlock(d_model, h, dropout)
        self.feed_forward    = FeedForwardBlock(d_model, d_ff, dropout)
        self.residual_attn   = ResidualConnection(d_model, dropout)
        self.residual_ff     = ResidualConnection(d_model, dropout)

    def forward(self, x, mask):
        # بخش اول: Multi-Head Attention + Residual
        x = self.residual_attn(x, lambda z: self.attention(z, z, z, mask))
        # بخش دوم: Feed Forward + Residual  ← این بخش قبلاً نبود
        x = self.residual_ff(x, self.feed_forward)
        return x


# ─────────────────────────────────────────────
#  AutoEncoder (تغییری نداشته)
# ─────────────────────────────────────────────
class encoder(nn.Module):
    def __init__(self, d_latent) -> None:
        super().__init__()
        self.fc1        = nn.Linear(1, 8)
        self.fc2        = nn.Linear(8, 16)
        self.fc3        = nn.Linear(16, d_latent)
        self.relu       = nn.ReLU()
        self.dropout    = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(8)
        self.layer_norm2 = nn.LayerNorm(16)

    def forward(self, x):
        x = self.relu(self.dropout(self.layer_norm1(self.fc1(x))))
        x = self.relu(self.dropout(self.layer_norm2(self.fc2(x))))
        return self.fc3(x)


class decoder(nn.Module):
    def __init__(self, d_latent) -> None:
        super().__init__()
        self.fc1        = nn.Linear(d_latent, 16)
        self.fc2        = nn.Linear(16, 8)
        self.fc3        = nn.Linear(8, 1)
        self.relu       = nn.ReLU()
        self.dropout    = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(16)
        self.layer_norm2 = nn.LayerNorm(8)

    def forward(self, x):
        x = self.relu(self.dropout(self.layer_norm1(self.fc1(x))))
        x = self.relu(self.dropout(self.layer_norm2(self.fc2(x))))
        return self.fc3(x)


class Auto_encoder(nn.Module):
    def __init__(self, d_latent) -> None:
        super().__init__()
        self.encoder = encoder(d_latent)
        self.decoder = decoder(d_latent)

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return enc, dec


class Multi_auto_encoder(nn.Module):
    def __init__(self, d_latent, N) -> None:
        super().__init__()
        self.auto_encoders = nn.ModuleList([Auto_encoder(d_latent) for _ in range(N)])

    def forward(self, x):
        b, seq_len, N = x.shape
        out_encoder, out_decoder = [], []
        for i, ae in enumerate(self.auto_encoders):
            enc, dec = ae(x[:, :, i].reshape(-1, 1))
            out_encoder.append(enc.reshape(b, seq_len, -1))
            out_decoder.append(dec.reshape(b, seq_len, -1))
        return (
            torch.cat(out_encoder, dim=-1),   # (b, seq, d_latent*N)
            torch.cat(out_decoder, dim=-1),   # (b, seq, N)
        )


# ─────────────────────────────────────────────
#  Compressor (تغییری نداشته)
# ─────────────────────────────────────────────
class compressor(nn.Module):
    """
    ورودی رو با attention pooling فشرده میکنه و
    یه بردار خلاصه v به سرور میفرسته.
    """

    def __init__(self, d_in, d_out) -> None:
        super().__init__()
        self.d_in  = d_in
        self.d_out = d_out

        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_out * 2),
            nn.LayerNorm(d_out * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_out * 2, d_out),
            nn.LayerNorm(d_out),
        )

        self.attention_pool = nn.Sequential(
            nn.Linear(d_in, d_in // 2),
            nn.Tanh(),
            nn.Linear(d_in // 2, 1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_out, d_out * 2),
            nn.LayerNorm(d_out * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_out * 2, d_in),
        )

        self.layer_norm = nn.LayerNorm(d_in)

    def forward(self, x, mask):
        # x: (b, seq_len, d_in) | mask: (b, seq_len)
        b, seq_len, _ = x.shape

        x = self.layer_norm(x)

        # محاسبه attention weights
        attn_scores = self.attention_pool(x.reshape(-1, self.d_in)).reshape(b, seq_len, 1)
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)   # (b, seq_len, 1)

        # encode همه timestepها
        encoded = self.encoder(x.reshape(-1, self.d_in)).reshape(b, seq_len, self.d_out)

        # attention-weighted pooling → بردار خلاصه
        v = (encoded * attn_weights).sum(dim=1)   # (b, d_out)

        # decode برای reconstruction loss
        v_exp     = v.unsqueeze(1).expand(b, seq_len, self.d_out)
        dec_out   = self.decoder(v_exp.reshape(-1, self.d_out)).reshape(b, seq_len, self.d_in)

        return v, dec_out


# ─────────────────────────────────────────────
#  ClientNetwork  ✅ با TransformerEncoderBlock کامل
# ─────────────────────────────────────────────
class ClientNetwork(nn.Module):

    def __init__(self, N, d_latent, h, dropout, seq_len, cap_in_dim, lr) -> None:
        super().__init__()

        d_model = d_latent * N          # بعد کامل بعد از multi-autoencoder
        d_ff    = d_model * 4           # استاندارد ترانسفورمر: 4 برابر d_model

        self.multi_autoEncoder = Multi_auto_encoder(d_latent, N)
        self.PE                = PositionalEncoding(d_model, seq_len, dropout)

        # ✅ TransformerEncoderBlock کامل (Attention + FFN) جایگزین کد قبلی
        self.transformer_block = TransformerEncoderBlock(d_model, h, d_ff, dropout)

        self.compressor = compressor(d_in=d_model, d_out=cap_in_dim)

        self.loss_fn   = nn.L1Loss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x, mask, train=True):
        mask = mask.to(x.device)

        # ── مرحله ۱: Multi AutoEncoder ──
        out_encoder, out_decoder1 = self.multi_autoEncoder(x)

        # ── مرحله ۲: Positional Encoding ──
        out_encoder = self.PE(out_encoder)

        # ── مرحله ۳: Transformer Encoder Block کامل (MHA + FFN) ──
        out_transformer = self.transformer_block(out_encoder, mask)

        # ── مرحله ۴: Compressor ──
        v, out_decoder2 = self.compressor(out_transformer, mask)

        if train:
            mask_exp = mask.unsqueeze(-1)

            # Reconstruction loss مرحله ۱: autoencoder
            loss1 = self.loss_fn(x * mask_exp, out_decoder1 * mask_exp)

            # Reconstruction loss مرحله ۲: compressor
            loss2 = self.loss_fn(out_transformer * mask_exp, out_decoder2 * mask_exp)

            loss_client = 0.6 * loss1 + 0.4 * loss2
            return v, loss_client
        else:
            return v

    def train_one_batch(self, loss, v, grad_back):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        grad_back = grad_back.to(v.device)
        v.backward(grad_back)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
