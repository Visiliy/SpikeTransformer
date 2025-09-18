import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class LinearPerformerAttention(nn.Module):
    def __init__(self, dim, heads=8, feature_dim=256, dropout=0.1, causal=False):
        super().__init__()
        self.heads = heads
        self.feature_dim = feature_dim
        self.head_dim = dim // heads
        self.causal = causal

        self.proj_matrix = nn.Parameter(torch.randn(heads, self.head_dim, feature_dim))
        nn.init.orthogonal_(self.proj_matrix)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def _feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, x):
        b, n, d = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        q_proj = torch.einsum('bhnd,hdf->bhnf', q, self.proj_matrix)
        k_proj = torch.einsum('bhnd,hdf->bhnf', k, self.proj_matrix)

        q_proj = self._feature_map(q_proj)
        k_proj = self._feature_map(k_proj)

        if not self.causal:
            k_v = torch.einsum('bhnf,bhnd->bhfd', k_proj, v)
            attention_out = torch.einsum('bhnf,bhfd->bhnd', q_proj, k_v)

            k_proj_sum = k_proj.sum(dim=2)  # (b,h,f)
            z = 1.0 / (torch.einsum('bhnf,bhf->bhn', q_proj, k_proj_sum) + 1e-8)
            attention_out = attention_out * z.unsqueeze(-1)

        else:
            k_cum = k_proj.cumsum(dim=2)
            kv_cum = (k_proj.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(dim=2)

            attention_out = torch.einsum('bhnf,bhnfd->bhnd', q_proj, kv_cum)
            denom = torch.einsum('bhnf,bhnf->bhn', q_proj, k_cum).unsqueeze(-1)
            attention_out = attention_out / (denom + 1e-8)

        attention_out = attention_out.transpose(1, 2).reshape(b, n, -1)
        out = self.to_out(attention_out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, feature_dim=256, dropout=0.1, causal=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attention = LinearPerformerAttention(dim, heads, feature_dim, dropout, causal=causal)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, layers, vocab_size, heads=8, feature_dim=256, dropout=0.1, causal=True):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, feature_dim, dropout, causal=causal)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.to_logits(x)


if __name__ == "__main__":
    vocab_size = 600
    model = Transformer(dim=512, layers=20, heads=8, feature_dim=256, dropout=0.1, causal=True, vocab_size=vocab_size)

    data = torch.randn(3, 20, 512)
    out = model(data)
    print(out.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

