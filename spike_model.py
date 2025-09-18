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

            k_proj_sum = k_proj.sum(dim=2)
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


class SpikeAttention(nn.Module):

    def __init__(self, dim, spike_dim, heads=8, feature_dim=256, dropout=0.1, causal=False):
        super().__init__()

        self.heads = heads
        self.feature_dim = feature_dim
        self.head_dim = dim // heads
        self.causal = causal

        self.proj_matrix = nn.Parameter(torch.randn(heads, self.head_dim, feature_dim))
        nn.init.orthogonal_(self.proj_matrix)

        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        self.layer3 = nn.Linear(dim, dim)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

        self.spike_norm = nn.LayerNorm(spike_dim)
        self.ffn = nn.Sequential(
            nn.Linear(spike_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        )

        self.spike_matrix = nn.Parameter(torch.randn(10000, 1))
        self.spike_norm2 = nn.LayerNorm(dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def _feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, x, spike):
        b, n, d = x.shape
        spike = self.spike_norm(spike)
        spike = self.ffn(spike)
        spike = torch.matmul(self.spike_matrix[:n], spike)
        spike = self.spike_norm2(spike)

        q = self.layer1(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.layer2(spike).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.layer3(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)

        q_proj = torch.einsum('bhnd,hdf->bhnf', q, self.proj_matrix)
        k_proj = torch.einsum('bhnd,hdf->bhnf', k, self.proj_matrix)

        q_proj = self._feature_map(q_proj)
        k_proj = self._feature_map(k_proj)

        if not self.causal:
            k_v = torch.einsum('bhnf,bhnd->bhfd', k_proj, v)
            attention_out = torch.einsum('bhnf,bhfd->bhnd', q_proj, k_v)

            k_proj_sum = k_proj.sum(dim=2)
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


class PredictError(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.GELU(),
            nn.Linear(dim // 8, 1),
        )
        self.adapt1 = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        out = self.block(x).squeeze(-1)
        out = self.adapt1(out)
        return out


class SelfCorrection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lr = nn.Parameter(torch.tensor(0.01))
        self.error_to_grad = nn.Linear(1, dim)

    def forward(self, x, error):
        grad_approx = self.error_to_grad(error)
        grad_approx = grad_approx.unsqueeze(1).expand_as(x)
        x_corrected = x - self.lr * grad_approx
        return x_corrected


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, feature_dim=256, dropout=0.1, causal=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.attention = LinearPerformerAttention(dim, heads, feature_dim, dropout, causal=causal)
        self.spike_attention = SpikeAttention(dim, 10, heads, feature_dim, dropout, causal=causal)

        self.error = PredictError(dim)
        self.self_corr = SelfCorrection(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, spike):
        x = x + self.attention(self.norm1(x))
        x = x + self.spike_attention(self.norm2(x), spike)
        error = self.error(x)
        x = self.self_corr(x, error)
        x = x + self.ffn(self.norm3(x))
        return x, error


class Transformer(nn.Module):
    def __init__(self, dim, layers, vocab_size, heads=8, feature_dim=256, dropout=0.1, causal=True):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, feature_dim, dropout, causal=causal)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x, spike):
        for layer in self.layers:
            x, _ = layer(x, spike)
        x = self.norm(x)
        return self.to_logits(x)


if __name__ == "__main__":
    vocab_size = 50500
    model = Transformer(dim=512, layers=5, heads=8, feature_dim=256, dropout=0.1, causal=True, vocab_size=vocab_size)
    data = torch.randn(3, 20, 512)
    spike = torch.randn(3, 1, 10)
    out = model(data, spike)
    print(out.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)
