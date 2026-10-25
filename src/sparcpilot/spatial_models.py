"""Track B models: non-spatial MLP, CONCERT-informed attention-bias transformer, prediction heads.

All models obey the sparcmap_lib forward signature forward(model, centre, neighbours, rel).
Compartments are computed inside the model by nearest-centroid assignment in a PCA space whose
mean / components / centroids are fit on training slides only and stored as buffers.
"""
from __future__ import annotations
import math, numpy as np, torch
from torch import nn
import torch.nn.functional as F
from preprocessing.map_models import CoordEncoder  # import sparcmap_lib first (puts sparc_repo on sys.path)
from sklearn.cluster import KMeans


class CentreMLP(nn.Module):
    """Centre-patch-only MLP, ~2.2M params (baseline transformer ~2.8M)."""
    def __init__(self, input_dim, P, dims=(1024, 512, 256), dropout=0.1):
        super().__init__()
        layers, d = [], input_dim
        for h in dims:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]; d = h
        layers += [nn.Linear(d, P)]
        self.net = nn.Sequential(*layers)

    def forward(self, c, n=None, r=None):
        return self.net(c)


def mlp_forward(model, c, n, r):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model(c).float()


def bf16_forward(model, c, n, r):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model(c, n, r).float()


def fit_compartments(slides, k=6, n_pca=32, n_sub=60000, seed=0):
    """k-means(k) on PCA(n_pca) of H-optimus features pooled over the given (training) slides."""
    rng = np.random.default_rng(seed)
    X = np.concatenate([s['X'] for s in slides])
    sub = X[rng.choice(len(X), min(n_sub, len(X)), replace=False)]
    mu = sub.mean(0)
    U, S, Vt = np.linalg.svd(sub - mu, full_matrices=False)
    W = Vt[:n_pca].T.astype(np.float32)
    Z = (sub - mu) @ W
    km = KMeans(k, n_init=10, random_state=seed).fit(Z)
    return dict(mu=mu.astype(np.float32), W=W, C=km.cluster_centers_.astype(np.float32), k=k)


def assign_np(X, comp):
    Z = (X - comp['mu']) @ comp['W']
    return ((Z[:, None, :] - comp['C'][None]) ** 2).sum(-1).argmin(1)


class BiasBlock(nn.Module):
    """Pre-LN transformer block identical to the SPARC TransformerBlock but with an additive float
    attention bias (B, heads, T, T) passed to scaled_dot_product_attention."""
    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.h = heads; self.dropout = dropout
        self.norm1 = nn.LayerNorm(dim); self.qkv = nn.Linear(dim, 3 * dim); self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hd = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hd), nn.GELU(), nn.Dropout(dropout), nn.Linear(hd, dim), nn.Dropout(dropout))

    def forward(self, x, bias=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.norm2(x))


def rbf(x, centers, width):
    return torch.exp(-((x.unsqueeze(-1) - centers) / width) ** 2)


class ConcertTransformer(nn.Module):
    """SPARC-Map transformer + CONCERT-style additive attention biases from
    (1) physical distance between tokens (learned RBF expansion of |pos_i - pos_j|),
    (2) H&E feature cosine similarity between tokens (feature-centred; RBF expansion),
    (3) tissue compartment (k-means on H-optimus PCA): learned compartment-pair bias table + compartment
        embedding added to tokens.
    Biases are per layer and per head; final bias projections are zero-initialised so training starts
    from the distance-only baseline. The relative-coordinate MLP of the baseline is kept in all variants."""
    def __init__(self, input_dim, P, comp=None, use_dist=True, use_cos=True, use_comp=True,
                 hidden=256, layers=3, heads=8, dropout=0.1, n_rbf=16, max_dist=6.0):
        super().__init__()
        self.L, self.H = layers, heads
        self.use_dist, self.use_cos, self.use_comp = use_dist, use_cos, use_comp and comp is not None
        self.input_proj = nn.Linear(input_dim, hidden)
        self.coord_encoder = CoordEncoder(hidden)
        self.blocks = nn.ModuleList([BiasBlock(hidden, heads, dropout=dropout) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden // 2, P))
        if self.use_dist:
            self.register_buffer('d_cent', torch.linspace(0, max_dist, n_rbf))
            self.d_width = max_dist / (n_rbf - 1) * 1.5
            self.dist_proj = nn.Linear(n_rbf, layers * heads); nn.init.zeros_(self.dist_proj.weight); nn.init.zeros_(self.dist_proj.bias)
        if self.use_cos:
            self.register_buffer('c_cent', torch.linspace(-1, 1, 9))
            self.cos_proj = nn.Linear(9, layers * heads); nn.init.zeros_(self.cos_proj.weight); nn.init.zeros_(self.cos_proj.bias)
        if self.use_cos or self.use_comp:
            mu = comp['mu'] if comp is not None else np.zeros(input_dim, np.float32)
            self.register_buffer('feat_mu', torch.tensor(mu))
        if self.use_comp:
            K = comp['k']; self.K = K
            self.register_buffer('pca_W', torch.tensor(comp['W'])); self.register_buffer('cent', torch.tensor(comp['C']))
            self.pair = nn.Parameter(torch.zeros(K, K, layers * heads))
            self.comp_emb = nn.Embedding(K, hidden); nn.init.normal_(self.comp_emb.weight, std=0.02)

    @torch.no_grad()
    def compartments(self, feats):
        z = (feats.float() - self.feat_mu) @ self.pca_W
        return torch.cdist(z, self.cent.unsqueeze(0).expand(z.shape[0], -1, -1)).argmin(-1)

    def embed(self, c, n, r):
        B, K, _ = n.shape
        feats = torch.cat([c.unsqueeze(1), n], 1)
        tok = torch.cat([self.input_proj(c).unsqueeze(1), self.input_proj(n) + self.coord_encoder(r)], 1)
        bias = 0.
        if self.use_dist:
            pos = torch.cat([torch.zeros(B, 1, 2, device=r.device, dtype=r.dtype), r], 1)
            d = ((pos.unsqueeze(2) - pos.unsqueeze(1)).pow(2).sum(-1) + 1e-8).sqrt()
            bias = bias + self.dist_proj(rbf(d, self.d_cent, self.d_width))
        if self.use_cos:
            fn = F.normalize(feats - self.feat_mu, dim=-1)
            cs = fn @ fn.transpose(1, 2)
            bias = bias + self.cos_proj(rbf(cs, self.c_cent, 0.25))
        if self.use_comp:
            cid = self.compartments(feats)
            tok = tok + self.comp_emb(cid)
            oh = F.one_hot(cid, self.K).to(tok.dtype)
            bias = bias + torch.einsum('btk,bsm,kml->btsl', oh, oh, self.pair.to(tok.dtype))
        if torch.is_tensor(bias):
            T = tok.shape[1]
            bias = bias.view(B, T, T, self.L, self.H).permute(3, 0, 4, 1, 2).contiguous()
        for li, blk in enumerate(self.blocks):
            tok = blk(tok, bias[li] if torch.is_tensor(bias) else None)
        return tok[:, 0]

    def forward(self, c, n, r):
        return self.head(self.embed(c, n, r))


class GaussianHead(nn.Module):
    """mean + low-rank (rank R) + diagonal covariance over P programs: Sigma = F F^T + diag(d)."""
    def __init__(self, hidden, P, rank=4, dropout=0.1):
        super().__init__()
        self.P, self.R = P, rank
        self.trunk = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout))
        self.mean = nn.Linear(hidden // 2, P)
        self.fac = nn.Linear(hidden // 2, P * rank); self.logd = nn.Linear(hidden // 2, P)
        nn.init.normal_(self.fac.weight, std=1e-3); nn.init.zeros_(self.fac.bias)
        nn.init.zeros_(self.logd.weight); nn.init.constant_(self.logd.bias, -1.0)

    def forward(self, h):
        t = self.trunk(h)
        return self.mean(t), self.fac(t).view(-1, self.P, self.R), F.softplus(self.logd(t)) + 1e-3

    @staticmethod
    def nll(mu, Fm, d, y, m):
        """Exact Gaussian NLL with masked programs handled by inflating their diagonal (marginalisation is
        not exact for missing entries, but missing entries are <0.2% for mlm and 0% for ssgsea)."""
        # missing entries: zero factor rows and unit variance, so they add only a constant
        Fm = Fm * m.unsqueeze(-1); d = d * m + (1 - m); r = (y - mu) * m
        dist = torch.distributions.LowRankMultivariateNormal(torch.zeros_like(mu), Fm, d)
        return -dist.log_prob(r).mean() / mu.shape[1]


class TimeEmb(nn.Module):
    def __init__(self, dim=32):
        super().__init__(); self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * f[None] * 2 * math.pi
        return torch.cat([a.sin(), a.cos()], -1)


class FlowHead(nn.Module):
    """Conditional rectified-flow velocity field v(x_t, t | h) over the P-dim program vector (STFlow-inspired,
    simplified to independent spots). Source x0 ~ N(0, I) in robust-normalised target space; OT/rectified path
    x_t = (1-t) x0 + t y, target velocity y - x0; Euler integration at sampling time."""
    def __init__(self, hidden, P, width=512, temb=32):
        super().__init__()
        self.P = P; self.te = TimeEmb(temb)
        self.cond = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width))
        self.net = nn.Sequential(nn.Linear(P + temb + width, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, P))

    def v(self, x, t, hc):
        return self.net(torch.cat([x, self.te(t), hc], -1))

    def loss(self, h, y, m):
        B = y.shape[0]; hc = self.cond(h)
        x0 = torch.randn_like(y); t = torch.rand(B, device=y.device)
        xt = (1 - t[:, None]) * x0 + t[:, None] * y
        return (((self.v(xt, t, hc) - (y - x0)) ** 2) * m).sum() / m.sum().clamp(min=1)

    @torch.no_grad()
    def sample(self, h, n_samples=16, steps=32):
        B = h.shape[0]; hc = self.cond(h).repeat_interleave(n_samples, 0)
        x = torch.randn(B * n_samples, self.P, device=h.device); dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x.shape[0],), i * dt, device=h.device)
            x = x + dt * self.v(x, t, hc)
        return x.view(B, n_samples, self.P)
