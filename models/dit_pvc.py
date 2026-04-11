"""
PointDiT: DiT-based backbone for 3D point cloud denoising.

Architecture overview
─────────────────────
  x (B, C, N)
      │
      ├─ GlobalPatchEncoder   PointNet over all N points → global patch feature g (B, d)
      │
  PointTokenizer          FPS → kNN grouping → mini-PointNet → K tokens
      │                   token_feats (B, K, d)  +  token_centers (B, K, 3)
      │ + pos_embed
      │
  DiTBlock × L            self-attention + FFN, both conditioned via adaLN-Zero
      │                   condition = t_emb + g  (timestep + global patch shape)
      │
  LayerNorm
      │
  PointDetokenizer        inverse-distance interpolation from K tokens → N points
      │                   + output MLP
      │
  out (B, out_dim, N)

Interface is identical to PVCNN2Unet:
    forward(x, t, x_cond=None) → (B, out_dim, N)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import pytorch3d.ops

from models.modules import Attention, SinusoidalPositionEmbeddings


# ─── Global Patch Encoder ────────────────────────────────────────────────────

class GlobalPatchEncoder(nn.Module):
    """
    Lightweight PointNet that summarises ALL N points in a patch into a single
    global feature vector of dimension token_dim.

    This gives every DiT block awareness of the overall patch shape, compensating
    for the fact that each of the K local tokens only sees M neighbours.

    Architecture: per-point shared MLP → max pool over N → projection
    """

    def __init__(self, in_dim: int, token_dim: int):
        super().__init__()
        mid = token_dim // 2
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, mid),
            nn.LayerNorm(mid),
            nn.GELU(),
            nn.Linear(mid, token_dim),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: (B, N, in_dim)
        Returns: (B, token_dim)  — one global vector per patch
        """
        per_point = self.encoder(feats)             # (B, N, token_dim)
        global_feat = per_point.max(dim=1).values   # (B, token_dim)  max pool
        return self.proj(global_feat)               # (B, token_dim)


# ─── Point Tokenizer ─────────────────────────────────────────────────────────

class PointTokenizer(nn.Module):
    """
    FPS + kNN grouping + mini-PointNet → K tokens.

    For each of the K FPS seed points, we group M nearest neighbors,
    compute relative coords, and apply a shared MLP followed by max-pooling
    to produce one d-dimensional token per seed.
    """

    def __init__(self, in_dim: int, token_dim: int, num_tokens: int, num_neighbors: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_neighbors = num_neighbors

        # mini-PointNet: (relative xyz 3) + (input features in_dim)
        encoder_in = 3 + in_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_in, token_dim // 2),
            nn.LayerNorm(token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, token_dim),
        )

    def forward(self, feats: torch.Tensor, coords: torch.Tensor):
        """
        feats:  (B, N, in_dim) — per-point input features
        coords: (B, N, 3)      — xyz coordinates for FPS / kNN

        Returns:
            token_feats:   (B, K, token_dim)
            token_centers: (B, K, 3)
        """
        B, N, _ = coords.shape
        K = self.num_tokens
        M = self.num_neighbors
        device = coords.device

        # ── FPS: pick K seed points ───────────────────────────────────────
        _, fps_idx = pytorch3d.ops.sample_farthest_points(coords, K=K)   # (B, K)
        b_range = torch.arange(B, device=device).unsqueeze(1)            # (B, 1)
        token_centers = coords[b_range, fps_idx]                         # (B, K, 3)

        # ── kNN: group M neighbors around each seed ────────────────────────
        _, knn_idx, _ = pytorch3d.ops.knn_points(
            token_centers, coords, K=M, return_nn=False
        )                                                                 # knn_idx: (B, K, M)

        # ── Gather neighbor coords and features ────────────────────────────
        knn_flat = knn_idx.reshape(B, -1)                                # (B, K*M)
        neighbor_coords = coords[b_range, knn_flat].reshape(B, K, M, 3)
        neighbor_feats = feats[b_range, knn_flat].reshape(B, K, M, -1)

        # ── Relative positions ─────────────────────────────────────────────
        rel_coords = neighbor_coords - token_centers.unsqueeze(2)        # (B, K, M, 3)

        # ── mini-PointNet: shared MLP + max pool ───────────────────────────
        patch_input = torch.cat([rel_coords, neighbor_feats], dim=-1)   # (B, K, M, 3+in_dim)
        patch_flat = patch_input.reshape(B * K, M, -1)
        token_feats = self.encoder(patch_flat).max(dim=1).values         # (B*K, token_dim)
        token_feats = token_feats.reshape(B, K, -1)                      # (B, K, token_dim)

        return token_feats, token_centers


# ─── AdaLN-Zero ──────────────────────────────────────────────────────────────

class AdaLNZero(nn.Module):
    """
    Adaptive LayerNorm-Zero conditioning from the DiT paper.

    Produces 6 modulation parameters per token dimension from the condition:
        (shift1, scale1, gate1)  →  attention sub-block
        (shift2, scale2, gate2)  →  FFN sub-block

    The projection is zero-initialized so that at training start each block
    acts as an identity, enabling stable learning from a pretrained (or random)
    initialization.
    """

    def __init__(self, token_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(token_dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * token_dim),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def modulate(self, x: torch.Tensor, cond: torch.Tensor):
        """
        x:    (B, K, token_dim)
        cond: (B, cond_dim)

        Returns:
            norm_fn: LayerNorm (no affine parameters — those are applied externally)
            chunks:  tuple of 6 tensors, each (B, 1, token_dim)
        """
        params = self.proj(cond).unsqueeze(1)    # (B, 1, 6*token_dim)
        chunks = params.chunk(6, dim=-1)         # 6 × (B, 1, token_dim)
        return self.norm, chunks


# ─── Feed-Forward Network ─────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, token_dim: int, expansion: int = 4):
        super().__init__()
        hidden = token_dim * expansion
        self.net = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, token_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── DiT Block ───────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """
    Single DiT transformer block with adaLN-Zero conditioning:

        x = x + gate1 * Attention(adaLN(x, t))
        x = x + gate2 * FFN(adaLN(x, t))
    """

    def __init__(self, token_dim: int, cond_dim: int, num_heads: int):
        super().__init__()
        assert token_dim % num_heads == 0, (
            f"token_dim ({token_dim}) must be divisible by num_heads ({num_heads})"
        )
        dim_head = token_dim // num_heads

        self.adaLN = AdaLNZero(token_dim, cond_dim)
        # norm=False: we apply normalization ourselves via adaLN
        self.attn = Attention(
            dim=token_dim,
            heads=num_heads,
            dim_head=dim_head,
            norm=False,
            flash=True,
        )
        self.ffn = FeedForward(token_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, K, token_dim)
        cond: (B, cond_dim)
        """
        norm, (shift1, scale1, gate1, shift2, scale2, gate2) = self.adaLN.modulate(x, cond)

        # Attention sub-block
        x_a = norm(x) * (1 + scale1) + shift1
        x = x + gate1 * self.attn(x_a)

        # FFN sub-block
        x_f = norm(x) * (1 + scale2) + shift2
        x = x + gate2 * self.ffn(x_f)

        return x


# ─── Point Detokenizer ───────────────────────────────────────────────────────

class PointDetokenizer(nn.Module):
    """
    Interpolates K token features back to N per-point features using
    inverse-distance weighting (PointNet++ style), then projects to output dim.
    """

    def __init__(self, token_dim: int, out_dim: int, k_interp: int = 3):
        super().__init__()
        self.k_interp = k_interp
        self.output_proj = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, out_dim),
        )

    def forward(
        self,
        token_feats: torch.Tensor,
        token_centers: torch.Tensor,
        query_pts: torch.Tensor,
    ) -> torch.Tensor:
        """
        token_feats:   (B, K, token_dim)
        token_centers: (B, K, 3)
        query_pts:     (B, N, 3)

        Returns: (B, N, out_dim)
        """
        B, N, _ = query_pts.shape
        k_interp = min(self.k_interp, token_centers.shape[1])

        # Squared distances to the k_interp nearest token centers
        dists_sq, knn_idx, _ = pytorch3d.ops.knn_points(
            query_pts, token_centers, K=k_interp, return_nn=False
        )                                                                # (B, N, k_interp)

        # 1/distance weights (PointNet++ style): w = 1 / sqrt(d²) = 1/d
        weights = 1.0 / (torch.sqrt(dists_sq) + 1e-6)                  # (B, N, k_interp)
        weights = weights / weights.sum(dim=-1, keepdim=True)           # normalize

        # Gather token features for the k_interp nearest tokens
        b_range = torch.arange(B, device=token_feats.device).unsqueeze(1)  # (B, 1)
        knn_flat = knn_idx.reshape(B, -1)                               # (B, N*k_interp)
        gathered = token_feats[b_range, knn_flat].reshape(B, N, k_interp, -1)  # (B, N, k_interp, d)

        # Weighted sum → interpolated features
        interp = (gathered * weights.unsqueeze(-1)).sum(dim=2)          # (B, N, token_dim)

        return self.output_proj(interp)                                 # (B, N, out_dim)


# ─── Full PointDiT Model ─────────────────────────────────────────────────────

class PointDiT(nn.Module):
    """
    DiT-based backbone for point cloud denoising in the P2P-Bridge framework.

    Drop-in replacement for PVCNN2Unet with an identical forward signature:
        forward(x, t, x_cond=None) → (B, out_dim, N)

    Required config keys under cfg.model.DiT:
        num_tokens    (int, default 128):  number of FPS token centers K
        token_dim     (int, default 384):  token / hidden dimension d
                                           must be divisible by num_heads
        num_neighbors (int, default 32):   kNN neighborhood size M
        num_layers    (int, default 6):    number of DiT blocks L
        num_heads     (int, default 6):    attention heads (token_dim // num_heads = dim_head)
        k_interp      (int, default 3):    neighbors used in detokenizer interpolation

    Other model config keys used:
        in_dim              (default 3)
        out_dim             (default 3)
        time_embed_dim      (default 64)
        extra_feature_channels (default 0)
    """

    def __init__(self, cfg: Dict):
        super().__init__()

        model_cfg = cfg.model
        dit_cfg = model_cfg.DiT

        self.in_dim = model_cfg.get("in_dim", 3)
        self.out_dim = model_cfg.get("out_dim", 3)
        self.embed_dim = model_cfg.get("time_embed_dim", 64)
        extra_feat_ch = model_cfg.get("extra_feature_channels", 0)

        num_tokens = dit_cfg.get("num_tokens", 128)
        token_dim = dit_cfg.get("token_dim", 384)
        num_neighbors = dit_cfg.get("num_neighbors", 32)
        num_layers = dit_cfg.get("num_layers", 6)
        num_heads = dit_cfg.get("num_heads", 6)
        k_interp = dit_cfg.get("k_interp", 3)

        # Effective feature channels (doubles when x_cond is concatenated)
        feat_dim = self.in_dim + extra_feat_ch

        # ── Global patch encoder ──────────────────────────────────────────────
        # Processes all N points → one global feature g (B, token_dim)
        # g is added to t_emb so every DiT block is conditioned on patch shape
        self.global_encoder = GlobalPatchEncoder(in_dim=feat_dim, token_dim=token_dim)

        # ── Timestep embedding ────────────────────────────────────────────────
        # SinusoidalPositionEmbeddings → (B, embed_dim), then project to token_dim
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(self.embed_dim),
            nn.Linear(self.embed_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )

        # ── Tokenizer ─────────────────────────────────────────────────────────
        self.tokenizer = PointTokenizer(
            in_dim=feat_dim,
            token_dim=token_dim,
            num_tokens=num_tokens,
            num_neighbors=num_neighbors,
        )

        # ── Positional encoding for token centers (3D xyz → token_dim) ───────
        self.pos_embed = nn.Sequential(
            nn.Linear(3, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )

        # ── DiT blocks ────────────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            DiTBlock(token_dim=token_dim, cond_dim=token_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        # ── Final layer norm ──────────────────────────────────────────────────
        self.final_norm = nn.LayerNorm(token_dim)

        # ── Detokenizer ───────────────────────────────────────────────────────
        self.detokenizer = PointDetokenizer(
            token_dim=token_dim,
            out_dim=self.out_dim,
            k_interp=k_interp,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:      (B, in_dim, N)       — noisy point cloud at current timestep
        t:      (B,)                 — noise levels (floats)
        x_cond: (B, in_dim, N) | None — optional conditioning (e.g. noisy input)

        Returns: (B, out_dim, N)
        """
        # Channel-last layout for per-point ops (before cond concat)
        feats = x.permute(0, 2, 1).contiguous()        # (B, N, in_dim)
        coords = feats[:, :, :3].contiguous()          # (B, N, 3) — first 3 channels are xyz

        # ── Global patch feature ──────────────────────────────────────────────
        # Computed from the original x only (before cond), matching in_dim
        global_feat = self.global_encoder(feats)                    # (B, d)

        # Append x_cond after global encoding to avoid in_dim mismatch
        if x_cond is not None:
            x = torch.cat([x, x_cond], dim=1)          # (B, 2*in_dim, N)
            feats = x.permute(0, 2, 1).contiguous()    # (B, N, 2*in_dim)

        # ── Tokenize ──────────────────────────────────────────────────────────
        token_feats, token_centers = self.tokenizer(feats, coords)  # (B, K, d), (B, K, 3)

        # ── Positional encoding ───────────────────────────────────────────────
        token_feats = token_feats + self.pos_embed(token_centers)   # (B, K, d)

        # ── Timestep + global conditioning ───────────────────────────────────
        # Combine timestep and global patch shape into one condition vector.
        # Every DiT block's adaLN-Zero then sees both "when" and "what shape".
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        t_emb = self.time_embed(t) + global_feat                    # (B, d)

        # ── DiT blocks ────────────────────────────────────────────────────────
        for block in self.blocks:
            token_feats = block(token_feats, t_emb)

        token_feats = self.final_norm(token_feats)                  # (B, K, d)

        # ── Detokenize ────────────────────────────────────────────────────────
        out = self.detokenizer(token_feats, token_centers, coords)  # (B, N, out_dim)

        return out.permute(0, 2, 1).contiguous()                    # (B, out_dim, N)
