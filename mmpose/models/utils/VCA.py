import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# -------------------------
# 辅助：RMSNorm（论文使用）
# -------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.norm(p=2, dim=-1, keepdim=True) / (x.size(-1) ** 0.5)
        x = x / (rms + self.eps)
        return x * self.weight


# -------------------------
# VCA 模块（即插即用）
# -------------------------
class VisualContrastAttention(nn.Module):
    """
    Visual–Contrast Attention (VCA) with *dynamic* e+/e- embeddings.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        contrast_pool_size: int = 8,
        init_lambda1: float = 0.5,
        init_lambda2: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.scale = qk_scale or self.head_dim ** -0.5

        # Standard QKV projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # === Dynamic e+/e- projection ===
        self.contrast_pool_size = contrast_pool_size
        self.n_contrast = contrast_pool_size * contrast_pool_size

        # Instead of static embeddings, use dynamic projections from global feature
        self.e_proj_pos = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.e_proj_neg = nn.Linear(self.head_dim, self.head_dim, bias=False)
        nn.init.normal_(self.e_proj_pos.weight, std=0.02)
        nn.init.normal_(self.e_proj_neg.weight, std=0.02)

        # Learnable lambda scalars (keep your simplified version)
        self.lambda1 = nn.Parameter(torch.tensor(init_lambda1))
        self.lambda2 = nn.Parameter(torch.tensor(init_lambda2))
        self.lambda1_init = init_lambda1
        self.lambda2_init = init_lambda2

        self.norm_stage1 = RMSNorm(self.head_dim)
        self.norm_stage2 = RMSNorm(self.head_dim)

        self.pool_conv = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1, groups=self.head_dim, bias=False),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        has_cls = (N == H * W + 1)

        if has_cls:
            cls_token = x[:, :1, :]
            x_patch = x[:, 1:, :]
            N_patch = H * W
        else:
            x_patch = x
            N_patch = N

        assert N_patch == H * W, f"Input spatial tokens {N_patch} != H*W={H*W}"

        # QKV
        qkv = self.qkv(x_patch).reshape(B, N_patch, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, M, N, d)

        # === Stage I: Global Contrast ===
        q_2d = q.view(B, self.num_heads, H, W, self.head_dim)
        q_2d_flat = q_2d.permute(0, 1, 4, 2, 3).contiguous()  # (B, M, d, H, W)
        q_2d_flat = q_2d_flat.view(B * self.num_heads, self.head_dim, H, W)

        # ✅ Replace with Learnable Pooling: DWConv + GELU + AdaptiveAvgPool
        # Add a small depth-wise conv to enrich spatial context before pooling
        q_enhanced = self.pool_conv(q_2d_flat)  # (B*M, d, H, W)
        q_pooled_flat = F.adaptive_avg_pool2d(
            q_enhanced,
            (self.contrast_pool_size, self.contrast_pool_size)
        )  # (B*M, d, h, w)

        q_pooled = q_pooled_flat.view(B, self.num_heads, self.head_dim, self.contrast_pool_size, self.contrast_pool_size)
        q_pooled = q_pooled.permute(0, 1, 3, 4, 2)  # (B, M, h, w, d)
        t_base = q_pooled.flatten(2, 3)  # (B, M, n, d)

        # === Dynamic e+ / e- generation (unchanged) ===
        global_q = q.mean(dim=2)  # (B, M, d)
        e_pos = self.e_proj_pos(global_q).unsqueeze(2)  # (B, M, 1, d)
        e_neg = self.e_proj_neg(global_q).unsqueeze(2)  # (B, M, 1, d)
        t_pos = t_base + e_pos
        t_neg = t_base + e_neg

        # === Stage I: Global Contrast ===
        attn_pos = (t_pos @ k.transpose(-2, -1)) * self.scale
        attn_pos = attn_pos.softmax(dim=-1)
        v_hat_pos = attn_pos @ v

        attn_neg = (t_neg @ k.transpose(-2, -1)) * self.scale
        attn_neg = attn_neg.softmax(dim=-1)
        v_hat_neg = attn_neg @ v

        v_contrast = v_hat_pos - self.lambda1 * v_hat_neg
        v_contrast = self.norm_stage1(v_contrast)
        v_contrast = (1 - self.lambda1_init) * v_contrast

        # === Stage II: Patch-wise Differential Attention ===
        attn1 = (q @ t_pos.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn2 = (q @ t_neg.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)

        attn_out = attn1 - self.lambda2 * attn2
        out_patch = attn_out @ v_contrast
        out_patch = self.norm_stage2(out_patch)
        out_patch = (1 - self.lambda2_init) * out_patch

        # Reshape and output
        out_patch = out_patch.transpose(1, 2).reshape(B, N_patch, C)
        out_patch = self.proj(out_patch)
        out_patch = self.proj_drop(out_patch)

        if has_cls:
            out = torch.cat([cls_token, out_patch], dim=1)
        else:
            out = out_patch

        return out