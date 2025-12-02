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
    Visual–Contrast Attention (VCA) as a drop-in replacement for MHSA.
    Paper: Linear Differential Vision Transformer (NeurIPS 2025)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        contrast_pool_size: int = 8,  # h = w = contrast_pool_size, n = h*w
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

        # === Visual-Contrast Token Generation ===
        self.contrast_pool_size = contrast_pool_size
        self.n_contrast = contrast_pool_size * contrast_pool_size

        # Dual learnable positional embeddings (positive / negative)
        self.pos_embed_pos = nn.Parameter(torch.zeros(1, self.n_contrast, self.head_dim))
        self.pos_embed_neg = nn.Parameter(torch.zeros(1, self.n_contrast, self.head_dim))
        nn.init.normal_(self.pos_embed_pos, std=0.02)
        nn.init.normal_(self.pos_embed_neg, std=0.02)

        # Learnable lambda scalars (simplified: scalar instead of vector inner product)
        self.lambda1 = nn.Parameter(torch.tensor(init_lambda1))
        self.lambda2 = nn.Parameter(torch.tensor(init_lambda2))
        self.lambda1_init = init_lambda1
        self.lambda2_init = init_lambda2

        self.norm_stage1 = RMSNorm(self.head_dim)
        self.norm_stage2 = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) — N = H * W (if no cls token) or H*W+1 (with cls token)
            H, W: spatial resolution of patch grid (excluding cls token if present)
        Returns:
            (B, N, C)
        """
        B, N, C = x.shape
        has_cls = (N == H * W + 1)

        # Separate cls token if exists
        if has_cls:
            cls_token = x[:, :1, :]  # (B, 1, C)
            x_patch = x[:, 1:, :]    # (B, H*W, C)
            N_patch = H * W
        else:
            x_patch = x
            N_patch = N

        assert N_patch == H * W, f"Input spatial tokens {N_patch} != H*W={H*W}"

        # QKV projection and reshape
        qkv = self.qkv(x_patch).reshape(B, N_patch, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N_patch, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, num_heads, N_patch, head_dim)

        # === Stage I: Global Contrast ===
        # Reshape q to 2D: (B, num_heads, H, W, head_dim)
        q_2d = q.view(B, self.num_heads, H, W, self.head_dim)
        # AvgPool each head independently
        # Pooling stride = kernel_size to get (h, w) = (contrast_pool_size, contrast_pool_size)
        pool_stride = (H // self.contrast_pool_size, W // self.contrast_pool_size)
        # Use adaptive pooling to avoid dimension mismatch
        q_pooled = F.adaptive_avg_pool3d(
            q_2d.permute(0, 1, 4, 2, 3),  # (B, num_heads, head_dim, H, W)
            (self.head_dim, self.contrast_pool_size, self.contrast_pool_size)
        ).permute(0, 1, 3, 4, 2)  # (B, num_heads, h, w, head_dim)
        t_base = q_pooled.flatten(2, 3)  # (B, num_heads, n, head_dim)

        # Add dual positional embeddings (broadcast over batch and heads)
        t_pos = t_base + self.pos_embed_pos.unsqueeze(0)  # (B, num_heads, n, d)
        t_neg = t_base + self.pos_embed_neg.unsqueeze(0)

        # Positive stream attention: (B, num_heads, n, N) @ (B, num_heads, N, d) → (B, num_heads, n, d)
        attn_pos = (t_pos @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, n, N_patch)
        attn_pos = attn_pos.softmax(dim=-1)
        v_hat_pos = attn_pos @ v  # (B, num_heads, n, head_dim)

        # Negative stream
        attn_neg = (t_neg @ k.transpose(-2, -1)) * self.scale
        attn_neg = attn_neg.softmax(dim=-1)
        v_hat_neg = attn_neg @ v

        # Differential + RMSNorm
        v_contrast = v_hat_pos - self.lambda1 * v_hat_neg
        v_contrast = self.norm_stage1(v_contrast)  # (B, num_heads, n, head_dim)
        v_contrast = (1 - self.lambda1_init) * v_contrast

        # === Stage II: Patch-wise Differential Attention ===
        # Compute attention from original q to contrast tokens
        attn1 = (q @ t_pos.transpose(-2, -1)) * self.scale  # (B, num_heads, N_patch, n)
        attn1 = attn1.softmax(dim=-1)
        attn2 = (q @ t_neg.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)

        attn_out = attn1 - self.lambda2 * attn2  # (B, num_heads, N_patch, n)
        out_patch = attn_out @ v_contrast  # (B, num_heads, N_patch, head_dim)
        out_patch = self.norm_stage2(out_patch)
        out_patch = (1 - self.lambda2_init) * out_patch

        # Reshape back to (B, N_patch, C)
        out_patch = out_patch.transpose(1, 2).reshape(B, N_patch, C)
        out_patch = self.proj(out_patch)
        out_patch = self.proj_drop(out_patch)

        # Re-attach cls token if needed
        if has_cls:
            out = torch.cat([cls_token, out_patch], dim=1)
        else:
            out = out_patch

        return out