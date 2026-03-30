"""Flow-matching acoustic transformer for Voxtral TTS.

3-layer bidirectional transformer that generates 36 acoustic codes per frame
via 8 Euler ODE integration steps with classifier-free guidance (alpha=1.2).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import VoxtralConfig, AcousticTransformerConfig


class AcousticRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return ((x.float() * norm) * self.weight.float()).to(input_dtype)


class AcousticAttention(nn.Module):
    """Bidirectional attention (no RoPE, no causal mask) for acoustic transformer."""

    def __init__(self, cfg: AcousticTransformerConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=cfg.use_biases)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=cfg.use_biases)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=cfg.use_biases)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=cfg.use_biases)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        bs, seq, n_kv, hd = x.shape
        return x[:, :, :, None, :].expand(bs, seq, n_kv, self.n_rep, hd).reshape(bs, seq, self.n_heads, hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        k = self._repeat_kv(k.transpose(1, 2)).transpose(1, 2)
        v = self._repeat_kv(v.transpose(1, 2)).transpose(1, 2)

        # Bidirectional attention (no causal mask)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class AcousticFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AcousticTransformerBlock(nn.Module):
    def __init__(self, cfg: AcousticTransformerConfig):
        super().__init__()
        self.attention = AcousticAttention(cfg)
        self.feed_forward = AcousticFeedForward(cfg.dim, cfg.hidden_dim, cfg.use_biases)
        self.attention_norm = AcousticRMSNorm(cfg.dim, cfg.sigma)
        self.ffn_norm = AcousticRMSNorm(cfg.dim, cfg.sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class AcousticTransformer(nn.Module):
    """Flow-matching acoustic transformer.

    Takes LLM hidden states as conditioning and produces 36 acoustic codes
    per frame via iterative denoising with Euler ODE integration.
    """

    def __init__(self, config: VoxtralConfig):
        super().__init__()
        ac_cfg = config.audio_model.acoustic_transformer
        self.config = config
        self.ac_cfg = ac_cfg
        self.n_acoustic = config.audio_model.n_acoustic_codebook  # 36

        # Project noisy acoustic input (36-dim) to transformer dim (3072)
        # safetensors key: acoustic_transformer.input_projection
        self.input_proj = nn.Linear(self.n_acoustic, ac_cfg.dim, bias=False)

        # Project LLM hidden state (3072) to acoustic space (3072)
        # safetensors key: acoustic_transformer.llm_projection
        self.llm_projection = nn.Linear(ac_cfg.input_dim, ac_cfg.dim, bias=False)

        # Timestep projection (3072 -> 3072)
        # safetensors key: acoustic_transformer.time_projection
        self.time_projection = nn.Linear(ac_cfg.dim, ac_cfg.dim, bias=False)

        # Semantic codebook output (3072 -> 8192) — semantic prediction head
        # safetensors key: acoustic_transformer.semantic_codebook_output
        self.semantic_codebook_output = nn.Linear(
            ac_cfg.dim, config.audio_model.semantic_codebook_size, bias=False
        )

        # Transformer layers (bidirectional)
        self.layers = nn.ModuleList([
            AcousticTransformerBlock(ac_cfg) for _ in range(ac_cfg.n_layers)
        ])

        self.norm = AcousticRMSNorm(ac_cfg.dim, ac_cfg.sigma)

        # Output projection: transformer dim (3072) -> acoustic dimensions (36)
        # safetensors key: acoustic_transformer.acoustic_codebook_output
        self.output_proj = nn.Linear(ac_cfg.dim, self.n_acoustic, bias=False)

    def _timestep_embedding(self, t: torch.Tensor, dim: int, dtype: torch.dtype) -> torch.Tensor:
        """Sinusoidal timestep embedding (cosine-first, matching vLLM-Omni convention)."""
        half_dim = dim // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim))
        emb = t.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
        # vLLM-Omni convention: [cos, sin] (NOT [sin, cos])
        emb = torch.cat([emb.cos(), emb.sin()], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb.to(dtype)

    def predict_velocity(self, noisy_x: torch.Tensor, t: torch.Tensor,
                         conditioning: torch.Tensor) -> torch.Tensor:
        """Predict velocity field v_theta(x_t, t, h).

        Args:
            noisy_x: (batch, 1, n_acoustic) noisy acoustic values
            t: (batch,) timestep values in [0, 1]
            conditioning: (batch, 1, dim) LLM hidden state

        Returns:
            (batch, 1, n_acoustic) predicted velocity
        """
        param_dtype = self.input_proj.weight.dtype

        # Project noisy acoustic (36) -> dim (3072)
        x_proj = self.input_proj(noisy_x.to(param_dtype))  # (batch, 1, dim)

        # Project LLM conditioning through learned projection
        h = self.llm_projection(conditioning.to(param_dtype))  # (batch, 1, dim)

        # Timestep embedding -> projection
        t_emb = self._timestep_embedding(t, self.ac_cfg.dim, param_dtype)  # (batch, dim)
        t_emb = self.time_projection(t_emb).unsqueeze(1)  # (batch, 1, dim)

        # Concatenate as 3-token sequence (per vLLM-Omni reference)
        # Position 0: noisy acoustic, Position 1: timestep, Position 2: LLM conditioning
        combined = torch.cat([x_proj, t_emb, h], dim=1)  # (batch, 3, dim)

        # Bidirectional transformer layers (attend across all 3 positions)
        for layer in self.layers:
            combined = layer(combined)

        combined = self.norm(combined)
        # Read velocity from position 0 (noisy acoustic position)
        return self.output_proj(combined[:, :1, :])  # (batch, 1, n_acoustic)

    @torch.no_grad()
    def generate(self, llm_hidden: torch.Tensor, n_steps: int = 8,
                 cfg_alpha: float = 1.2) -> torch.Tensor:
        """Generate acoustic codes via flow matching with Euler ODE.

        Args:
            llm_hidden: (batch, 1, dim) hidden state from LLM
            n_steps: number of Euler integration steps (default 8)
            cfg_alpha: classifier-free guidance strength (default 1.2)

        Returns:
            (batch, 36) discrete acoustic codes in [0, 20]
        """
        batch = llm_hidden.shape[0]
        device = llm_hidden.device
        dtype = llm_hidden.dtype

        # Initialize from Gaussian noise
        x = torch.randn(batch, 1, self.n_acoustic, device=device, dtype=dtype)

        # Null conditioning for CFG
        null_cond = torch.zeros_like(llm_hidden)

        # vLLM-Omni: linspace(0, 1, n_steps) → n_steps-1 Euler intervals
        timesteps = torch.linspace(0, 1, n_steps, device=device, dtype=dtype)

        for i in range(n_steps - 1):
            t = timesteps[i].expand(batch)
            dt = timesteps[i + 1] - timesteps[i]

            # Conditional velocity
            v_cond = self.predict_velocity(x, t, llm_hidden)
            # Unconditional velocity
            v_uncond = self.predict_velocity(x, t, null_cond)
            # CFG interpolation
            v = cfg_alpha * v_cond + (1.0 - cfg_alpha) * v_uncond

            # Euler step (velocity ADDED, t goes 0→1)
            x = x + v * dt

        # FSQ quantization: clamp to [-1,1] → linear map to [0, levels-1] → round
        # C ref: NO tanh — just clamp then scale
        levels = self.config.audio_model.acoustic_codebook_size  # 21
        x_clamped = x.clamp(-1.0, 1.0)
        scaled = ((x_clamped + 1.0) / 2.0) * (levels - 1)
        codes = scaled.round().long().clamp(0, levels - 1)

        return codes.squeeze(1)  # (batch, 36)
