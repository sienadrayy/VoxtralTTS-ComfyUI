"""Mistral LLM backbone for Voxtral TTS.

26-layer decoder-only transformer with GQA, RoPE, and multi-codebook audio embeddings.
Autoregressively produces hidden states that are projected to semantic token logits.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import VoxtralConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return ((x.float() * norm) * self.weight.float()).to(input_dtype)


def precompute_freqs_cis(dim: int, end: int, theta: float = 1000000.0, device: torch.device = None):
    """Precompute RoPE complex exponentials."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary positional embeddings using interleaved pair convention.

    Mistral pairs adjacent dimensions: (0,1), (2,3), (4,5), ...
    NOT the half-rotate convention: (0,64), (1,65), ...

    Uses complex multiplication: (a + bi)(cos θ + i sin θ) = (a cos - b sin) + (a sin + b cos)i
    """
    # xq, xk: (batch, seq, n_heads, head_dim=128)
    # cos, sin: (seq, head_dim//2=64)
    def _rotate_interleaved(x, cos_t, sin_t):
        # Reshape last dim into pairs: (..., 128) → (..., 64, 2)
        x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        # Broadcast cos/sin: (seq, 64) → (1, seq, 1, 64)
        c = cos_t.unsqueeze(0).unsqueeze(2)
        s = sin_t.unsqueeze(0).unsqueeze(2)
        # Complex rotation per pair: (a, b) → (a*cos - b*sin, a*sin + b*cos)
        out = torch.stack([
            x_pairs[..., 0] * c - x_pairs[..., 1] * s,
            x_pairs[..., 0] * s + x_pairs[..., 1] * c,
        ], dim=-1).flatten(-2)
        return out.to(x.dtype)

    return _rotate_interleaved(xq, cos, sin), _rotate_interleaved(xk, cos, sin)


class Attention(nn.Module):
    def __init__(self, config: VoxtralConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=config.use_biases)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=config.use_biases)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=config.use_biases)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=config.use_biases)

        # KV cache (populated during generation)
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match query heads for GQA."""
        if self.n_rep == 1:
            return x
        bs, seq, n_kv, hd = x.shape
        return x[:, :, :, None, :].expand(bs, seq, n_kv, self.n_rep, hd).reshape(bs, seq, self.n_heads, hd)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                start_pos: int = 0, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Update KV cache — cast new k/v to match cache dtype to avoid mismatches
        # (RMSNorm upcasts to float32, but weights/cache may be bf16)
        if self.cache_k is not None:
            k = k.to(self.cache_k.dtype)
            v = v.to(self.cache_v.dtype)
            k = torch.cat([self.cache_k, k], dim=1)
            v = torch.cat([self.cache_v, v], dim=1)
        self.cache_k = k.detach()
        self.cache_v = v.detach()

        # GQA: repeat KV heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Transpose for attention: (bsz, n_heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Ensure all tensors share the same dtype for SDPA
        compute_dtype = q.dtype
        k = k.to(compute_dtype)
        v = v.to(compute_dtype)
        if mask is not None:
            mask = mask.to(compute_dtype)

        # Scaled dot-product attention
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None and seqlen > 1))
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)   # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)   # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: VoxtralConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config.dim, config.hidden_dim, config.use_biases)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                start_pos: int = 0, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), cos, sin, start_pos, mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def reset_cache(self):
        self.attention.reset_cache()


class AudioEmbeddings(nn.Module):
    """Flat multi-codebook audio token embeddings (MultiVocabEmbeddings).

    Layout of the [9088, 3072] embedding table:
      - Each codebook has 2 special tokens prepended: EMPTY_AUDIO(0), END_AUDIO(1)
      - Semantic codebook: indices [0, 8194) = 8192 codes + 2 specials
      - Acoustic codebook k: indices [8194 + k*23, 8194 + k*23 + 23)
        = 21 FSQ levels + 2 specials, stride = 23
      - Total: 8194 + 36*23 = 9022, padded to multiple of 128 = 9088

    All codes are shifted by +2 before lookup (to skip special tokens).
    At inference, the semantic code and all 36 acoustic codes are looked up and summed.
    """

    N_SPECIAL_TOKENS = 2  # EMPTY_AUDIO=0, END_AUDIO=1

    def __init__(self, config: VoxtralConfig):
        super().__init__()
        am = config.audio_model
        self.raw_semantic_size = am.semantic_codebook_size  # 8192
        self.raw_acoustic_size = am.acoustic_codebook_size  # 21
        self.n_acoustic = am.n_acoustic_codebook  # 36

        # Size of each codebook slot in the flat table (codes + special tokens)
        self.semantic_slot = self.raw_semantic_size + self.N_SPECIAL_TOKENS  # 8194
        self.acoustic_slot = self.raw_acoustic_size + self.N_SPECIAL_TOKENS  # 23

        # Total before padding
        total = self.semantic_slot + self.n_acoustic * self.acoustic_slot  # 9022
        # Pad to multiple of 128
        padded = 128 * ((total + 127) // 128)  # 9088
        self.embeddings = nn.Embedding(padded, config.dim)

    def forward(self, semantic_code: torch.Tensor, acoustic_codes: torch.Tensor) -> torch.Tensor:
        """Embed audio codes by summing across all codebooks.

        Args:
            semantic_code: (batch, 1) raw semantic indices [0, 8191]
            acoustic_codes: (batch, 36) raw acoustic indices [0, 20]

        Returns:
            (batch, 1, dim) summed embedding
        """
        # Shift semantic code by +2 to skip special tokens
        sem_idx = semantic_code + self.N_SPECIAL_TOKENS  # (batch, 1)
        emb = self.embeddings(sem_idx)  # (batch, 1, dim)

        # Acoustic embeddings: offset into flat table + shift by +2
        for i in range(self.n_acoustic):
            offset = self.semantic_slot + i * self.acoustic_slot + self.N_SPECIAL_TOKENS
            codes_i = acoustic_codes[:, i:i+1] + offset  # (batch, 1)
            emb = emb + self.embeddings(codes_i)

        return emb


class MistralLM(nn.Module):
    """Mistral LLM backbone with text + audio token support."""

    def __init__(self, config: VoxtralConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.audio_embeddings = AudioEmbeddings(config)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)

        # Semantic head is TIED with the audio embeddings:
        # logits = hidden @ audio_codebook_embeddings[:8192].T
        # No separate nn.Linear needed — we use the first 8192 rows of audio_embeddings.

    def forward_text(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Forward pass for text tokens. Returns hidden states."""
        h = self.tok_embeddings(tokens)
        return self._forward_layers(h, start_pos)

    def forward_audio_embed(self, audio_emb: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Forward pass for pre-embedded audio (voice embeddings). Returns hidden states."""
        return self._forward_layers(audio_emb, start_pos)

    def forward_audio_codes(self, semantic_code: torch.Tensor, acoustic_codes: torch.Tensor,
                            start_pos: int = 0) -> torch.Tensor:
        """Forward pass for generated audio codes. Returns hidden states."""
        h = self.audio_embeddings(semantic_code, acoustic_codes)
        return self._forward_layers(h, start_pos)

    def _forward_layers(self, h: torch.Tensor, start_pos: int) -> torch.Tensor:
        bsz, seqlen, _ = h.shape
        device = h.device

        # Get RoPE for the right positions
        cos, sin = precompute_freqs_cis(
            self.config.head_dim, start_pos + seqlen,
            self.config.rope_theta, device
        )
        cos = cos[start_pos:start_pos + seqlen]
        sin = sin[start_pos:start_pos + seqlen]

        # Build causal mask for prefill (seqlen > 1)
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            # Extend mask for KV cache
            if start_pos > 0:
                mask = torch.cat([torch.zeros(seqlen, start_pos, device=device), mask], dim=-1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, total_seq)

        for layer in self.layers:
            h = layer(h, cos, sin, start_pos, mask)

        return self.norm(h)

    def get_semantic_logits(self, hidden: torch.Tensor,
                            semantic_head: torch.nn.Linear = None) -> torch.Tensor:
        """Project hidden states to semantic codebook logits.

        Uses the acoustic transformer's semantic_codebook_output linear layer.
        Falls back to dot-product with audio embeddings if no head provided.
        """
        if semantic_head is not None:
            return semantic_head(hidden.to(semantic_head.weight.dtype)).float()
        # Fallback: dot product with audio codebook embeddings
        semantic_weight = self.audio_embeddings.embeddings.weight[:self.config.audio_model.semantic_codebook_size]
        h = hidden.to(semantic_weight.dtype)
        return torch.matmul(h, semantic_weight.T).float()

    def reset_cache(self):
        for layer in self.layers:
            layer.reset_cache()
