"""Audio codec decoder for Voxtral TTS.

Converts discrete semantic + acoustic codes into a 24kHz waveform.

Structure matches safetensors keys: decoder_blocks.{i}.*
  - Even blocks (0,2,4,6): Conv1d with weight_norm (decoder_blocks.{i}.conv.*)
  - Odd blocks (1,3,5,7): Transformer groups with 2 layers each (decoder_blocks.{i}.layers.{0,1}.*)

Pipeline:
  1. Dequantize codes: semantic lookup (first 8192 of audio codebook) projected to 256-dim,
     acoustic FSQ dequant to 36-dim, concat to 292-dim
  2. decoder_blocks.0: Conv1d 292 -> 1024 (input projection, stride=1)
  3. decoder_blocks.1: 2 transformer layers
  4. decoder_blocks.2: ConvTranspose1d 1024 -> 1024 (stride=2, 2x upsample)
  5. decoder_blocks.3: 2 transformer layers
  6. decoder_blocks.4: ConvTranspose1d (stride=2)
  7. decoder_blocks.5: 2 transformer layers
  8. decoder_blocks.6: ConvTranspose1d (stride=2)
  9. decoder_blocks.7: 2 transformer layers
  Total upsampling: 1 * 2 * 2 * 2 = 8x
  Final: reshape 1024-dim channels to waveform patches
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VoxtralConfig, AudioTokenizerConfig


# ---- Conv blocks (even-indexed decoder_blocks) ----

class ConvBlock(nn.Module):
    """Conv1d with weight_norm. Matches decoder_blocks.{i}.conv.* keys."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, transpose: bool = False, causal: bool = True):
        super().__init__()
        self.stride = stride
        self.transpose = transpose
        self.causal = causal
        self.kernel_size = kernel_size

        if transpose:
            raw_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                          stride=stride, bias=False)
        else:
            raw_conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                 stride=stride, bias=False)

        # Apply weight_norm to match safetensors parametrizations.weight.original0/1
        self.conv = nn.utils.parametrizations.weight_norm(raw_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        if self.transpose:
            y = self.conv(x)
            # Trim for causal alignment
            trim = self.kernel_size - self.stride
            if trim > 0 and self.causal:
                y = y[:, :, :-trim]
            return y
        else:
            # Causal left-padding
            if self.causal:
                pad = (self.kernel_size - 1)
                x = F.pad(x, (pad, 0))
            return self.conv(x)


# ---- Transformer blocks (odd-indexed decoder_blocks) ----

def get_alibi_slopes(n_heads: int):
    def _power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return _power_of_2(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    slopes = _power_of_2(closest)
    extra = _power_of_2(2 * closest)
    slopes.extend(extra[0::2][:n_heads - closest])
    return slopes


class CodecTransformerLayer(nn.Module):
    """Single transformer layer matching decoder_blocks.{i}.layers.{j}.* keys.

    Keys:
      attention.{wq,wk,wv,wo}.weight
      attention.{q_norm,k_norm}.weight
      attention_norm.weight
      attention_scale   (nn.Parameter, 1-D)
      feed_forward.{w1,w2,w3}.weight
      ffn_norm.weight
      ffn_scale         (nn.Parameter, 1-D)
    """

    def __init__(self, cfg: AudioTokenizerConfig, window_size: int):
        super().__init__()
        dim = cfg.dim

        # Attention
        self.attention = CodecAttention(cfg, window_size)
        self.attention_norm = nn.LayerNorm(dim, eps=cfg.norm_eps, elementwise_affine=True, bias=False)

        # Feed-forward
        self.feed_forward = CodecFFN(dim, cfg.hidden_dim, bias=cfg.use_biases)
        self.ffn_norm = nn.LayerNorm(dim, eps=cfg.norm_eps, elementwise_affine=True, bias=False)

        # Layer scale (direct parameters, not wrapped in a module)
        self.attention_scale = nn.Parameter(torch.full((dim,), cfg.layer_scale_init))
        self.ffn_scale = nn.Parameter(torch.full((dim,), cfg.layer_scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x)) * self.attention_scale
        x = x + self.feed_forward(self.ffn_norm(x)) * self.ffn_scale
        return x


class CodecAttention(nn.Module):
    """Causal attention with ALiBi and sliding window.

    Keys: wq, wk, wv, wo, q_norm, k_norm
    """

    def __init__(self, cfg: AudioTokenizerConfig, window_size: int):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.window_size = window_size

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=cfg.use_biases)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=cfg.use_biases)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=cfg.use_biases)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=cfg.use_biases)

        # QK norms operate on the full concatenated q/k (n_heads * head_dim = dim)
        self.q_norm = nn.LayerNorm(cfg.n_heads * cfg.head_dim, eps=cfg.qk_norm_eps, elementwise_affine=True, bias=False)
        self.k_norm = nn.LayerNorm(cfg.n_kv_heads * cfg.head_dim, eps=cfg.qk_norm_eps, elementwise_affine=True, bias=False)

        slopes = get_alibi_slopes(cfg.n_heads)
        self.register_buffer("alibi_slopes", torch.tensor(slopes, dtype=torch.float32))

    def _repeat_kv(self, x):
        if self.n_rep == 1:
            return x
        bs, nh, sl, hd = x.shape
        return x[:, :, None, :, :].expand(bs, nh, self.n_rep, sl, hd).reshape(bs, self.n_heads, sl, hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        q = self.q_norm(self.wq(x)).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.k_norm(self.wk(x)).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = self._repeat_kv(k.transpose(1, 2))
        v = self._repeat_kv(v.transpose(1, 2))

        # Causal + sliding window + ALiBi mask
        positions = torch.arange(seqlen, device=x.device)
        dist = positions.unsqueeze(0) - positions.unsqueeze(1)
        mask = torch.where(dist >= 0, 0.0, float("-inf"))
        if self.window_size > 0:
            mask = torch.where(dist <= self.window_size, mask,
                               torch.tensor(float("-inf"), device=x.device))

        alibi = self.alibi_slopes.view(1, -1, 1, 1) * dist.unsqueeze(0).unsqueeze(0).float()
        alibi = -alibi.abs()
        full_mask = (mask.unsqueeze(0).unsqueeze(0) + alibi).to(q.dtype)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=full_mask)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


class CodecFFN(nn.Module):
    def __init__(self, dim, hidden_dim, bias=False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerGroup(nn.Module):
    """Group of N transformer layers. Matches decoder_blocks.{i}.layers.{j}.* keys."""

    def __init__(self, cfg: AudioTokenizerConfig, n_layers: int, window_size: int):
        super().__init__()
        self.layers = nn.ModuleList([
            CodecTransformerLayer(cfg, window_size) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time) — transpose for transformers
        x = x.transpose(1, 2)  # (batch, time, channels)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)  # back to (batch, channels, time)


# ---- Dequantization ----

class AcousticFSQDequantizer(nn.Module):
    def __init__(self, levels: int = 21):
        super().__init__()
        self.levels = levels

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return codes.float() / (self.levels - 1) * 2.0 - 1.0


class SemanticCodebook(nn.Module):
    """EMA VQ semantic codebook.

    Stores embedding_sum and cluster_usage buffers.
    The actual codebook = embedding_sum / cluster_usage.unsqueeze(-1).
    Matches keys: quantizer.semantic_codebook.{embedding_sum, cluster_usage}
    """

    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        self.register_buffer("embedding_sum", torch.randn(codebook_size, dim))
        self.register_buffer("cluster_usage", torch.ones(codebook_size))

    def get_codebook(self) -> torch.Tensor:
        """Get codebook vectors.

        Try using embedding_sum directly first (some implementations store
        the actual codebook here). If cluster_usage values suggest EMA tracking,
        divide by usage to get centroids.
        """
        # If usage values are all ~1.0, embedding_sum IS the codebook
        # If usage values are large (>100), it's cumulative sums needing division
        max_usage = self.cluster_usage.max().item()
        if max_usage > 10.0:
            # EMA mode: divide by usage count
            usage = self.cluster_usage.clamp(min=1e-6).unsqueeze(-1)
            return self.embedding_sum / usage
        else:
            # Direct mode: embedding_sum IS the codebook
            return self.embedding_sum

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up codebook vectors by index.

        Args:
            indices: (batch, time) integer indices

        Returns:
            (batch, time, dim) codebook vectors
        """
        codebook = self.get_codebook()  # (codebook_size, dim)
        return F.embedding(indices, codebook)


class SemanticQuantizer(nn.Module):
    """Wrapper matching key path: quantizer.semantic_codebook.*"""

    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        self.semantic_codebook = SemanticCodebook(codebook_size, dim)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return self.semantic_codebook.decode(indices)


# ---- Full codec decoder ----

class CodecDecoder(nn.Module):
    """Codec decoder: codes -> waveform at 24kHz.

    Uses decoder_blocks ModuleList matching the safetensors structure:
      - decoder_blocks.0: ConvBlock (input, 292->1024, k=3, s=1)
      - decoder_blocks.1: TransformerGroup (2 layers)
      - decoder_blocks.2: ConvBlock (transpose, k=4, s=2)
      - decoder_blocks.3: TransformerGroup (2 layers)
      - decoder_blocks.4: ConvBlock (transpose, k=4, s=2)
      - decoder_blocks.5: TransformerGroup (2 layers)
      - decoder_blocks.6: ConvBlock (transpose, k=4, s=2)
      - decoder_blocks.7: TransformerGroup (2 layers)
    """

    def __init__(self, config: VoxtralConfig):
        super().__init__()
        cfg = config.audio_tokenizer
        self.config = config
        self.cfg = cfg

        # Dequantizer for acoustic codes
        self.acoustic_dequant = AcousticFSQDequantizer(cfg.acoustic_codebook_size)

        # Semantic codebook (EMA VQ-style): codebook derived from embedding_sum / cluster_usage
        # Keys: quantizer.semantic_codebook.{embedding_sum, cluster_usage}
        self.quantizer = SemanticQuantizer(cfg.semantic_codebook_size, cfg.semantic_dim)

        # Input dim = semantic_dim + acoustic_dim = 256 + 36 = 292
        input_dim = cfg.semantic_dim + cfg.acoustic_dim

        # Build decoder_blocks to match safetensors structure
        strides = cfg.decoder_convs_strides   # [1, 2, 2, 2]
        kernels = cfg.decoder_convs_kernels   # [3, 4, 4, 4]
        n_transformers = cfg.decoder_transformer_lengths  # [2, 2, 2, 2]
        window_size = cfg.attn_sliding_window_size  # 16

        blocks = []
        for i in range(len(strides)):
            # Conv block
            is_first = (i == 0)
            in_ch = input_dim if is_first else cfg.dim
            is_transpose = (strides[i] > 1)
            blocks.append(ConvBlock(
                in_ch, cfg.dim, kernels[i], strides[i],
                transpose=is_transpose, causal=cfg.causal
            ))

            # Transformer group
            blocks.append(TransformerGroup(cfg, n_transformers[i], window_size))

            # Halve window after upsampling
            if cfg.half_attn_window_upon_downsampling and strides[i] > 1:
                window_size = max(1, window_size // 2)

        self.decoder_blocks = nn.ModuleList(blocks)

        # Output projection: 1024 -> patch_size (240)
        # Uses weight_norm Conv1d matching checkpoint key: output_proj.conv.parametrizations.*
        self.output_proj = ConvBlock(cfg.dim, cfg.pretransform_patch_size,
                                     cfg.patch_proj_kernel_size,  # 7
                                     stride=1, transpose=False, causal=cfg.causal)

    def forward(self, semantic_codes: torch.Tensor, acoustic_codes: torch.Tensor,
                audio_embeddings=None) -> torch.Tensor:
        """Decode audio codes to waveform.

        Args:
            semantic_codes: (batch, time) semantic indices [0, 8191]
            acoustic_codes: (batch, time, 36) acoustic indices [0, 20]
            audio_embeddings: unused (kept for API compatibility)

        Returns:
            (batch, samples) waveform at 24kHz
        """
        param_dtype = next(self.parameters()).dtype

        # Dequantize acoustic codes -> (batch, time, 36) floats in [-1, 1]
        acoustic_vals = self.acoustic_dequant(acoustic_codes).to(param_dtype)

        # Semantic embedding via EMA VQ codebook -> (batch, time, 256)
        batch, time = semantic_codes.shape
        semantic_emb = self.quantizer.decode(semantic_codes).to(param_dtype)

        # Concatenate: (batch, time, 256+36=292)
        latent = torch.cat([semantic_emb, acoustic_vals], dim=-1)

        # Conv expects (batch, channels, time)
        x = latent.transpose(1, 2)

        # Run through decoder blocks
        for block in self.decoder_blocks:
            x = block(x)

        # Output projection: (batch, 1024, time*8) -> (batch, 240, time*8)
        x = self.output_proj(x)

        # Reshape to waveform: (batch, 240, T) -> (batch, 240*T)
        batch, patch_size, t = x.shape
        waveform = x.transpose(1, 2).contiguous().view(batch, -1)

        return waveform
