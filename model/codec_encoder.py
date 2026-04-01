"""Audio codec encoder for Voxtral TTS voice cloning.

Encodes a 24kHz waveform into discrete semantic + acoustic codes.

Architecture from the Voxtral paper (arxiv 2603.25551), mirroring the decoder:
  Input: 240-sample patches → causal Conv1d(k=7) → 1024-dim
  4 blocks: [Transformer(2 layers, ALiBi, sliding window) → strided Conv1d]
    Block 1: window=16, stride=2 (100Hz → 50Hz)
    Block 2: window=8,  stride=2 (50Hz → 25Hz)
    Block 3: window=4,  stride=2 (25Hz → 12.5Hz)
    Block 4: window=2,  stride=1, projects 1024 → 292
  Latent: 256-dim semantic (VQ) + 36-dim acoustic (tanh → FSQ)

Weights initialized from the decoder by transposing conv kernels (adjoint).
No separate encoder weights file required.
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .config import VoxtralConfig, AudioTokenizerConfig
from .codec_decoder import ConvBlock, TransformerGroup

logger = logging.getLogger("Voxtral")


class CodecEncoder(nn.Module):
    """Codec encoder: waveform at 24kHz → codes.

    Mirror of CodecDecoder. Uses strided convolutions for downsampling.
    Transformer blocks come BEFORE convolutions (opposite of decoder).
    """

    def __init__(self, config: VoxtralConfig):
        super().__init__()
        cfg = config.audio_tokenizer
        self.config = config
        self.cfg = cfg

        # Input projection: patch_size (240) → dim (1024), kernel=7, stride=1
        # Mirror of decoder's output_proj
        self.input_proj = ConvBlock(
            cfg.pretransform_patch_size, cfg.dim,
            cfg.patch_proj_kernel_size,  # 7
            stride=1, transpose=False, causal=cfg.causal
        )

        # 4 encoder blocks: Transformer → strided Conv
        # Strides: [2, 2, 2, 1] (8x total downsampling, last block just projects)
        # Window sizes: [16, 8, 4, 2] (halved at each downsampling)
        encoder_strides = [2, 2, 2, 1]
        encoder_kernels = [4, 4, 4, 3]  # matching decoder mirrors
        n_transformers = [2, 2, 2, 2]

        # Window sizes from paper: 16 → 8 → 4 → 2
        window_size = cfg.attn_sliding_window_size  # 16
        window_sizes = [window_size]
        for s in encoder_strides[:-1]:
            if cfg.half_attn_window_upon_downsampling and s > 1:
                window_size = max(1, window_size // 2)
            window_sizes.append(window_size)

        blocks = []
        for i in range(4):
            # Transformer group first (opposite of decoder)
            blocks.append(TransformerGroup(cfg, n_transformers[i], window_sizes[i]))

            # Strided convolution for downsampling
            is_last = (i == 3)
            out_ch = (cfg.semantic_dim + cfg.acoustic_dim) if is_last else cfg.dim
            blocks.append(ConvBlock(
                cfg.dim, out_ch, encoder_kernels[i], encoder_strides[i],
                transpose=False, causal=cfg.causal
            ))

        self.encoder_blocks = nn.ModuleList(blocks)

    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode waveform to codes.

        Args:
            waveform: (batch, samples) raw audio at 24kHz

        Returns:
            semantic_codes: (batch, time) indices in [0, 8191]
            acoustic_codes: (batch, time, 36) indices in [0, 20]
        """
        batch = waveform.shape[0]
        param_dtype = next(self.parameters()).dtype

        # Reshape waveform into patches: (batch, samples) → (batch, patch_size, n_patches)
        patch_size = self.cfg.pretransform_patch_size  # 240
        n_patches = waveform.shape[-1] // patch_size
        x = waveform[:, :n_patches * patch_size]
        x = x.reshape(batch, n_patches, patch_size).transpose(1, 2)  # (B, 240, T)
        x = x.to(param_dtype)

        # Input projection: (B, 240, T) → (B, 1024, T)
        x = self.input_proj(x)

        # Encoder blocks: [Transformer → Conv] × 4
        for block in self.encoder_blocks:
            x = block(x)

        # x: (B, 292, T_down) → split into semantic + acoustic
        x = x.transpose(1, 2)  # (B, T, 292)
        semantic_dim = self.cfg.semantic_dim  # 256
        semantic_features = x[:, :, :semantic_dim]   # (B, T, 256)
        acoustic_features = x[:, :, semantic_dim:]   # (B, T, 36)

        # Quantize semantic: VQ nearest codebook entry
        # (codebook loaded separately from decoder)
        semantic_codes = self._vq_encode(semantic_features)

        # Quantize acoustic: tanh → FSQ to 21 levels (per paper)
        acoustic_tanh = torch.tanh(acoustic_features)
        acoustic_scaled = (acoustic_tanh + 1.0) / 2.0 * (self.cfg.acoustic_codebook_size - 1)
        acoustic_codes = acoustic_scaled.round().long().clamp(0, self.cfg.acoustic_codebook_size - 1)

        return semantic_codes, acoustic_codes

    def _vq_encode(self, features: torch.Tensor) -> torch.Tensor:
        """Vector-quantize semantic features using nearest codebook lookup."""
        # codebook set by init_from_decoder()
        if not hasattr(self, '_semantic_codebook'):
            raise RuntimeError("Encoder codebook not initialized. Call init_from_decoder() first.")

        B, T, D = features.shape
        flat = features.reshape(-1, D).float()
        cb = self._semantic_codebook.float()

        # Efficient L2 nearest neighbor: ||a-b||² = ||a||² - 2a·b + ||b||²
        dists = (flat.pow(2).sum(-1, keepdim=True)
                 - 2.0 * flat @ cb.T
                 + cb.pow(2).sum(-1, keepdim=True).T)
        return dists.argmin(dim=-1).reshape(B, T)

    def init_from_decoder(self, codec_decoder: nn.Module):
        """Initialize encoder weights from decoder weights (adjoint transposition).

        The encoder mirrors the decoder architecture. We derive encoder weights by:
        - Transformer blocks: direct copy (same architecture)
        - Conv kernels: transpose dims 0,1 (adjoint of linear operator)
        - Weight norms: recompute from transposed directions
        """
        dec_sd = codec_decoder.state_dict()

        # --- Semantic codebook for VQ encoding ---
        cb_sum = dec_sd.get("quantizer.semantic_codebook.embedding_sum")
        cb_usage = dec_sd.get("quantizer.semantic_codebook.cluster_usage")
        if cb_sum is not None and cb_usage is not None:
            max_usage = cb_usage.max().item()
            if max_usage > 10.0:
                codebook = cb_sum / cb_usage.clamp(min=1e-6).unsqueeze(-1)
            else:
                codebook = cb_sum
            self.register_buffer('_semantic_codebook', codebook.clone())
            logger.info(f"Encoder: loaded semantic codebook ({codebook.shape[0]} entries)")

        # --- Input projection: encoder Conv(240→1024,k=7) from decoder output_proj Conv(1024→240,k=7) ---
        self._transpose_conv_weights(
            dec_sd, "output_proj.conv", self.input_proj.conv
        )

        # --- Encoder blocks from decoder blocks (reversed order) ---
        # Decoder layout: [Conv₀, Trans₁, ConvT₂, Trans₃, ConvT₄, Trans₅, ConvT₆, Trans₇]
        # Encoder layout: [Trans₀, Conv₁, Trans₂, Conv₃, Trans₄, Conv₅, Trans₆, Conv₇]
        #
        # Mapping (encoder ← decoder, reversed):
        #   Enc Trans₀ (block 0) ← Dec Trans₇ (decoder_blocks.7)  window=16
        #   Enc Conv₁  (block 1) ← Dec ConvT₆ (decoder_blocks.6)  stride=2
        #   Enc Trans₂ (block 2) ← Dec Trans₅ (decoder_blocks.5)  window=8
        #   Enc Conv₃  (block 3) ← Dec ConvT₄ (decoder_blocks.4)  stride=2
        #   Enc Trans₄ (block 4) ← Dec Trans₃ (decoder_blocks.3)  window=4
        #   Enc Conv₅  (block 5) ← Dec ConvT₂ (decoder_blocks.2)  stride=2
        #   Enc Trans₆ (block 6) ← Dec Trans₁ (decoder_blocks.1)  window=2
        #   Enc Conv₇  (block 7) ← Dec Conv₀  (decoder_blocks.0)  stride=1, 1024→292

        dec_block_map = [7, 6, 5, 4, 3, 2, 1, 0]  # encoder block i ← decoder block dec_block_map[i]

        for enc_i in range(8):
            dec_i = dec_block_map[enc_i]
            enc_block = self.encoder_blocks[enc_i]
            dec_prefix = f"decoder_blocks.{dec_i}."

            if enc_i % 2 == 0:
                # Transformer block: direct copy
                self._copy_transformer_weights(dec_sd, dec_prefix, enc_block)
            else:
                # Conv block: transpose weights
                self._transpose_conv_weights(
                    dec_sd, dec_prefix + "conv", enc_block.conv
                )

        logger.info("Encoder: initialized from decoder weights (adjoint transposition)")

    def _copy_transformer_weights(self, dec_sd: dict, dec_prefix: str,
                                   enc_transformer: TransformerGroup):
        """Copy transformer weights directly from decoder."""
        enc_sd = {}
        for key, val in dec_sd.items():
            if key.startswith(dec_prefix):
                enc_key = key[len(dec_prefix):]
                enc_sd[enc_key] = val.clone()
        if enc_sd:
            missing, unexpected = enc_transformer.load_state_dict(enc_sd, strict=False)
            if missing:
                logger.debug(f"Transformer copy: {len(missing)} missing keys")

    def _transpose_conv_weights(self, dec_sd: dict, dec_conv_prefix: str,
                                 enc_conv: nn.Module):
        """Transpose conv weights from decoder to encoder (adjoint).

        Weight norm stores:
          parametrizations.weight.original0 = g (norm, per output channel)
          parametrizations.weight.original1 = v (direction)

        For the adjoint: transpose dims 0,1 of v, recompute g.
        """
        v_key = f"{dec_conv_prefix}.parametrizations.weight.original1"
        g_key = f"{dec_conv_prefix}.parametrizations.weight.original0"

        if v_key not in dec_sd:
            logger.warning(f"Encoder init: missing {v_key}")
            return

        v_dec = dec_sd[v_key]  # (C_out_dec, C_in_dec, K)
        v_enc = v_dec.transpose(0, 1).contiguous().clone()  # (C_in_dec, C_out_dec, K)

        # Recompute norm per output channel
        g_enc = v_enc.reshape(v_enc.shape[0], -1).norm(dim=1, keepdim=True).unsqueeze(-1)

        # Load into encoder conv via state dict
        enc_sd = {
            "parametrizations.weight.original0": g_enc,
            "parametrizations.weight.original1": v_enc,
        }
        missing, unexpected = enc_conv.load_state_dict(enc_sd, strict=False, assign=True)
        if missing:
            logger.debug(f"Conv transpose: {len(missing)} missing for {dec_conv_prefix}")
