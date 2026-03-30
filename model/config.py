"""Configuration dataclasses for Voxtral-4B-TTS, parsed from params.json."""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class AcousticTransformerConfig:
    input_dim: int = 3072
    dim: int = 3072
    n_layers: int = 3
    head_dim: int = 128
    hidden_dim: int = 9216
    n_heads: int = 32
    n_kv_heads: int = 8
    use_biases: bool = False
    rope_theta: float = 10000.0
    sigma: float = 1e-5
    sigma_max: float = 1.0


@dataclass
class AudioEncodingConfig:
    codebook_pattern: str = "parallel"
    num_codebooks: int = 37
    sampling_rate: int = 24000
    frame_rate: float = 12.5


@dataclass
class AudioTokenizerConfig:
    channels: int = 1
    sampling_rate: int = 24000
    pretransform_patch_size: int = 240
    patch_proj_kernel_size: int = 7
    semantic_codebook_size: int = 8192
    semantic_dim: int = 256
    acoustic_codebook_size: int = 21
    acoustic_dim: int = 36
    conv_weight_norm: bool = True
    causal: bool = True
    attn_sliding_window_size: int = 16
    half_attn_window_upon_downsampling: bool = True
    dim: int = 1024
    hidden_dim: int = 4096
    head_dim: int = 128
    n_heads: int = 8
    n_kv_heads: int = 8
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    use_biases: bool = False
    norm_eps: float = 0.01
    layer_scale: bool = True
    layer_scale_init: float = 0.01
    decoder_transformer_lengths: list = field(default_factory=lambda: [2, 2, 2, 2])
    decoder_convs_kernels: list = field(default_factory=lambda: [3, 4, 4, 4])
    decoder_convs_strides: list = field(default_factory=lambda: [1, 2, 2, 2])
    voice: Dict[str, int] = field(default_factory=dict)


@dataclass
class AudioModelConfig:
    semantic_codebook_size: int = 8192
    acoustic_codebook_size: int = 21
    n_acoustic_codebook: int = 36
    audio_token_id: int = 24
    begin_audio_token_id: int = 25
    input_embedding_concat_type: str = "sum"
    p_uncond: float = 0.0
    condition_dropped_token_id: int = 42
    audio_encoding: AudioEncodingConfig = field(default_factory=AudioEncodingConfig)
    acoustic_transformer: AcousticTransformerConfig = field(default_factory=AcousticTransformerConfig)


@dataclass
class VoxtralConfig:
    dim: int = 3072
    n_layers: int = 26
    head_dim: int = 128
    hidden_dim: int = 9216
    n_heads: int = 32
    n_kv_heads: int = 8
    rope_theta: float = 1000000.0
    norm_eps: float = 1e-5
    vocab_size: int = 131072
    max_seq_len: int = 65536
    max_position_embeddings: int = 128000
    tied_embeddings: bool = True
    causal: bool = True
    use_biases: bool = False
    audio_model: AudioModelConfig = field(default_factory=AudioModelConfig)
    audio_tokenizer: AudioTokenizerConfig = field(default_factory=AudioTokenizerConfig)

    # Derived constants
    @property
    def n_codes_per_frame(self) -> int:
        return 1 + self.audio_model.n_acoustic_codebook  # 37

    @property
    def sample_rate(self) -> int:
        return self.audio_tokenizer.sampling_rate

    @property
    def frame_rate(self) -> float:
        return self.audio_model.audio_encoding.frame_rate


def load_config(model_dir: str) -> VoxtralConfig:
    """Load VoxtralConfig from a model directory containing params.json."""
    params_path = os.path.join(model_dir, "params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.json not found in {model_dir}")

    with open(params_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    mm = raw.get("multimodal", {})
    am = mm.get("audio_model_args", {})
    at = mm.get("audio_tokenizer_args", {})
    ae = am.get("audio_encoding_args", {})
    ac = am.get("acoustic_transformer_args", {})

    # Parse string lists in tokenizer config
    def _parse_int_list(s):
        if isinstance(s, str):
            return [int(x) for x in s.split(",")]
        return s if s else []

    audio_encoding = AudioEncodingConfig(
        codebook_pattern=ae.get("codebook_pattern", "parallel"),
        num_codebooks=ae.get("num_codebooks", 37),
        sampling_rate=ae.get("sampling_rate", 24000),
        frame_rate=ae.get("frame_rate", 12.5),
    )

    acoustic_transformer = AcousticTransformerConfig(
        input_dim=ac.get("input_dim", 3072),
        dim=ac.get("dim", 3072),
        n_layers=ac.get("n_layers", 3),
        head_dim=ac.get("head_dim", 128),
        hidden_dim=ac.get("hidden_dim", 9216),
        n_heads=ac.get("n_heads", 32),
        n_kv_heads=ac.get("n_kv_heads", 8),
        use_biases=ac.get("use_biases", False),
        rope_theta=ac.get("rope_theta", 10000.0),
        sigma=ac.get("sigma", 1e-5),
        sigma_max=ac.get("sigma_max", 1.0),
    )

    audio_model = AudioModelConfig(
        semantic_codebook_size=am.get("semantic_codebook_size", 8192),
        acoustic_codebook_size=am.get("acoustic_codebook_size", 21),
        n_acoustic_codebook=am.get("n_acoustic_codebook", 36),
        audio_token_id=am.get("audio_token_id", 24),
        begin_audio_token_id=am.get("begin_audio_token_id", 25),
        input_embedding_concat_type=am.get("input_embedding_concat_type", "sum"),
        p_uncond=am.get("p_uncond", 0.0),
        condition_dropped_token_id=am.get("condition_dropped_token_id", 42),
        audio_encoding=audio_encoding,
        acoustic_transformer=acoustic_transformer,
    )

    audio_tokenizer = AudioTokenizerConfig(
        channels=at.get("channels", 1),
        sampling_rate=at.get("sampling_rate", 24000),
        pretransform_patch_size=at.get("pretransform_patch_size", 240),
        patch_proj_kernel_size=at.get("patch_proj_kernel_size", 7),
        semantic_codebook_size=at.get("semantic_codebook_size", 8192),
        semantic_dim=at.get("semantic_dim", 256),
        acoustic_codebook_size=at.get("acoustic_codebook_size", 21),
        acoustic_dim=at.get("acoustic_dim", 36),
        conv_weight_norm=at.get("conv_weight_norm", True),
        causal=at.get("causal", True),
        attn_sliding_window_size=at.get("attn_sliding_window_size", 16),
        half_attn_window_upon_downsampling=at.get("half_attn_window_upon_downsampling", True),
        dim=at.get("dim", 1024),
        hidden_dim=at.get("hidden_dim", 4096),
        head_dim=at.get("head_dim", 128),
        n_heads=at.get("n_heads", 8),
        n_kv_heads=at.get("n_kv_heads", 8),
        qk_norm=at.get("qk_norm", True),
        qk_norm_eps=at.get("qk_norm_eps", 1e-6),
        use_biases=at.get("use_biases", False),
        norm_eps=at.get("norm_eps", 0.01),
        layer_scale=at.get("layer_scale", True),
        layer_scale_init=at.get("layer_scale_init", 0.01),
        decoder_transformer_lengths=_parse_int_list(at.get("decoder_transformer_lengths_str", "2,2,2,2")),
        decoder_convs_kernels=_parse_int_list(at.get("decoder_convs_kernels_str", "3,4,4,4")),
        decoder_convs_strides=_parse_int_list(at.get("decoder_convs_strides_str", "1,2,2,2")),
        voice=at.get("voice", {}),
    )

    return VoxtralConfig(
        dim=raw.get("dim", 3072),
        n_layers=raw.get("n_layers", 26),
        head_dim=raw.get("head_dim", 128),
        hidden_dim=raw.get("hidden_dim", 9216),
        n_heads=raw.get("n_heads", 32),
        n_kv_heads=raw.get("n_kv_heads", 8),
        rope_theta=raw.get("rope_theta", 1000000.0),
        norm_eps=raw.get("norm_eps", 1e-5),
        vocab_size=raw.get("vocab_size", 131072),
        max_seq_len=raw.get("max_seq_len", 65536),
        max_position_embeddings=raw.get("max_position_embeddings", 128000),
        tied_embeddings=raw.get("tied_embeddings", True),
        causal=raw.get("causal", True),
        use_biases=raw.get("use_biases", False),
        audio_model=audio_model,
        audio_tokenizer=audio_tokenizer,
    )
