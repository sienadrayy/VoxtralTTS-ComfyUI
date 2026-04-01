"""Full Voxtral TTS inference pipeline.

Loads model weights from consolidated.safetensors, constructs the prompt,
runs autoregressive generation, and decodes to a waveform.

Pipeline:
  1. Tokenize text with Tekken BPE
  2. Construct prompt: [BOS] [BEGIN_AUDIO] voice_emb... [INST_END] text_tokens [INST_START] [BEGIN_AUDIO]
  3. Prefill LLM on full prompt (hidden at last position = BEGIN_AUDIO)
  4. Autoregressive loop:
     a. LLM forward -> hidden_state
     b. Semantic logits -> greedy argmax -> semantic_code
     c. If semantic_code indicates end: break
     d. Acoustic transformer (flow matching) -> 36 acoustic codes
     e. Embed 37 codes (summed) -> next LLM input
     f. Collect codes
  5. Codec decode all collected frames -> 24kHz waveform
"""

import logging
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple

from safetensors.torch import load_file

from .config import VoxtralConfig, load_config
from .mistral_lm import MistralLM
from .acoustic_transformer import AcousticTransformer
from .codec_decoder import CodecDecoder

logger = logging.getLogger("Voxtral")

# Special token IDs
BOS_ID = 1
EOS_ID = 2
INST_START_ID = 3
INST_END_ID = 4
AUDIO_TOKEN_ID = 24
BEGIN_AUDIO_ID = 25

# Voice presets
VOICE_PRESETS = [
    "casual_female", "casual_male", "cheerful_female",
    "neutral_female", "neutral_male",
    "ar_male", "de_female", "de_male",
    "es_female", "es_male", "fr_female", "fr_male",
    "hi_female", "hi_male", "it_female", "it_male",
    "nl_female", "nl_male", "pt_female", "pt_male",
]


class VoxtralTTS:
    """Complete Voxtral TTS inference engine."""

    def __init__(self, model_dir: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.config: Optional[VoxtralConfig] = None
        self.llm: Optional[MistralLM] = None
        self.acoustic: Optional[AcousticTransformer] = None
        self.codec: Optional[CodecDecoder] = None
        self.encoder: Optional['CodecEncoder'] = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        """Load config, weights, and tokenizer."""
        if self._loaded:
            return

        logger.info(f"Loading Voxtral model from {self.model_dir}")

        # 1. Load config
        self.config = load_config(self.model_dir)
        logger.info(f"Config loaded: {self.config.n_layers} layers, dim={self.config.dim}")

        # 2. Build model components
        self.llm = MistralLM(self.config)
        self.acoustic = AcousticTransformer(self.config)
        self.codec = CodecDecoder(self.config)

        # 3. Load weights
        self._load_weights()

        # 4. Move to device
        self.llm = self.llm.to(self.device, self.dtype).eval()
        self.acoustic = self.acoustic.to(self.device, self.dtype).eval()
        self.codec = self.codec.to(self.device, self.dtype).eval()

        # 5. Load tokenizer
        self._load_tokenizer()

        self._loaded = True
        logger.info("Voxtral model loaded successfully")

    def _load_tokenizer(self):
        """Load MistralTokenizer for speech request encoding."""
        tekken_path = os.path.join(self.model_dir, "tekken.json")
        if not os.path.exists(tekken_path):
            raise FileNotFoundError(f"tekken.json not found in {self.model_dir}")

        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            self.tokenizer = MistralTokenizer.from_file(tekken_path)
            logger.info("MistralTokenizer loaded (supports encode_speech_request)")
        except ImportError:
            raise ImportError(
                "Cannot load tokenizer. Install mistral_common>=1.10.0: "
                "pip install mistral_common>=1.10.0"
            )

    def _load_weights(self):
        """Load consolidated.safetensors and distribute to model components."""
        weights_path = os.path.join(self.model_dir, "consolidated.safetensors")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"consolidated.safetensors not found in {self.model_dir}")

        logger.info("Loading weights from consolidated.safetensors...")
        state_dict = load_file(weights_path, device="cpu")

        logger.info(f"Loaded {len(state_dict)} weight tensors")

        # Map weights to model components
        llm_sd = {}
        acoustic_sd = {}
        codec_sd = {}
        unmapped = []

        for key, tensor in state_dict.items():
            mapped = self._map_weight(key, tensor, llm_sd, acoustic_sd, codec_sd)
            if not mapped:
                unmapped.append(key)

        if unmapped:
            logger.warning(f"{len(unmapped)} unmapped weight keys:")
            for k in unmapped:
                logger.warning(f"  UNMAPPED: {k} {state_dict[k].shape}")

        # For tied embeddings: if tok_embeddings is still missing, try output.weight
        if "tok_embeddings.weight" not in llm_sd and "output.weight" in llm_sd:
            logger.info("Using output.weight as tok_embeddings (tied embeddings)")
            llm_sd["tok_embeddings.weight"] = llm_sd["output.weight"]

        # Resize audio embeddings table if needed (safetensors may be smaller than pre-allocated)
        if "audio_embeddings.embeddings.weight" in llm_sd:
            actual_size = llm_sd["audio_embeddings.embeddings.weight"].shape[0]
            expected_size = self.llm.audio_embeddings.embeddings.num_embeddings
            if actual_size != expected_size:
                logger.info(f"Resizing audio embeddings: {expected_size} -> {actual_size}")
                dim = llm_sd["audio_embeddings.embeddings.weight"].shape[1]
                self.llm.audio_embeddings.embeddings = torch.nn.Embedding(actual_size, dim)

        # Resize acoustic semantic_codebook_output if needed
        if "semantic_codebook_output.weight" in acoustic_sd:
            actual_out = acoustic_sd["semantic_codebook_output.weight"].shape[0]
            expected_out = self.acoustic.semantic_codebook_output.out_features
            if actual_out != expected_out:
                logger.info(f"Resizing semantic_codebook_output: {expected_out} -> {actual_out}")
                in_f = self.acoustic.semantic_codebook_output.in_features
                self.acoustic.semantic_codebook_output = torch.nn.Linear(in_f, actual_out, bias=False)

        # Load state dicts with strict=False to allow partial loading
        self._load_partial(self.llm, llm_sd, "LLM")
        self._load_partial(self.acoustic, acoustic_sd, "Acoustic")
        self._load_partial(self.codec, codec_sd, "Codec")

    def _load_partial(self, module: torch.nn.Module, state_dict: dict, name: str):
        """Load state dict with helpful error reporting."""
        if not state_dict:
            logger.warning(f"No weights mapped for {name}!")
            return

        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"{name}: {len(missing)} missing keys")
            for k in missing[:20]:
                logger.warning(f"  MISSING: {k}")
        if unexpected:
            logger.warning(f"{name}: {len(unexpected)} unexpected keys")
            for k in unexpected[:20]:
                logger.warning(f"  UNEXPECTED: {k}")
        loaded = len(state_dict) - len(unexpected)
        logger.info(f"{name}: loaded {loaded}/{len(state_dict)} tensors ({len(missing)} missing)")

    def _map_weight(self, key: str, tensor: torch.Tensor,
                    llm_sd: dict, acoustic_sd: dict, codec_sd: dict) -> bool:
        """Route a safetensors weight key to the correct model component.

        Actual key patterns from consolidated.safetensors:
          LLM backbone:
            layers.{0-25}.attention.{wq,wk,wv,wo}.weight
            layers.{0-25}.feed_forward.{w1,w2,w3}.weight
            layers.{0-25}.{attention_norm,ffn_norm}.weight
            norm.weight
            tok_embeddings.weight  (may or may not exist)
            output.weight          (may or may not exist)

          Acoustic transformer:
            acoustic_transformer.layers.{0-2}.attention.{wq,wk,wv,wo}.weight
            acoustic_transformer.layers.{0-2}.feed_forward.{w1,w2,w3}.weight
            acoustic_transformer.layers.{0-2}.{attention_norm,ffn_norm}.weight
            acoustic_transformer.input_projection.weight
            acoustic_transformer.acoustic_codebook_output.weight
            acoustic_transformer.norm.weight (if present)
            acoustic_transformer.output_projection.weight (if present)

          Codec decoder:
            decoder_blocks.{0..N}.conv.* (convolution layers)
            decoder_blocks.{1..N}.layers.{0..1}.attention.{wq,wk,wv,wo}.weight
            decoder_blocks.{1..N}.layers.{0..1}.attention.{q_norm,k_norm}.weight
            decoder_blocks.{1..N}.layers.{0..1}.{attention_norm,ffn_norm}.weight
            decoder_blocks.{1..N}.layers.{0..1}.{attention_scale,ffn_scale}
            decoder_blocks.{1..N}.layers.{0..1}.feed_forward.{w1,w2,w3}.weight
            + semantic codebook, patch projection, output conv, etc.
        """
        # --- Multimodal audio embeddings (text tok_embeddings + audio codebook) ---
        if key == "mm_audio_embeddings.tok_embeddings.weight":
            llm_sd["tok_embeddings.weight"] = tensor
            return True

        if key == "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight":
            llm_sd["audio_embeddings.embeddings.weight"] = tensor
            return True

        # --- LLM backbone ---
        if key == "tok_embeddings.weight":
            llm_sd["tok_embeddings.weight"] = tensor
            return True

        if key == "output.weight":
            llm_sd["output.weight"] = tensor
            return True

        if key == "norm.weight":
            llm_sd["norm.weight"] = tensor
            return True

        if key.startswith("layers."):
            parts = key.split(".")
            if len(parts) >= 3:
                layer_idx = int(parts[1])
                if layer_idx < 26:
                    llm_sd[key] = tensor
                    return True
            return False

        # --- Acoustic transformer ---
        if key.startswith("acoustic_transformer."):
            remaining = key[len("acoustic_transformer."):]

            # Map safetensors names to our model names
            name_map = {
                "acoustic_codebook_output.weight": "output_proj.weight",
                "input_projection.weight": "input_proj.weight",
            }
            if remaining in name_map:
                acoustic_sd[name_map[remaining]] = tensor
                return True

            # Everything else maps directly (layers, norm, llm_projection,
            # time_projection, semantic_codebook_output, etc.)
            acoustic_sd[remaining] = tensor
            return True

        # --- Codec decoder ---
        # Keys come as "audio_tokenizer.decoder_blocks.{i}.*" — strip prefix
        if key.startswith("audio_tokenizer."):
            remaining = key[len("audio_tokenizer."):]
            codec_sd[remaining] = tensor
            return True

        if key.startswith("decoder_blocks."):
            codec_sd[key] = tensor
            return True

        # Other codec-related top-level keys
        codec_prefixes = (
            "semantic_codebook.", "acoustic_codebook.",
            "patch_proj.", "output_proj.", "input_proj.",
            "codec.",
        )
        for prefix in codec_prefixes:
            if key.startswith(prefix):
                codec_sd[key] = tensor
                return True

        # --- Audio embeddings (may be at top level) ---
        audio_emb_prefixes = (
            "audio_embeddings.", "multimodal_embeddings.",
            "semantic_embedding.", "acoustic_embedding.",
        )
        for prefix in audio_emb_prefixes:
            if key.startswith(prefix):
                llm_sd[key] = tensor
                return True

        return False

    def _encode_speech_request(self, text: str, voice: str) -> List[int]:
        """Encode a speech request using mistral_common's official format.

        Returns the full token sequence including BOS, BEGIN_AUDIO,
        audio placeholders, separators, text tokens, and final BEGIN_AUDIO.
        """
        from mistral_common.protocol.speech.request import SpeechRequest

        request = SpeechRequest(input=text, voice=voice)
        result = self.tokenizer.encode_speech_request(request)
        return result.tokens

    def _encode_speech_request_with_frames(self, text: str, n_frames: int) -> List[int]:
        """Build a speech request token sequence for a given number of voice frames.

        Used for voice cloning where we have a pre-computed voice embedding
        instead of a named voice preset.

        The token layout matches the official format:
        [BOS, BEGIN_AUDIO, AUDIO_TOKEN * (n_frames + 1), TEXT_TO_AUDIO, text_tokens, AUDIO_TO_TEXT, BEGIN_AUDIO]
        """
        from mistral_common.protocol.speech.request import SpeechRequest

        # Generate reference audio bytes of the right length to get n_frames tokens
        # n_frames + 1 because the tokenizer adds +1 for END_OUTPUT_AUDIO
        # frame_rate = 12.5 Hz, so audio_length = n_frames / 12.5 * 24000 samples
        import struct
        n_samples = int(n_frames / self.config.frame_rate * self.config.sample_rate)
        # Create minimal WAV bytes
        fake_audio_bytes = self._make_wav_bytes(n_samples, self.config.sample_rate)
        request = SpeechRequest(input=text, ref_audio=fake_audio_bytes)
        result = self.tokenizer.encode_speech_request(request)
        return result.tokens

    @staticmethod
    def _make_wav_bytes(n_samples: int, sample_rate: int) -> bytes:
        """Create minimal WAV file bytes with silence (for tokenizer frame counting)."""
        import struct
        import io

        # WAV header for 16-bit mono PCM
        data_size = n_samples * 2  # 16-bit = 2 bytes per sample
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF', 36 + data_size, b'WAVE',
            b'fmt ', 16, 1, 1,  # PCM, mono
            sample_rate, sample_rate * 2, 2, 16,  # byte rate, block align, bits
            b'data', data_size,
        )
        return header + b'\x00' * data_size

    def load_voice_embedding(self, voice_name: str) -> torch.Tensor:
        """Load a pre-computed voice embedding.

        Args:
            voice_name: name of voice preset (e.g., "casual_male")

        Returns:
            (N, dim) tensor of voice embeddings in BF16
        """
        voice_path = os.path.join(self.model_dir, "voice_embedding", f"{voice_name}.pt")
        if not os.path.exists(voice_path):
            raise FileNotFoundError(f"Voice embedding not found: {voice_path}")

        voice_emb = torch.load(voice_path, map_location="cpu", weights_only=True)
        logger.info(f"Loaded voice embedding '{voice_name}': shape={voice_emb.shape}")
        return voice_emb.to(self.device, self.dtype)

    def get_available_voices(self) -> List[str]:
        """List available voice presets in the model directory."""
        voice_dir = os.path.join(self.model_dir, "voice_embedding")
        if not os.path.exists(voice_dir):
            return VOICE_PRESETS  # Return defaults even if dir missing

        voices = []
        for f in sorted(os.listdir(voice_dir)):
            if f.endswith(".pt"):
                voices.append(f[:-3])
        return voices if voices else VOICE_PRESETS

    def get_encoder(self):
        """Lazily build the codec encoder from decoder weights."""
        if self.encoder is not None:
            return self.encoder

        from .codec_encoder import CodecEncoder

        logger.info("Building codec encoder from decoder weights...")
        self.encoder = CodecEncoder(self.config)
        self.encoder.init_from_decoder(self.codec)
        self.encoder = self.encoder.to(self.device, self.dtype).eval()
        logger.info("Codec encoder ready")
        return self.encoder

    def encode_audio(self, waveform_24k: torch.Tensor) -> tuple:
        """Encode 24kHz audio to codec codes using the encoder.

        Args:
            waveform_24k: (samples,) mono waveform at 24kHz

        Returns:
            semantic_codes: (1, time) indices
            acoustic_codes: (1, time, 36) indices
        """
        encoder = self.get_encoder()
        waveform_batch = waveform_24k.unsqueeze(0).to(self.device)
        with torch.no_grad():
            semantic_codes, acoustic_codes = encoder(waveform_batch)
        return semantic_codes, acoustic_codes

    def compute_voice_embedding(self, semantic_codes: 'torch.Tensor',
                                acoustic_codes: 'torch.Tensor') -> 'torch.Tensor':
        """Compute voice embedding from codec codes using LLM's audio codebook.

        Args:
            semantic_codes: (batch, time) semantic indices [0, 8191]
            acoustic_codes: (batch, time, 36) acoustic indices [0, 20]

        Returns:
            (time, dim) voice embedding tensor ready for LLM input
        """
        n_frames = semantic_codes.shape[1]
        emb_list = []
        for t in range(n_frames):
            sc = semantic_codes[:, t:t+1]  # (1, 1)
            ac = acoustic_codes[:, t, :]   # (1, 36)
            frame_emb = self.llm.audio_embeddings(sc, ac)  # (1, 1, dim)
            emb_list.append(frame_emb.squeeze(0).squeeze(0))
        return torch.stack(emb_list)  # (time, dim)

    @torch.no_grad()
    def generate(self, text: str, voice: str = "casual_male",
                 voice_embedding: Optional[torch.Tensor] = None,
                 max_frames: int = 1500, n_flow_steps: int = 8,
                 cfg_alpha: float = 1.2, temperature: float = 0.0,
                 top_p: float = 1.0, top_k: int = 0,
                 repetition_penalty: float = 1.1) -> Tuple[np.ndarray, int]:
        """Generate speech from text.

        Args:
            text: input text to synthesize
            voice: voice preset name (ignored if voice_embedding is provided)
            voice_embedding: optional pre-computed (N, dim) voice embedding tensor;
                if provided, bypasses voice preset loading
            max_frames: maximum number of audio frames (~80ms each, 1500 = ~2 min)
            n_flow_steps: number of Euler integration steps for flow matching
            cfg_alpha: classifier-free guidance strength
            temperature: sampling temperature (0 = greedy/official default)
            top_p: nucleus sampling threshold (1.0 = disabled/official default)
            top_k: top-k sampling (0 = disabled/official default)
            repetition_penalty: penalize repeated semantic codes (1.1 = official default)

        Returns:
            (waveform_numpy, sample_rate) tuple
        """
        if not self._loaded:
            self.load()

        # Reset KV caches
        self.llm.reset_cache()

        # Determine voice embedding source
        if voice_embedding is not None:
            voice_emb = voice_embedding.to(self.device, self.dtype)
            n_voice_frames = voice_emb.shape[0]
            # Use ref_audio path in tokenizer: create right number of AUDIO_TOKEN placeholders
            prompt_token_ids = self._encode_speech_request_with_frames(text, n_voice_frames)
        else:
            prompt_token_ids = self._encode_speech_request(text, voice)
            voice_emb = self.load_voice_embedding(voice)
            n_voice_frames = voice_emb.shape[0]

        logger.info(f"Prompt: {len(prompt_token_ids)} tokens")
        logger.info(f"Prompt tokens (first 10): {prompt_token_ids[:10]}")
        logger.info(f"Prompt tokens (last 10): {prompt_token_ids[-10:]}")
        logger.info(f"Voice embedding: {n_voice_frames} frames")

        # 3. Build input embeddings for prefill
        prompt_ids = torch.tensor([prompt_token_ids], device=self.device, dtype=torch.long)
        all_emb = self.llm.tok_embeddings(prompt_ids)  # (1, seq, dim)

        # 4. Replace AUDIO_TOKEN placeholder positions with voice embeddings
        #    Find positions of AUDIO_TOKEN_ID (24) in the prompt
        audio_positions = [i for i, t in enumerate(prompt_token_ids) if t == AUDIO_TOKEN_ID]
        if len(audio_positions) != n_voice_frames:
            logger.warning(f"Voice frames ({n_voice_frames}) != audio placeholders ({len(audio_positions)})")
            # Use the smaller count
            n_replace = min(len(audio_positions), n_voice_frames)
        else:
            n_replace = n_voice_frames

        for i in range(n_replace):
            all_emb[0, audio_positions[i]] = voice_emb[i]

        # 5. Prefill: run LLM on full prompt
        hidden = self.llm.forward_audio_embed(all_emb, start_pos=0)
        last_hidden = hidden[:, -1:, :]  # (1, 1, dim) — hidden at last position (BEGIN_AUDIO=25)
        pos = len(prompt_token_ids)

        # 6. Autoregressive generation loop
        all_semantic_codes = []
        all_acoustic_codes = []

        for frame_idx in range(max_frames):
            # a. Get semantic logits from LLM hidden state
            #    Direct linear projection: semantic_codebook_output(llm_hidden) — confirmed by
            #    vLLM-Omni and mudler/voxtral-tts.c references
            #    Output has 8320 logits: [EMPTY(0), END(1), code_0(2), ..., code_8191(8193), extras...]
            semantic_logits = self.llm.get_semantic_logits(
                last_hidden, semantic_head=self.acoustic.semantic_codebook_output
            )  # (1, 1, 8320)

            # Mask invalid positions:
            #   Index 0 = EMPTY_AUDIO → mask to -inf (never select, per C ref)
            #   Indices >= 8194 = padding → mask to -inf
            n_special = 2  # EMPTY_AUDIO=0, END_AUDIO=1
            n_valid = self.config.audio_model.semantic_codebook_size + n_special  # 8194
            semantic_logits[:, :, 0] = float("-inf")       # mask EMPTY_AUDIO
            semantic_logits[:, :, n_valid:] = float("-inf") # mask padding

            # b. Sample semantic code (temperature + top-k + top-p)
            logits_for_sampling = semantic_logits[0, 0].clone()  # (8320,)

            # Repetition penalty: reduce logits for codes already generated
            if repetition_penalty > 1.0 and all_semantic_codes:
                for prev_code in set(all_semantic_codes):
                    idx = prev_code + n_special  # convert back to logit index
                    if 0 <= idx < logits_for_sampling.shape[0]:
                        if logits_for_sampling[idx] > 0:
                            logits_for_sampling[idx] /= repetition_penalty
                        else:
                            logits_for_sampling[idx] *= repetition_penalty

            if temperature <= 0:
                # Greedy
                raw_val = logits_for_sampling.argmax().item()
            else:
                # Temperature scaling
                logits_for_sampling = logits_for_sampling / temperature

                # Top-k filtering
                if top_k > 0:
                    topk_vals, topk_idx = logits_for_sampling.topk(min(top_k, logits_for_sampling.size(0)))
                    mask = torch.full_like(logits_for_sampling, float("-inf"))
                    mask.scatter_(0, topk_idx, topk_vals)
                    logits_for_sampling = mask

                # Top-p (nucleus) filtering
                if 0 < top_p < 1.0:
                    sorted_logits, sorted_indices = logits_for_sampling.sort(descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = probs.cumsum(dim=-1)
                    # Remove tokens with cumulative probability above threshold
                    sorted_mask = cumulative_probs - probs > top_p
                    sorted_logits[sorted_mask] = float("-inf")
                    logits_for_sampling.scatter_(0, sorted_indices, sorted_logits)

                # Sample from distribution
                probs = F.softmax(logits_for_sampling, dim=-1)
                raw_val = torch.multinomial(probs, 1).item()

            raw_code = torch.tensor([[raw_val]], device=self.device, dtype=torch.long)

            # Check for end-of-audio: index 1 = END_AUDIO special token
            if raw_val == 1:
                logger.info(f"End of audio (END_AUDIO) at frame {frame_idx}")
                break

            # Convert from shifted index to raw semantic code: subtract 2 special tokens
            semantic_code = raw_code - n_special  # (1, 1) — now in [0, 8191]
            sc_val = semantic_code.item()

            # c. Generate acoustic codes via flow matching
            acoustic_codes = self.acoustic.generate(
                last_hidden, n_steps=n_flow_steps, cfg_alpha=cfg_alpha
            )  # (1, 36)

            all_semantic_codes.append(sc_val)
            all_acoustic_codes.append(acoustic_codes.squeeze(0).cpu())

            # d. Embed the generated codes and feed back to LLM
            last_hidden = self.llm.forward_audio_codes(
                semantic_code, acoustic_codes, start_pos=pos
            )
            pos += 1

            if (frame_idx + 1) % 100 == 0:
                duration = (frame_idx + 1) * 0.08  # 80ms per frame
                logger.info(f"Generated {frame_idx + 1} frames ({duration:.1f}s of audio)")

        n_frames = len(all_semantic_codes)
        if n_frames == 0:
            logger.warning("No audio frames generated!")
            return np.zeros(24000, dtype=np.float32), self.config.sample_rate

        logger.info(f"Generated {n_frames} frames ({n_frames * 0.08:.1f}s of audio), "
                     f"{len(set(all_semantic_codes))} unique semantic codes")

        # 7. Decode all frames to waveform
        semantic_tensor = torch.tensor(all_semantic_codes, device=self.device).unsqueeze(0)  # (1, T)
        acoustic_tensor = torch.stack(all_acoustic_codes).unsqueeze(0).to(self.device)  # (1, T, 36)

        waveform = self.codec(semantic_tensor, acoustic_tensor)  # (1, samples)
        waveform_np = waveform.squeeze(0).float().cpu().numpy()

        duration = len(waveform_np) / self.config.sample_rate
        logger.info(f"Decoded {duration:.2f}s waveform at {self.config.sample_rate}Hz")

        # Normalize to [-1, 1]
        max_val = np.abs(waveform_np).max()
        if max_val > 0:
            waveform_np = waveform_np / max_val * 0.95

        return waveform_np, self.config.sample_rate

    def free_memory(self):
        """Free all model memory."""
        import gc

        self.llm = None
        self.acoustic = None
        self.codec = None
        self.encoder = None
        self.tokenizer = None
        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("Voxtral model memory freed")
