"""Voxtral TTS voice cloning node for ComfyUI.

Takes audio input for voice cloning instead of a voice preset.
Encoder is built from decoder weights (adjoint transposition) — no extra files needed.
"""

import logging
import time
import torch
import torchaudio
import numpy as np
from typing import Tuple

logger = logging.getLogger("Voxtral")

from .voxtral_tts_node import (
    HF_REPO_ID,
    _model_cache,
    get_optimal_device,
    _get_voxtral_dir,
    _is_valid_model_dir,
    get_available_models,
    download_model,
)


class VoxtralVoiceCloneNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        available_models = get_available_models()
        model_choices = [name for _, name in available_models]
        if not model_choices:
            model_choices = [HF_REPO_ID.split("/")[-1] + " (auto-download)"]
        default_model = model_choices[0]

        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, this is a test of the Voxtral text to speech system.",
                    "tooltip": "Text to convert to speech",
                    "dynamicPrompts": True,
                }),
                "voice_to_clone": ("AUDIO", {
                    "tooltip": "Reference audio to clone the voice from. Best results with 3-30 seconds of clean speech.",
                }),
                "model": (model_choices, {
                    "default": default_model,
                    "tooltip": "Select a model from ComfyUI/models/voxtral/ folder.",
                }),
                "flow_steps": ("INT", {
                    "default": 8, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Number of Euler ODE steps for flow matching.",
                }),
                "cfg_alpha": ("FLOAT", {
                    "default": 1.2, "min": 0.5, "max": 3.0, "step": 0.1,
                    "tooltip": "Classifier-free guidance strength.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Sampling temperature. 0 = greedy (official default).",
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Nucleus sampling threshold.",
                }),
                "top_k": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 1,
                    "tooltip": "Top-k sampling. 0 = disabled.",
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Repetition penalty for semantic codes.",
                }),
                "max_duration_sec": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0,
                    "tooltip": "Maximum audio duration in seconds.",
                }),
                "free_memory_after": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Free model from VRAM after generation.",
                }),
            },
            "optional": {
                "text_input": ("STRING", {
                    "tooltip": "Optional: text from another node (overrides text field)",
                    "forceInput": True,
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "Voxtral"
    DESCRIPTION = "Generate speech from text using voice cloned from reference audio."

    def _audio_to_waveform(self, audio_input) -> Tuple[torch.Tensor, int]:
        """Convert ComfyUI AUDIO format to mono waveform tensor."""
        waveform = audio_input["waveform"]  # (B, C, T)
        sample_rate = audio_input["sample_rate"]

        if waveform.dim() == 3:
            waveform = waveform[0]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        return waveform.float(), sample_rate

    def _resample_to_24k(self, waveform: torch.Tensor, src_rate: int) -> torch.Tensor:
        """Resample waveform to 24kHz if needed."""
        if src_rate == 24000:
            return waveform
        resampler = torchaudio.transforms.Resample(src_rate, 24000)
        return resampler(waveform.unsqueeze(0)).squeeze(0)

    def generate_speech(self, text: str, voice_to_clone, model: str,
                        flow_steps: int = 8, cfg_alpha: float = 1.2,
                        temperature: float = 0.0, top_p: float = 1.0,
                        top_k: int = 0, repetition_penalty: float = 1.1,
                        max_duration_sec: float = 30.0,
                        free_memory_after: bool = True,
                        text_input: str = None):
        try:
            # Resolve text
            final_text = text_input if text_input and text_input.strip() else text
            if not final_text or not final_text.strip():
                raise ValueError("No text provided")
            final_text = final_text.strip()

            # Resolve model path
            available_models = get_available_models(auto_download=False)
            model_path = None
            for path, name in available_models:
                if name == model:
                    model_path = path
                    break

            if not model_path:
                logger.info(f"Model '{model}' not found locally, triggering auto-download...")
                voxtral_dir = _get_voxtral_dir()
                if not voxtral_dir:
                    raise ValueError("Cannot determine models directory")
                model_path = download_model(voxtral_dir)
                if not _is_valid_model_dir(model_path):
                    raise ValueError(f"Download completed but model files are incomplete")

            # Load or reuse cached TTS model
            from ..model.inference import VoxtralTTS

            if (_model_cache["model"] is not None and
                    _model_cache["model_path"] == model_path):
                tts = _model_cache["model"]
                logger.info("Reusing cached Voxtral model")
            else:
                if _model_cache["model"] is not None:
                    _model_cache["model"].free_memory()

                device = get_optimal_device()
                tts = VoxtralTTS(model_path, device=device)
                tts.load()

                if not free_memory_after:
                    _model_cache["model"] = tts
                    _model_cache["model_path"] = model_path

            # Convert input audio to 24kHz mono
            waveform, src_rate = self._audio_to_waveform(voice_to_clone)
            waveform_24k = self._resample_to_24k(waveform, src_rate)
            duration = len(waveform_24k) / 24000
            logger.info(f"Reference audio: {duration:.1f}s at 24kHz")

            # Encode audio → codes → voice embedding
            semantic_codes, acoustic_codes = tts.encode_audio(waveform_24k)
            logger.info(f"Encoded: {semantic_codes.shape[1]} frames, "
                         f"{len(torch.unique(semantic_codes))} unique semantic codes")

            voice_emb = tts.compute_voice_embedding(semantic_codes, acoustic_codes)
            logger.info(f"Voice embedding: {voice_emb.shape[0]} frames, "
                         f"norm={voice_emb.norm(dim=-1).mean():.2f}")

            # Generate speech with cloned voice
            max_frames = int(max_duration_sec * 12.5)
            logger.info(f"Generating speech: cloned voice, text='{final_text[:80]}...'")

            waveform_np, sample_rate = tts.generate(
                text=final_text,
                voice_embedding=voice_emb,
                max_frames=max_frames,
                n_flow_steps=flow_steps,
                cfg_alpha=cfg_alpha,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

            # Convert to ComfyUI AUDIO format
            waveform_tensor = torch.from_numpy(waveform_np).float().unsqueeze(0).unsqueeze(0)
            audio_dict = {
                "waveform": waveform_tensor,
                "sample_rate": sample_rate,
            }

            logger.info(f"Generated {len(waveform_np) / sample_rate:.2f}s of audio at {sample_rate}Hz")

            # Free memory if requested
            if free_memory_after:
                tts.free_memory()
                _model_cache["model"] = None
                _model_cache["model_path"] = None

            return (audio_dict,)

        except Exception as e:
            import comfy.model_management as mm
            if isinstance(e, mm.InterruptProcessingException):
                logger.info("Generation interrupted by user")
                raise
            logger.error(f"Voxtral voice clone failed: {e}")
            raise RuntimeError(f"Voxtral voice clone error: {e}")

    @classmethod
    def IS_CHANGED(cls, text="", model="", voice_to_clone=None, text_input=None, **kwargs):
        return time.time()
