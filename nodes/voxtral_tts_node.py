"""Main Voxtral TTS generation node for ComfyUI."""

import logging
import os
import threading
import torch
import numpy as np
from typing import List, Tuple, Optional

logger = logging.getLogger("Voxtral")

# HuggingFace repo ID for Voxtral TTS
HF_REPO_ID = "mistralai/Voxtral-4B-TTS-2603"

# Global model cache to avoid reloading
_model_cache = {
    "model": None,
    "model_path": None,
}

# Download state (for progress reporting)
_download_state = {
    "in_progress": False,
    "message": "",
}


def get_optimal_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_voxtral_dir() -> str:
    """Get the voxtral models directory path."""
    try:
        import folder_paths
        models_dir = folder_paths.get_folder_paths("checkpoints")[0]
        return os.path.join(os.path.dirname(models_dir), "voxtral")
    except Exception:
        return ""


def _is_valid_model_dir(folder_path: str) -> bool:
    """Check if a directory contains all required Voxtral model files."""
    return (
        os.path.exists(os.path.join(folder_path, "params.json"))
        and os.path.exists(os.path.join(folder_path, "consolidated.safetensors"))
        and os.path.exists(os.path.join(folder_path, "tekken.json"))
    )


def download_model(dest_dir: str, repo_id: str = HF_REPO_ID) -> str:
    """Download the Voxtral model from HuggingFace Hub.

    Uses huggingface_hub snapshot_download to fetch all model files
    (params.json, consolidated.safetensors, tekken.json, voice_embedding/*.pt).

    Args:
        dest_dir: parent directory (models/voxtral/) where model folder will be created
        repo_id: HuggingFace repository ID

    Returns:
        Path to the downloaded model directory
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for auto-download. "
            "Install it with: pip install huggingface_hub"
        )

    # Use the repo name as the folder name
    model_name = repo_id.split("/")[-1]
    model_dir = os.path.join(dest_dir, model_name)
    os.makedirs(dest_dir, exist_ok=True)

    # If already fully downloaded, skip
    if _is_valid_model_dir(model_dir):
        logger.info(f"Model already downloaded at {model_dir}")
        return model_dir

    logger.info(f"Downloading {repo_id} from HuggingFace Hub (~8GB)...")
    logger.info(f"Destination: {model_dir}")
    _download_state["in_progress"] = True
    _download_state["message"] = f"Downloading {repo_id}..."

    try:
        # snapshot_download fetches the entire repo to a local directory
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            # Only download the files we need (skip .gitattributes etc)
            allow_patterns=[
                "params.json",
                "tekken.json",
                "consolidated.safetensors",
                "voice_embedding/*.pt",
                "README.md",
            ],
        )
        logger.info(f"Download complete: {downloaded_path}")
        _download_state["message"] = "Download complete"
        return downloaded_path
    except Exception as e:
        _download_state["message"] = f"Download failed: {e}"
        logger.error(f"Failed to download model: {e}")
        raise
    finally:
        _download_state["in_progress"] = False


def get_available_models(auto_download: bool = False) -> List[Tuple[str, str]]:
    """Scan models/voxtral/ directory for available models.

    If no models found and auto_download is True, downloads the default model
    from HuggingFace Hub.

    Returns list of (folder_path, display_name) tuples.
    """
    voxtral_dir = _get_voxtral_dir()
    if not voxtral_dir:
        return []

    try:
        if not os.path.exists(voxtral_dir):
            os.makedirs(voxtral_dir, exist_ok=True)
            logger.info(f"Created voxtral models directory: {voxtral_dir}")

        models = []
        for folder in os.listdir(voxtral_dir):
            folder_path = os.path.join(voxtral_dir, folder)
            if not os.path.isdir(folder_path) or folder.startswith("."):
                continue

            if _is_valid_model_dir(folder_path):
                models.append((folder_path, folder))
            else:
                missing = []
                for f in ["params.json", "consolidated.safetensors", "tekken.json"]:
                    if not os.path.exists(os.path.join(folder_path, f)):
                        missing.append(f)
                logger.debug(f"Skipping {folder}: missing {', '.join(missing)}")

        # Auto-download if no models found and requested
        if not models and auto_download:
            logger.info("No Voxtral models found locally. Starting auto-download...")
            downloaded_path = download_model(voxtral_dir)
            folder_name = os.path.basename(downloaded_path)
            if _is_valid_model_dir(downloaded_path):
                models.append((downloaded_path, folder_name))

        models.sort(key=lambda x: x[1])
        if models:
            logger.info(f"Found {len(models)} Voxtral model(s)")
        else:
            logger.warning(f"No Voxtral models found. Place models in: {voxtral_dir}")

        return models

    except Exception as e:
        logger.error(f"Error scanning models: {e}")
        return []


def get_available_voices(model_path: str) -> List[str]:
    """Get available voice presets for a model."""
    voice_dir = os.path.join(model_path, "voice_embedding")
    if not os.path.exists(voice_dir):
        return ["casual_male", "casual_female", "neutral_male", "neutral_female", "cheerful_female"]

    voices = []
    for f in sorted(os.listdir(voice_dir)):
        if f.endswith(".pt"):
            voices.append(f[:-3])
    return voices if voices else ["casual_male"]


# Default voices (shown before model is selected)
DEFAULT_VOICES = [
    "casual_female", "casual_male", "cheerful_female",
    "neutral_female", "neutral_male",
    "ar_male", "de_female", "de_male",
    "es_female", "es_male", "fr_female", "fr_male",
    "hi_female", "hi_male", "it_female", "it_male",
    "nl_female", "nl_male", "pt_female", "pt_male",
]


class VoxtralTTSNode:
    def __init__(self):
        self._tts = None

    @classmethod
    def INPUT_TYPES(cls):
        available_models = get_available_models()
        model_choices = [name for _, name in available_models]
        # If no models found, show the HF repo name as a hint that it will auto-download
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
                "model": (model_choices, {
                    "default": default_model,
                    "tooltip": "Select a model from ComfyUI/models/voxtral/ folder. If none found, the model will be auto-downloaded from HuggingFace on first run.",
                }),
                "voice": (DEFAULT_VOICES, {
                    "default": "casual_male",
                    "tooltip": "Voice preset. 21 built-in voices across 9 languages.",
                }),
                "flow_steps": ("INT", {
                    "default": 8, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Number of Euler ODE steps for flow matching. 8 is optimal. More steps = slightly better quality but slower.",
                }),
                "cfg_alpha": ("FLOAT", {
                    "default": 1.2, "min": 0.5, "max": 3.0, "step": 0.1,
                    "tooltip": "Classifier-free guidance strength. 1.2 = default. Higher = more professional but less emotional.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Sampling temperature for semantic codes. 0 = greedy (official default). Try 0.7-0.9 for more variety.",
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Nucleus sampling threshold. 1.0 = disabled (official default).",
                }),
                "top_k": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 1,
                    "tooltip": "Top-k sampling. 0 = disabled (official default).",
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Repetition penalty for semantic codes. 1.1 = official default. Penalizes repeated codes.",
                }),
                "max_duration_sec": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0,
                    "tooltip": "Maximum audio duration in seconds. Each frame = 80ms.",
                }),
                "free_memory_after": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Free model from VRAM after generation. Disable to keep loaded for faster subsequent runs.",
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
    DESCRIPTION = "Generate speech from text using Mistral Voxtral-4B TTS with 21 voice presets across 9 languages"

    def generate_speech(self, text: str, model: str, voice: str,
                        flow_steps: int = 8, cfg_alpha: float = 1.2,
                        temperature: float = 0.0, top_p: float = 1.0,
                        top_k: int = 0, repetition_penalty: float = 1.1,
                        max_duration_sec: float = 30.0,
                        free_memory_after: bool = True,
                        text_input: str = None):
        try:
            # Use connected text input if provided
            final_text = text_input if text_input and text_input.strip() else text
            if not final_text or not final_text.strip():
                raise ValueError("No text provided")

            final_text = final_text.strip()

            # Resolve model path — try local first, auto-download if needed
            available_models = get_available_models(auto_download=False)
            model_path = None
            for path, name in available_models:
                if name == model:
                    model_path = path
                    break

            if not model_path:
                # Auto-download from HuggingFace
                logger.info(f"Model '{model}' not found locally, triggering auto-download...")
                voxtral_dir = _get_voxtral_dir()
                if not voxtral_dir:
                    raise ValueError("Cannot determine models directory")
                model_path = download_model(voxtral_dir)
                if not _is_valid_model_dir(model_path):
                    raise ValueError(
                        f"Download completed but model files are incomplete at {model_path}"
                    )

            # Load or reuse cached model
            from ..model.inference import VoxtralTTS

            if (_model_cache["model"] is not None and
                    _model_cache["model_path"] == model_path):
                tts = _model_cache["model"]
                logger.info("Reusing cached Voxtral model")
            else:
                # Free old model if any
                if _model_cache["model"] is not None:
                    _model_cache["model"].free_memory()

                device = get_optimal_device()
                tts = VoxtralTTS(model_path, device=device)
                tts.load()

                if not free_memory_after:
                    _model_cache["model"] = tts
                    _model_cache["model_path"] = model_path

            # Convert max duration to max frames (80ms per frame = 12.5 Hz)
            max_frames = int(max_duration_sec * 12.5)

            # Generate
            logger.info(f"Generating speech: voice={voice}, text='{final_text[:80]}...'")
            waveform_np, sample_rate = tts.generate(
                text=final_text,
                voice=voice,
                max_frames=max_frames,
                n_flow_steps=flow_steps,
                cfg_alpha=cfg_alpha,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

            # Convert to ComfyUI AUDIO format: {"waveform": tensor(B,C,T), "sample_rate": int}
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
            logger.error(f"Voxtral TTS failed: {e}")
            raise RuntimeError(f"Voxtral TTS error: {e}")

    @classmethod
    def IS_CHANGED(cls, text="", model="", voice="", text_input=None, **kwargs):
        # Always re-run (temperature sampling is non-deterministic)
        import time
        return time.time()
