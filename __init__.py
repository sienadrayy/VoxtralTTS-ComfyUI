"""Voxtral TTS for ComfyUI — local inference from Mistral Voxtral-4B-TTS-2603.

Generates speech from text using 21 voice presets across 9 languages.
Model weights loaded directly from consolidated.safetensors (no vLLM server required).
"""

__version__ = "0.1.0"
__author__ = "Voxtral-ComfyUI"
__title__ = "Voxtral TTS ComfyUI"

import logging

logger = logging.getLogger("Voxtral")
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[Voxtral] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Register text loading node (always available)
try:
    from .nodes.load_text_node import VoxtralLoadTextNode
    NODE_CLASS_MAPPINGS["VoxtralLoadText"] = VoxtralLoadTextNode
    NODE_DISPLAY_NAME_MAPPINGS["VoxtralLoadText"] = "Voxtral Load Text From File"
except Exception as e:
    logger.error(f"Failed to register VoxtralLoadText: {e}")

# Register TTS node
try:
    from .nodes.voxtral_tts_node import VoxtralTTSNode
    NODE_CLASS_MAPPINGS["VoxtralTTS"] = VoxtralTTSNode
    NODE_DISPLAY_NAME_MAPPINGS["VoxtralTTS"] = "Voxtral TTS"
    logger.info("Voxtral TTS node registered")
except Exception as e:
    logger.error(f"Failed to register VoxtralTTS: {e}")

# Register free memory node
try:
    from .nodes.free_memory_node import VoxtralFreeMemoryNode
    NODE_CLASS_MAPPINGS["VoxtralFreeMemory"] = VoxtralFreeMemoryNode
    NODE_DISPLAY_NAME_MAPPINGS["VoxtralFreeMemory"] = "Voxtral Free Memory"
except Exception as e:
    logger.error(f"Failed to register VoxtralFreeMemory: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
