"""Free Voxtral model memory node."""

import logging
import gc
import torch

logger = logging.getLogger("Voxtral")


class VoxtralFreeMemoryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio passthrough — triggers memory cleanup then forwards audio unchanged",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "free_memory"
    CATEGORY = "Voxtral"
    DESCRIPTION = "Free Voxtral model from VRAM/RAM. Audio passes through unchanged."

    def free_memory(self, audio):
        try:
            from .voxtral_tts_node import _model_cache

            if _model_cache["model"] is not None:
                _model_cache["model"].free_memory()
                _model_cache["model"] = None
                _model_cache["model_path"] = None
                logger.info("Voxtral model freed from memory")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception as e:
            logger.error(f"Error freeing Voxtral memory: {e}")

        return (audio,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
