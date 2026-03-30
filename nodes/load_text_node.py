"""Load text from file node for Voxtral TTS."""

import os
import logging

logger = logging.getLogger("Voxtral")


class VoxtralLoadTextNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to a .txt file to load"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "load_text"
    CATEGORY = "Voxtral"
    DESCRIPTION = "Load text from a .txt file for Voxtral TTS"

    def load_text(self, file_path: str):
        if not file_path or not file_path.strip():
            raise ValueError("No file path provided")

        file_path = file_path.strip()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            raise ValueError(f"File is empty: {file_path}")

        logger.info(f"Loaded {len(text)} characters from {os.path.basename(file_path)}")
        return (text,)
