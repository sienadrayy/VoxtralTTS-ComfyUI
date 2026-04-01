# Voxtral TTS ComfyUI Nodes

A ComfyUI integration for Mistral's [Voxtral-4B-TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) text-to-speech model, enabling high-quality speech synthesis and voice cloning directly within your ComfyUI workflows. Fully local inference — no API or vLLM server required.

## Features

- **Text-to-Speech** with 21 voice presets across 9 languages (English, French, Spanish, German, Italian, Portuguese, Dutch, Arabic, Hindi)
- **Voice Cloning** from short reference audio (3-30 seconds of clean speech)
- **Fully Local** — no external API calls, runs entirely on your hardware
- **Auto Model Download** — downloads model from HuggingFace on first use
- **Memory Management** — dedicated node to free VRAM between generations
- **Cross-Platform** — CUDA, MPS (Apple Silicon), and CPU support

## Installation

1. Clone this repository into your ComfyUI custom nodes folder:
```
cd ComfyUI/custom_nodes
git clone https://github.com/sienadrayy/VoxtralTTS-ComfyUI.git
```

2. Install requirements:
```
cd VoxtralTTS-ComfyUI
pip install -r requirements.txt
```

3. Restart ComfyUI — nodes will appear under the **Voxtral** category

The model is automatically downloaded from HuggingFace on first use and saved to `ComfyUI/models/voxtral/`.

## Available Nodes

| Node | Description |
|------|-------------|
| **Voxtral TTS** | Generate speech from text using 21 voice presets |
| **Voxtral Voice Clone** | Generate speech using a voice cloned from reference audio |
| **Voxtral Load Text From File** | Load text from a `.txt` file for TTS input |
| **Voxtral Free Memory** | Free Voxtral model from VRAM; audio passes through unchanged |

## Model Info

- **Model**: [Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) by Mistral AI (4B parameters)
- **Output**: 24 kHz mono audio
- **License**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## License

This wrapper is licensed under **Apache-2.0**. The Voxtral model itself is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
