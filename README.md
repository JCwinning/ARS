[English](README.md) | [中文](_README_CN.md)

# 🎙️ Voice Studio: ASR & TTS Suite

A powerful, multi-model Streamlit application for **Automatic Speech Recognition (ASR)** and **Text-to-Speech (TTS)**. This project allows users to compare state-of-the-art cloud models from NVIDIA and Google with local MLX-optimized models.

## Live demo

https://jcwinning-speech-text-model.share.connect.posit.cloud/


## ✨ Features

### 🎤 Speech to Text (STT)
- **Google Gemini 2.5 Flash Lite**: High-speed, accurate cloud transcription via OpenRouter.
- **NVIDIA Parakeet-CTC**: Industry-leading ASR performance via NVIDIA Riva Cloud.
- **Local MLX Models**: Fast, private, on-device transcription optimized for Apple Silicon.
  - **GLM-ASR-Nano**: Ultra-lightweight and efficient.
  - **Whisper-Large-v3-Turbo**: High-accuracy state-of-the-art transcription.
- **Dual Input**: Support for live microphone recording or file uploads (WAV, MP3, M4A).
- **Incremental Display**: Results are displayed in real-time as each model finishes; no waiting for the slowest model.
- **Auto-Normalization**: Automatically converts and normalizes audio to 16kHz Mono WAV to ensure maximum ASR accuracy.
- **Download Results**: Save transcription results to localized `.md` files for each model.

### 🔊 Text to Speech (TTS)
- **Qwen TTS (DashScope)**: Natural-sounding synthesis with 7 distinct voices (Dylan, Cherry, Serena, etc.).
- **NVIDIA Riva (Magpie)**: Professional-grade multilingual synthesis featuring the latest Magpie-Multilingual models.
- **Dynamic Voice Selection**: Interactive selection of voices for both Chinese (Mandarin) and English.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Apple Silicon (for Local MLX features)
- API Keys for:
  - [OpenRouter](https://openrouter.ai/)
  - [NVIDIA NIM](https://build.nvidia.com/)
  - [Aliyun DashScope](https://dashscope.console.aliyun.com/)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ARS
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file for your API keys:
   ```env
   OPENROUTER_API_KEY=your_key_here
   DASHSCOPE_API_KEY=your_key_here
   NVIDIA_API_KEY=your_key_here
   ```

### Running the App
```bash
streamlit run app.py
```

## ☁️ Deployment
This project is pre-configured for **Streamlit Cloud**. 
- It automatically detects the cloud environment and disables local-only models (MLX) to prevent crashes.
- API keys can be managed via Streamlit's "Secrets" management.

## 🛠️ Technology Stack
- **Frontend**: Streamlit
- **ML Engine**: MLX (Apple Silicon optimized)
- **Cloud APIs**: NVIDIA Riva, OpenRouter (Gemini), Aliyun DashScope (Qwen)
- **Audio Processing**: Wave, SoundFile, Streamlit Mic Recorder

---
*Created for advanced AI voice workflows.*
