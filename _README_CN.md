[English](README.md) | [中文](_README_CN.md)

# 🎙️ Voice Studio: 语音识别与即时语音合成套件

一个强大且美观的 Streamlit 应用，集成了 **自动语音识别 (ASR)** 和 **文本转语音 (TTS)** 功能。本项目旨在提供一个便捷的平台，用于对比 NVIDIA、Google 的前沿云端模型与本地 MLX 优化模型的表现。

## 在线演示

https://jcwinning-speech-text-model.share.connect.posit.cloud/


## ✨ 核心功能

### 🎤 语音转文字 (STT)
- **Google Gemini 2.5 Flash Lite**: 通过 OpenRouter 提供的高速、精准云端转写。
- **NVIDIA Parakeet-CTC**: 行业领先的 ASR 性能，基于 NVIDIA Riva Cloud。
- **MLX-GLM-Nano (本地)**: 专为 Apple Silicon 优化的本地私密转写模型（在云端部署环境中将自动禁用）。
- **双输入模式**: 支持实时麦克风录音或上传音频文件（WAV, MP3, M4A）。
- **并发处理**: 支持多个模型同时进行转写，方便快速评价转写质量。

### 🔊 文本转语音 (TTS)
- **Qwen TTS (DashScope)**: 阿里通义千问提供的自然语音合成，内置 7 种性格各异的声音。
- **NVIDIA Riva (Magpie)**: 专业级多语言合成，采用最新的 Magpie-Multilingual 模型。
- **动态声音选择**: 提供丰富的中文（普通话）和英文发音人选项。

## 🚀 快速上手

### 环境要求
- Python 3.10+
- Apple Silicon (若需使用本地 MLX 功能)
- API 密钥:
  - [OpenRouter](https://openrouter.ai/)
  - [NVIDIA NIM](https://build.nvidia.com/)
  - [阿里云 DashScope](https://dashscope.console.aliyun.com/)

### 安装步骤

1. 克隆仓库:
   ```bash
   git clone <repository-url>
   cd ARS
   ```

2. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```

3. 在项目根目录创建 `.env` 文件并填入密钥:
   ```env
   OPENROUTER_API_KEY=你的密钥
   DASHSCOPE_API_KEY=你的密钥
   NVIDIA_API_KEY=你的密钥
   ```

### 运行应用
```bash
streamlit run app.py
```

## ☁️ 云端部署
本项目已针对 **Streamlit Cloud** 进行预配置：
- 自动检测运行环境，在云端部署时禁用本地模型 (MLX) 以确保系统稳定。
- API 密钥可以通过 Streamlit 的 "Secrets" 面板进行安全管理。

## 🛠️ 技术栈
- **界面**: Streamlit
- **本地推理**: MLX (针对 Mac M 芯片优化)
- **云端服务**: NVIDIA Riva, OpenRouter (Gemini), 阿里云 DashScope (Qwen)
- **音频处理**: Wave, SoundFile, Streamlit Mic Recorder

---


