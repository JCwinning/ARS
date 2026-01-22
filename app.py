import streamlit as st
import os
import tempfile
import time
import wave
import concurrent.futures
import base64
import requests
import dashscope
from pathlib import Path
from dotenv import load_dotenv

# Optional local imports
try:
    import riva.client
    RIVA_AVAILABLE = True
except ImportError:
    RIVA_AVAILABLE = False

try:
    from mlx_audio.stt.utils import load_model
    from mlx_audio.stt.generate import generate_transcription
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from streamlit_mic_recorder import mic_recorder

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Voice Studio - ASR & TTS",
    page_icon="🎙️",
    layout="centered"
)

# Detect if running on Streamlit Cloud
IS_STREAMLIT_CLOUD = os.environ.get("STREAMLIT_RUNTIME_ID") is not None

# Custom Styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e1e2f 0%, #121212 100%);
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        color: white;
    }
    .transcription-box {
        padding: 20px;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 20px;
        min-height: 100px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Voice Studio")
st.markdown("A powerful suite for Speech-to-Text and Text-to-Speech transformations.")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")

# API Keys handling
def get_api_key(key_name, label, help_text):
    # 1. Try environment variable (or .env)
    api_key = os.environ.get(key_name)
    
    # 2. If not found, show user input in sidebar (no warning)
    if not api_key:
        api_key = st.sidebar.text_input(label, type="password", help=help_text)
    
    return api_key

nvidia_api_key = get_api_key("NVIDIA_API_KEY", "NVIDIA API Key", "Get from NVIDIA NIM")
openrouter_api_key = get_api_key("OPENROUTER_API_KEY", "OpenRouter API Key", "Get from openrouter.ai")
dashscope_api_key = get_api_key("DASHSCOPE_API_KEY", "DashScope API Key", "Get from Aliyun DashScope")

# Model selection filtering
available_stt_models = ["Gemini 2.5 Flash Lite (OpenRouter)"]
if RIVA_AVAILABLE:
    available_stt_models.append("NVIDIA Parakeet-CTC (Cloud)")

# Only allow Local MLX if not on cloud and libraries are available
if not IS_STREAMLIT_CLOUD and MLX_AVAILABLE:
    available_stt_models.append("MLX-GLM-Nano (Local)")

selected_models = st.sidebar.multiselect(
    "Select STT Models",
    available_stt_models,
    default=["Gemini 2.5 Flash Lite (OpenRouter)"]
)

available_tts_models = ["Qwen TTS (DashScope)"]
if RIVA_AVAILABLE:
    available_tts_models.append("NVIDIA Riva TTS (Cloud)")

selected_tts_model = st.sidebar.selectbox(
    "Select TTS Model",
    available_tts_models,
    index=0
)

# --- STT Helper Functions ---

@st.cache_resource
def get_mlx_model():
    if not MLX_AVAILABLE or IS_STREAMLIT_CLOUD:
        return None
    with st.spinner("🚀 Loading MLX Model (Local)..."):
        try:
            model = load_model("mlx-community/GLM-ASR-Nano-2512-4bit")
            return model
        except Exception as e:
            st.error(f"❌ Error loading MLX model: {e}")
            return None

def encode_audio_to_base64(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.standard_b64encode(audio_file.read()).decode("utf-8")

def get_audio_format(audio_path):
    extension = Path(audio_path).suffix.lower()
    format_map = {
        ".mp3": "mp3", ".wav": "wav", ".m4a": "m4a", ".ogg": "ogg",
        ".flac": "flac", ".aac": "aac", ".webm": "webm", ".aiff": "aiff",
    }
    return format_map.get(extension, "wav")

def transcribe_with_mlx(model, audio_path):
    try:
        start_time = time.time()
        transcription = generate_transcription(
            model=model,
            audio_path=audio_path,
            verbose=False,
        )
        end_time = time.time()
        return transcription.text, end_time - start_time
    except Exception as e:
        return f"Error: {e}", 0

def transcribe_with_riva(audio_path, api_key):
    if not api_key: return "❌ Missing NVIDIA API Key", 0
    try:
        start_time = time.time()
        auth = riva.client.Auth(
            uri="grpc.nvcf.nvidia.com:443", use_ssl=True,
            metadata_args=[["function-id", "9add5ef7-322e-47e0-ad7a-5653fb8d259b"],
                          ["authorization", f"Bearer {api_key}"]]
        )
        service = riva.client.ASRService(auth)
        with wave.open(audio_path, 'rb') as wf:
            rate = wf.getframerate()
            data = wf.readframes(wf.getnframes())
        config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=rate, language_code="zh-CN",
            max_alternatives=1, enable_automatic_punctuation=True, verbatim_transcripts=True
        )
        response = service.offline_recognize(data, config)
        end_time = time.time()
        text = " ".join([res.alternatives[0].transcript for res in response.results if res.alternatives])
        return text.strip(), end_time - start_time
    except Exception as e:
        return f"Riva Error: {str(e)}", 0

def transcribe_with_openrouter(audio_path, api_key):
    if not api_key: return "❌ Missing OpenRouter API Key", 0
    try:
        start_time = time.time()
        audio_base64 = encode_audio_to_base64(audio_path)
        audio_format = get_audio_format(audio_path)
        
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "google/gemini-2.5-flash-lite",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please transcribe this audio accurately. Output only the transcription, nothing else."},
                    {"type": "input_audio", "input_audio": {"data": audio_base64, "format": audio_format}}
                ]
            }]
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=120)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            text = result["choices"][0]["message"]["content"] if "choices" in result else f"Error: {result}"
            return text, end_time - start_time
        return f"API Error ({response.status_code}): {response.text}", 0
    except Exception as e:
        return f"OpenRouter Error: {str(e)}", 0

# --- TTS Helper Functions ---

def text_to_speech_qwen(text, voice, api_key):
    if not api_key:
        st.error("❌ DashScope API Key not found.")
        return None
    try:
        response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
            model="qwen-tts-latest", api_key=api_key, text=text, voice=voice
        )
        if response and "output" in response and "audio" in response["output"]:
            audio_url = response["output"]["audio"]["url"]
            audio_response = requests.get(audio_url, timeout=60)
            if audio_response.status_code == 200:
                return audio_response.content
        st.error("❌ Failed to generate or download audio.")
        return None
    except Exception as e:
        st.error(f"❌ TTS Error: {str(e)}")
        return None

def text_to_speech_riva(text, voice, api_key):
    if not api_key:
        st.error("❌ NVIDIA API Key not found.")
        return None
    try:
        auth = riva.client.Auth(
            uri="grpc.nvcf.nvidia.com:443", use_ssl=True,
            metadata_args=[["function-id", "877104f7-e885-42b9-8de8-f6e4c6303969"], # Updated Magpie-TTS function id
                          ["authorization", f"Bearer {api_key}"]]
        )
        service = riva.client.SpeechSynthesisService(auth)
        
        # Determine language based on voice name for Magpie
        lang = "zh-CN" if "ZH-CN" in voice else "en-US"
        
        response = service.synthesize(
            text=text,
            voice_name=voice,
            language_code=lang,
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hz=22050
        )
        
        if not response or not response.audio:
            st.error("❌ Riva returned an empty audio response.")
            return None

        # Riva returns raw audio, wrap it in WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            with wave.open(tmp_file.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                wf.writeframes(response.audio)
            
            with open(tmp_file.name, 'rb') as f:
                content = f.read()
            os.remove(tmp_file.name)
            return content
            
    except Exception as e:
        st.error(f"❌ Riva TTS Error: {str(e)}")
        return None

# --- Main Tabs ---
tab1, tab2 = st.tabs(["🎤 Speech to Text", "🔊 Text to Speech"])

with tab1:
    st.header("1. Provide Audio")
    input_method = st.radio("Select Input Method:", ["🎙️ Record Live", "📁 Upload File"], horizontal=True)

    audio_bytes = None
    if input_method == "🎙️ Record Live":
        audio = mic_recorder(start_prompt="⏺ Start Recording", stop_prompt="⏹ Stop & Transcribe", key='recorder', format='wav')
        if audio: audio_bytes = audio['bytes']
    else:
        uploaded_file = st.file_uploader("Upload an audio file (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"])
        if uploaded_file: audio_bytes = uploaded_file.read()

    if audio_bytes:
        st.audio(audio_bytes, format='audio/wav')
        st.header("2. Transcription Results")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            mlx_model = get_mlx_model() if "MLX-GLM-Nano (Local)" in selected_models else None
            
            def process_model(model_name):
                if model_name == "MLX-GLM-Nano (Local)":
                    return transcribe_with_mlx(mlx_model, tmp_path) if mlx_model else ("Local model not loaded.", 0)
                elif model_name == "NVIDIA Parakeet-CTC (Cloud)":
                    return transcribe_with_riva(tmp_path, nvidia_api_key)
                elif model_name == "Gemini 2.5 Flash Lite (OpenRouter)":
                    return transcribe_with_openrouter(tmp_path, openrouter_api_key)
                return "Unknown model", 0

            cols = st.columns(len(selected_models)) if selected_models else [st]
            results = {}
            with st.spinner("Processing..."):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = {executor.submit(process_model, m): m for m in selected_models}
                    for future in concurrent.futures.as_completed(futures):
                        m = futures[future]
                        results[m] = future.result()

            for idx, model_name in enumerate(selected_models):
                text, duration = results.get(model_name, ("Pending...", 0))
                with cols[idx]:
                    st.subheader(model_name)
                    st.markdown(f'<div class="transcription-box">{text}</div>', unsafe_allow_html=True)
                    st.caption(f"⏱️ {duration:.2f}s")
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)
    elif not selected_models:
        st.warning("⚠️ Please select at least one model from the sidebar.")

with tab2:
    st.header("3. Generate Speech")
    tts_text = st.text_area("Enter text to convert to speech:", placeholder="Hello, I can speak now!")
    
    # Dynamic Voice Selection based on Model
    if selected_tts_model == "Qwen TTS (DashScope)":
        voices = ["Dylan", "Cherry", "Serena", "Ethan", "Chelsie", "Jada", "Sunny"]
        default_voice = "Dylan"
    else: # Riva / Magpie-TTS
        voices = [
            "Magpie-Multilingual.ZH-CN.Mia", 
            "Magpie-Multilingual.ZH-CN.Aria", 
            "Magpie-Multilingual.EN-US.Sofia", 
            "Magpie-Multilingual.EN-US.Jason", 
            "Magpie-Multilingual.EN-US.Aria", 
            "Magpie-Multilingual.EN-US.Leo"
        ]
        default_voice = "Magpie-Multilingual.ZH-CN.Mia"

    col_v, col_b = st.columns([2, 1])
    with col_v:
        selected_voice = st.selectbox("Select Voice", voices, index=voices.index(default_voice))
    with col_b:
        st.write("<br>", unsafe_allow_html=True) # spacing
        generate_btn = st.button("🎵 Generate")

    if generate_btn and tts_text:
        with st.spinner(f"Generating speech with {selected_tts_model}..."):
            if selected_tts_model == "Qwen TTS (DashScope)":
                audio_content = text_to_speech_qwen(tts_text, selected_voice, dashscope_api_key)
            else:
                audio_content = text_to_speech_riva(tts_text, selected_voice, nvidia_api_key)
            
            if audio_content:
                st.audio(audio_content, format='audio/wav')
                st.success("✅ Speech generated successfully!")
                st.download_button("📥 Download Audio", audio_content, file_name="speech.wav", mime="audio/wav")

# Footer
st.markdown("---")
st.caption("Powered by Gemini, NVIDIA Riva & Qwen TTS")
