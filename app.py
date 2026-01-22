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
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

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
    layout="wide"
)

# --- Localization ---
LOCALES = {
    "中文": {
        "title": "Voice Studio",
        "subtitle": "强大的语音转文字与文字转语音工具集。",
        "sidebar_header": "⚙️ 配置",
        "nvidia_label": "NVIDIA API 密钥",
        "nvidia_help": "从 NVIDIA NIM 获取",
        "openrouter_label": "OpenRouter API 密钥",
        "openrouter_help": "从 openrouter.ai 获取",
        "dashscope_label": "DashScope API 密钥",
        "dashscope_help": "从阿里云 DashScope 获取",
        "stt_models_label": "选择语音转文字模型",
        "tts_models_label": "选择文字转语音模型",
        "tab1_label": "🎤 语音转文字",
        "tab2_label": "🔊 文字转语音",
        "stt_header": "1. 提供音频",
        "stt_method_label": "选择输入方式：",
        "stt_method_record": "🎙️ 现场录音",
        "stt_method_upload": "📁 上传文件",
        "stt_record_start": "⏺ 开始录音",
        "stt_record_stop": "⏹ 停止并转写",
        "stt_upload_label": "上传音频文件 (WAV, MP3, M4A)",
        "stt_results_header": "2. 转写结果",
        "tts_header": "3. 生成语音",
        "tts_text_label": "输入要转换为语音的文字：",
        "tts_placeholder": "你好，我现在可以说话了！",
        "tts_voice_label": "选择发音人",
        "tts_generate_btn": "🎵 生成",
        "tts_success": "✅ 语音生成成功！",
        "tts_download": "📥 下载音频",
        "footer": "由 Gemini, NVIDIA Riva & Qwen TTS 提供技术支持",
        "processing": "正在处理...",
        "generating": "正在生成语音...",
        "no_model_warning": "⚠️ 请从侧边栏选择至少一个模型。",
        "local_model_warning": "本地模型未加载。",
        "stt_download_md": "📥 下载结果 (.md)",
        "local_model_switch": "🔄 已自动切换本地模型（每次仅限一个）"
    },
    "English": {
        "title": "Voice Studio",
        "subtitle": "A powerful suite for Speech-to-Text and Text-to-Speech transformations.",
        "sidebar_header": "⚙️ Configuration",
        "nvidia_label": "NVIDIA API Key",
        "nvidia_help": "Get from NVIDIA NIM",
        "openrouter_label": "OpenRouter API Key",
        "openrouter_help": "Get from openrouter.ai",
        "dashscope_label": "DashScope API Key",
        "dashscope_help": "Get from Aliyun DashScope",
        "stt_models_label": "Select STT Models",
        "tts_models_label": "Select TTS Model",
        "tab1_label": "🎤 Speech to Text",
        "tab2_label": "🔊 Text to Speech",
        "stt_header": "1. Provide Audio",
        "stt_method_label": "Select Input Method:",
        "stt_method_record": "🎙️ Record Live",
        "stt_method_upload": "📁 Upload File",
        "stt_record_start": "⏺ Start Recording",
        "stt_record_stop": "⏹ Stop & Transcribe",
        "stt_upload_label": "Upload an audio file (WAV, MP3, M4A)",
        "stt_results_header": "2. Transcription Results",
        "tts_header": "3. Generate Speech",
        "tts_text_label": "Enter text to convert to speech:",
        "tts_placeholder": "Hello, I can speak now!",
        "tts_voice_label": "Select Voice",
        "tts_generate_btn": "🎵 Generate",
        "tts_success": "✅ Speech generated successfully!",
        "tts_download": "📥 Download Audio",
        "footer": "Powered by Gemini, NVIDIA Riva & Qwen TTS",
        "processing": "Processing...",
        "generating": "Generating speech...",
        "no_model_warning": "⚠️ Please select at least one model from the sidebar.",
        "local_model_warning": "Local model not loaded.",
        "stt_download_md": "📥 Download Results (.md)",
        "local_model_switch": "🔄 Local model switched automatically (one at a time)"
    }
}

# --- Session State Initialization ---
if "stt_models" not in st.session_state:
    st.session_state.stt_models = []
if "prev_stt_models" not in st.session_state:
    st.session_state.prev_stt_models = []

def enforce_mutual_exclusion():
    """Ensure only one local MLX model can be selected at a time."""
    current = st.session_state.stt_models
    prev = st.session_state.prev_stt_models
    
    local_models = ["MLX-GLM-Nano (Local)", "MLX-Whisper-Turbo (Local)"]
    selected_locals = [m for m in current if m in local_models]
    
    if len(selected_locals) > 1:
        # User just added a second local model. Keep the new one, remove the old one.
        newly_added = [m for m in selected_locals if m not in prev]
        if newly_added:
            keep = newly_added[0]
            st.session_state.stt_models = [m for m in current if m not in local_models or m == keep]
            st.toast(L["local_model_switch"])
        else:
            # Fallback if logic fails: keep Whisper
            st.session_state.stt_models = [m for m in current if m != "MLX-GLM-Nano (Local)"]
    
    st.session_state.prev_stt_models = st.session_state.stt_models

# Language Selector in top right
# Initialize Session State for Language
if "language" not in st.session_state:
    st.session_state.language = "中文"

def toggle_language():
    st.session_state.language = "English" if st.session_state.language == "中文" else "中文"

# Language Toggle in top right
col_title, col_lang = st.columns([7, 2])
with col_lang:
    st.button("中文/EN", on_click=toggle_language, help="Switch Language / 切换语言")

L = LOCALES[st.session_state.language]

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
    /* Language selector positioning tweak */
    div[data-testid="stHorizontalBlock"] > div:last-child {
        display: flex;
        justify-content: flex-end;
    }
    </style>
    """, unsafe_allow_html=True)

with col_title:
    st.title(L["title"])

st.markdown(L["subtitle"])

# --- Sidebar Configuration ---
st.sidebar.header(L["sidebar_header"])

# API Keys handling
def get_api_key(key_name, label, help_text):
    # 1. Try environment variable (or .env)
    api_key = os.environ.get(key_name)
    
    # 2. If not found, show user input in sidebar (no warning)
    if not api_key:
        api_key = st.sidebar.text_input(label, type="password", help=help_text)
    
    return api_key

nvidia_api_key = get_api_key("NVIDIA_API_KEY", L["nvidia_label"], L["nvidia_help"])
openrouter_api_key = get_api_key("OPENROUTER_API_KEY", L["openrouter_label"], L["openrouter_help"])
dashscope_api_key = get_api_key("DASHSCOPE_API_KEY", L["dashscope_label"], L["dashscope_help"])

# Model selection filtering
available_stt_models = []
if openrouter_api_key:
    available_stt_models.append("Gemini 2.5 Flash Lite (OpenRouter)")
if RIVA_AVAILABLE and nvidia_api_key:
    available_stt_models.append("NVIDIA Parakeet-CTC (Cloud)")

# Only allow Local MLX if not on cloud and libraries are available
if not IS_STREAMLIT_CLOUD and MLX_AVAILABLE:
    available_stt_models.append("MLX-GLM-Nano (Local)")
    available_stt_models.append("MLX-Whisper-Turbo (Local)")

# Filter session state to only include currently available models (e.g. if key was removed)
st.session_state.stt_models = [m for m in st.session_state.stt_models if m in available_stt_models]

selected_models = st.sidebar.multiselect(
    L["stt_models_label"],
    available_stt_models,
    key="stt_models",
    on_change=enforce_mutual_exclusion
)

available_tts_models = []
if dashscope_api_key:
    available_tts_models.append("Qwen TTS (DashScope)")
if RIVA_AVAILABLE and nvidia_api_key:
    available_tts_models.append("NVIDIA Riva TTS (Cloud)")

if not available_tts_models:
    st.sidebar.warning("⚠️ No TTS models available. Please provide an API key.")
    selected_tts_model = None
else:
    selected_tts_model = st.sidebar.selectbox(
        L["tts_models_label"],
        available_tts_models,
        index=0
    )

# --- STT Helper Functions ---

@st.cache_resource
def get_mlx_model(model_id):
    if not MLX_AVAILABLE or IS_STREAMLIT_CLOUD:
        return None
    with st.spinner(f"🚀 Loading {model_id}..."):
        try:
            model = load_model(model_id)
            return model
        except Exception as e:
            st.error(f"❌ Error loading {model_id}: {e}")
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
    if not api_key: return f"❌ {L['nvidia_label']} Missing", 0
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
    if not api_key: return f"❌ {L['openrouter_label']} Missing", 0
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
        st.error(f"❌ {L['dashscope_label']} NOT Found")
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
        st.error("❌ Synthesis failed.")
        return None
    except Exception as e:
        st.error(f"❌ TTS Error: {str(e)}")
        return None

def text_to_speech_riva(text, voice, api_key):
    if not api_key:
        st.error(f"❌ {L['nvidia_label']} NOT Found")
        return None
    try:
        auth = riva.client.Auth(
            uri="grpc.nvcf.nvidia.com:443", use_ssl=True,
            metadata_args=[["function-id", "877104f7-e885-42b9-8de8-f6e4c6303969"],
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
            st.error("❌ Riva empty response.")
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
tab1, tab2 = st.tabs([L["tab1_label"], L["tab2_label"]])

with tab1:
    st.header(L["stt_header"])
    input_method = st.radio(L["stt_method_label"], [L["stt_method_record"], L["stt_method_upload"]], horizontal=True)

    audio_bytes = None
    if input_method == L["stt_method_record"]:
        audio = mic_recorder(start_prompt=L["stt_record_start"], stop_prompt=L["stt_record_stop"], key='recorder', format='wav')
        if audio: audio_bytes = audio['bytes']
    else:
        uploaded_file = st.file_uploader(L["stt_upload_label"], type=["wav", "mp3", "m4a"])
        if uploaded_file: audio_bytes = uploaded_file.read()

    if audio_bytes:
        st.audio(audio_bytes, format='audio/wav')
        if "transcription_results" not in st.session_state:
            st.session_state.transcription_results = {}
        
        # Audio Hash for caching
        import hashlib
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        
        # Determine if we need to run models
        needs_processing = "last_audio_hash" not in st.session_state or st.session_state.last_audio_hash != audio_hash
        
        if needs_processing:
            st.session_state.last_audio_hash = audio_hash
            st.session_state.transcription_results = {}

        # Create columns and placeholders for all selected models
        cols = st.columns(len(selected_models)) if selected_models else [st]
        model_placeholders = {}
        
        for idx, model_name in enumerate(selected_models):
            with cols[idx]:
                st.subheader(model_name)
                model_placeholders[model_name] = st.empty()
                
                # If already in cache (e.g. from previous run or just finished), show it
                if model_name in st.session_state.transcription_results:
                    text, duration = st.session_state.transcription_results[model_name]
                    with model_placeholders[model_name].container():
                        st.markdown(f'<div class="transcription-box">{text}</div>', unsafe_allow_html=True)
                        st.caption(f"⏱️ {duration:.2f}s")
                        st.download_button(
                            label=L["stt_download_md"],
                            data=f"# STT Transcription ({model_name})\n\n{text}",
                            file_name=f"{model_name.replace(' ', '_')}_transcription.md",
                            mime="text/markdown",
                            key=f"dl_{idx}_{audio_hash[:8]}" # Key includes hash to prevent stale buttons
                        )
                else:
                    model_placeholders[model_name].info(f"⏳ {L['processing']}")

        # Start processing if needed
        if needs_processing:
            tmp_path = None
            try:
                # 1. Normalize Audio (Force 16kHz, Mono, 16-bit WAV)
                # This ensures maximum compatibility and accuracy for all ASR models (especially Riva)
                import io
                if PYDUB_AVAILABLE:
                    try:
                        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
                        # Normalize: Mono, 16000Hz, 16-bit (sample_width=2)
                        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
                        
                        wav_io = io.BytesIO()
                        audio.export(wav_io, format="wav")
                        final_audio_bytes = wav_io.getvalue()
                    except Exception as e:
                        st.error(f"❌ Conversion/Normalization Error: {e}")
                        final_audio_bytes = audio_bytes # Fallback
                else:
                    st.warning("⚠️ pydub not found. Using raw audio - this may affect Riva accuracy.")
                    final_audio_bytes = audio_bytes

                # 2. Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(final_audio_bytes)
                    tmp_path = tmp_file.name

                # Load models if selected
                glm_model = get_mlx_model("mlx-community/GLM-ASR-Nano-2512-4bit") if "MLX-GLM-Nano (Local)" in selected_models else None
                whisper_model = get_mlx_model("mlx-community/whisper-large-v3-turbo") if "MLX-Whisper-Turbo (Local)" in selected_models else None
                
                def process_model(m_name):
                    try:
                        if m_name == "MLX-GLM-Nano (Local)":
                            return transcribe_with_mlx(glm_model, tmp_path) if glm_model else (L["local_model_warning"], 0)
                        elif m_name == "MLX-Whisper-Turbo (Local)":
                            return transcribe_with_mlx(whisper_model, tmp_path) if whisper_model else (L["local_model_warning"], 0)
                        elif m_name == "NVIDIA Parakeet-CTC (Cloud)":
                            return transcribe_with_riva(tmp_path, nvidia_api_key)
                        elif m_name == "Gemini 2.5 Flash Lite (OpenRouter)":
                            return transcribe_with_openrouter(tmp_path, openrouter_api_key)
                        return "Unknown model", 0
                    except Exception as e:
                        return f"Thread Error: {str(e)}", 0

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = {executor.submit(process_model, m): m for m in selected_models}
                    for future in concurrent.futures.as_completed(futures):
                        m = futures[future]
                        try:
                            text, duration = future.result()
                        except Exception as e:
                            text, duration = f"Fatal Error: {str(e)}", 0
                            
                        st.session_state.transcription_results[m] = (text, duration)
                        
                        # Update the specific placeholder immediately
                        with model_placeholders[m].container():
                            st.markdown(f'<div class="transcription-box">{text}</div>', unsafe_allow_html=True)
                            st.caption(f"⏱️ {duration:.2f}s")
                            st.download_button(
                                label=L["stt_download_md"],
                                data=f"# STT Transcription ({m})\n\n{text}",
                                file_name=f"{m.replace(' ', '_')}_transcription.md",
                                mime="text/markdown",
                                key=f"dl_{selected_models.index(m)}_{audio_hash[:8]}"
                            )
            finally:
                if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)

    elif not selected_models:
        st.warning(L["no_model_warning"])

with tab2:
    st.header(L["tts_header"])
    tts_text = st.text_area(L["tts_text_label"], placeholder=L["tts_placeholder"])
    
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
        selected_voice = st.selectbox(L["tts_voice_label"], voices, index=voices.index(default_voice))
    with col_b:
        st.write("<br>", unsafe_allow_html=True) # spacing
        generate_btn = st.button(L["tts_generate_btn"])

    if generate_btn and tts_text:
        with st.spinner(L["generating"]):
            if selected_tts_model == "Qwen TTS (DashScope)":
                audio_content = text_to_speech_qwen(tts_text, selected_voice, dashscope_api_key)
            else:
                audio_content = text_to_speech_riva(tts_text, selected_voice, nvidia_api_key)
            
            if audio_content:
                st.audio(audio_content, format='audio/wav')
                st.success(L["tts_success"])
                st.download_button(L["tts_download"], audio_content, file_name="speech.wav", mime="audio/wav")

# Footer
st.markdown("---")
st.caption(L["footer"])
