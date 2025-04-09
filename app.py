import streamlit as st
import whisper
import requests
import tempfile
from gtts import gTTS
import os

def load_api_key(filepath="groq_api_key.txt"):
    with open(filepath, "r") as f:
        return f.read().strip()

GROQ_API_KEY = load_api_key()

# Whisper 모델 캐시 로딩
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

# Groq API 요청 함수
def get_groq_feedback(user_input):
    prompt = f"""
    You are an English conversation tutor.
    Here is what the student said:

    "{user_input}"

    Please give:
    1. Corrected version (if there are any grammar issues)
    2. A more natural way to say it in casual English
    3. Short feedback or tip

    Output in Korean and English.
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful English tutor."},
            {"role": "user", "content": prompt}
        ]
    }

    res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
    return res.json()["choices"][0]["message"]["content"]

# Streamlit UI
st.title("🎧 영어 회화 피드백 챗봇 (Groq + Whisper)")
uploaded_file = st.file_uploader("🔊 음성/영상 파일 업로드 (.wav, .mp3, .mp4)", type=["wav", "mp3", "mp4"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        media_path = tmp.name

    with st.spinner("🧠 음성 인식 중..."):
        text = load_whisper().transcribe(media_path)["text"]
    st.success("🗣️ 인식된 문장:")
    st.write(text)

    with st.spinner("🤖 Groq로 피드백 생성 중..."):
        feedback = get_groq_feedback(text)
    st.success("📘 회화 피드백:")
    st.markdown(feedback)

    tts = gTTS(feedback, lang="en")
    tts_path = media_path.replace(".mp4", "_feedback.mp3")
    tts.save(tts_path)
    st.audio(tts_path)

    # 정리
    os.remove(media_path)
    os.remove(tts_path)
