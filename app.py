import streamlit as st
import whisper
import requests
import tempfile
from gtts import gTTS
import os
import json

# Set page config
st.set_page_config(layout="centered")
st.markdown(
    """
    <style>
        .block-container { padding-top: 0rem !important; }
        header[data-testid="stHeader"] { visibility: hidden; }
        .main .block-container { padding-left: 2rem; padding-right: 2rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# Whisper 모델 캐시 로딩
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

# Groq API 키
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# script_library.json 불러오기
@st.cache_resource
def load_script_library():
    with open("script_library.json", "r", encoding="utf-8") as f:
        return json.load(f)

# Groq API 요청 함수들
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

def transform_quiz(sentence):
    prompt = f"""
    Given the sentence: "{sentence}"
    1. Provide a more natural version in casual English.
    2. Provide a more formal/polite version.
    Include explanations in Korean.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are an English grammar coach."},
            {"role": "user", "content": prompt}
        ]
    }
    res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
    return res.json()["choices"][0]["message"]["content"]

# 페이지 상태 관리
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page):
    st.session_state.page = page

# 홈 화면
if st.session_state.page == "home":
    st.title("🎧 Opic & 회화 피드백 머신")
    st.subheader("연습하고 싶은 기능을 선택하세요:")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎤 음성 피드백"):
            go_to("음성 피드백")
    with col2:
        if st.button("📘 모범 답변 듣기"):
            go_to("모범 답변 듣기")
    with col3:
        if st.button("🧠 문장 변환 퀴즈"):
            go_to("문장 변환 퀴즈")

# 🎤 음성 피드백 페이지
elif st.session_state.page == "음성 피드백":
    st.button("← 홈으로", on_click=lambda: go_to("home"))
    st.subheader("🎤 음성 피드백")

    uploaded_file = st.file_uploader("🔊 음성/영상 파일 업로드 (.wav, .mp3, .mp4, .m4a)", type=["wav", "mp3", "mp4", "m4a"])
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            st.audio(tmp.name)

        os.remove(media_path)

# 📘 모범 답변 듣기
elif st.session_state.page == "모범 답변 듣기":
    st.button("← 홈으로", on_click=lambda: go_to("home"))
    st.subheader("📘 모범 답변 듣기")

    script_library = load_script_library()
    topic = st.selectbox("📚 주제를 선택하세요", list(script_library.keys()))
    if topic:
        questions = list(script_library[topic].keys())
        question = st.selectbox("❓ 질문을 선택하세요", questions)

        if question:
            entry = script_library[topic][question]
            question_en = entry["question_en"]
            script_text = entry["script"]

            st.markdown(f"**🗨️ 질문 (한글):** {question}")
            st.markdown(f"**🗨️ 질문 (영어):** {question_en}")
            st.markdown(f"**📘 모범 답변:**\n\n{script_text}")

            if st.button("🎧 모범 답변 듣기"):
                tts = gTTS(script_text, lang="en")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tts.save(tmp.name)
                    st.audio(tmp.name)

# 🧠 문장 변환 퀴즈
elif st.session_state.page == "문장 변환 퀴즈":
    st.button("← 홈으로", on_click=lambda: go_to("home"))
    st.subheader("🧠 문장 변환 퀴즈")

    sentence = st.text_input("✏️ 변환하고 싶은 문장 입력")
    if st.button("자연스럽게 바꾸기") and sentence:
        result = transform_quiz(sentence)
        st.markdown(result)
