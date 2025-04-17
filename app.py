# 전체 수정된 Streamlit 앱 코드

import streamlit as st
import whisper
import requests
import tempfile
from gtts import gTTS
import os
import json

# 페이지 설정
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

# 페이지 상태 관리
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page):
    st.session_state.page = page
    st.experimental_rerun()

# Groq API 함수들
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
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
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

def generate_opic_questions(topic):
    prompt = f"Generate 3 realistic OPIc questions for the topic: '{topic}'. Include Korean translations."
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "llama3-70b-8192", "messages": [{"role": "system", "content": "You are an expert OPIc question generator."}, {"role": "user", "content": prompt}]}
    res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
    return res.json()["choices"][0]["message"]["content"]

def get_frequent_opic_words():
    prompt = "Please provide 10 commonly used English words/phrases in OPIc with examples and Korean translations."
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "llama3-70b-8192", "messages": [{"role": "system", "content": "You are an OPIc tutor."}, {"role": "user", "content": prompt}]}
    res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
    return res.json()["choices"][0]["message"]["content"]

# ---------------------- 페이지 구성 ----------------------

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

    st.markdown("---")
    col4, col5, col6 = st.columns(3)
    with col4:
        if st.button("📚 스크립트 학습"):
            go_to("스크립트 학습")
    with col5:
        if st.button("❓ 오픽 문제은행"):
            go_to("오픽 문제은행")
    with col6:
        if st.button("🗂 자주 쓰는 단어 학습"):
            go_to("자주 쓰는 단어 학습")

# 음성 피드백
elif st.session_state.page == "음성 피드백":
    st.button("← 홈으로", on_click=lambda: go_to("home"))
    st.subheader("🎤 음성 피드백")
    uploaded_file = st.file_uploader("🔊 음성 파일 업로드 (.wav, .mp3, .mp4, .m4a)", type=["wav", "mp3", "mp4", "m4a"])
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

# 모범 답변 듣기
elif st.session_state.page == "모범 답변 듣기":
    st.button("← 홈으로", on_click=lambda: go_to("home"))
    st.subheader("📘 모범 답변 듣기")
    script_library = load_script_library()
    topic = st.selectbox("📚 주제를 선택하세요", list(script_library.keys()))
    if topic:
        question = st.selectbox("❓ 질문을 선택하세요", list(script_library[topic].keys()))
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

# 문장 변환 퀴즈
elif st.session_state.page == "문장 변환 퀴즈":
    st.button("← 홈으로", on_click=lambda: go_to("home"))
    st.subheader("🧠 문장 변환 퀴즈")
    sentence = st.text_input("✏️ 변환하고 싶은 문장 입력")
    if st.button("자연스럽게 바꾸기") and sentence:
        result = transform_quiz(sentence)
        st.markdown(result)

# 스크립트 학습
elif st.session_state.page == "스크립트 학습":
    st.button("← 홈으로", on_click=lambda: go_to("home"))
    st.subheader("🎙️ 주제별 스크립트를 선택해 들어보세요")
    script_library = load_script_library()
    topic = st.selectbox("📚 주제를 선택하세요", list(script_library.keys()))
    if topic:
        question = st.selectbox("❓ 질문을 선택하세요", list(script_library[topic].keys()))
        entry = script_library[topic][question]
        question_en = entry["question_en"]
        script_text = entry["script"]
        st.markdown(f"**🗨️ 질문 (한글):** {question}")
        st.markdown(f"**🗨️ 질문 (영어):** {question_en}")
        st.markdown(f"**📘 스크립트:**\n\n{script_text}")
        if st.button("🎧 질문과 스크립트 듣기"):
            q_tts = gTTS(question_en, lang="en")
            s_tts = gTTS(script_text, lang="en")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as q_tmp:
                q_tts.save(q_tmp.name)
                st.audio(q_tmp.name)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as s_tmp:
                s_tts.save(s_tmp.name)
                st.audio(s_tmp.name)

# 오픽 문제은행
elif st.session_state.page == "오픽 문제은행":
    st.button("← 홈으로", on_click=lambda: go_to("home"))
    st.subheader("❓ 오픽 문제 생성")
    topic = st.text_input("📌 주제를 입력하세요 (예: 여행, 음악, 집안일 등)")
    if st.button("문제 생성") and topic:
        questions = generate_opic_questions(topic)
        st.markdown(questions)

# 자주 쓰는 단어 학습
elif st.session_state.page == "자주 쓰는 단어 학습":
    st.button("← 홈으로", on_click=lambda: go_to("home"))
    st.subheader("🗂 자주 쓰는 단어 학습")
    if st.button("단어 목록 보기"):
        words = get_frequent_opic_words()
        st.markdown(words)
