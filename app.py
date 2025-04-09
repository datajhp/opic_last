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

# 모범 답변 생성 함수
def get_model_answer(question):
    prompt = f"""
    Please generate a model answer to the following OPIc question in natural spoken English, and provide a Korean translation.
    Question: {question}
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a native English speaker who gives model answers for OPIc."},
            {"role": "user", "content": prompt}
        ]
    }
    res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
    return res.json()["choices"][0]["message"]["content"]

# 문장 변환 퀴즈 함수
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

# 문제은행 질문 생성 함수
def generate_opic_questions(topic):
    prompt = f"""
    Generate 3 realistic OPIc questions for the topic: "{topic}". Include Korean translations.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are an expert OPIc question generator."},
            {"role": "user", "content": prompt}
        ]
    }
    res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
    return res.json()["choices"][0]["message"]["content"]

# 자주 쓰는 단어 학습 기능
def get_frequent_opic_words():
    prompt = """
    Please provide a list of 10 commonly used English words or phrases in OPIc speaking tests.

    Format each item as:
    1. [English phrase] - [Korean meaning] (Example: [English sentence] / [Korean translation])

    Requirements:
    - Use only English and Korean. Do not use any other language (e.g., Japanese, Chinese).
    - Ensure that all English expressions and example sentences are natural and grammatically correct.
    - All Korean translations should be clear and accurate.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are an OPIc tutor. Only use English and Korean. Be accurate and natural."},
            {"role": "user", "content": prompt}
        ]
    }
    res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
    return res.json()["choices"][0]["message"]["content"]

# Streamlit 앱 UI
st.set_page_config(layout="centered")
st.title("🎧 영어 회화 피드백 & 오픽 연습 (Groq + Whisper)")

menu = st.radio("기능 선택", [
    "음성 피드백", "모범 답변 듣기", "문장 변환 퀴즈", "오픽 문제은행", "자주 쓰는 단어 학습"
], horizontal=True)

if menu == "음성 피드백":
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

        os.remove(media_path)
        os.remove(tts_path)

elif menu == "모범 답변 듣기":
    question = st.text_input("📝 오픽 질문 입력")
    if st.button("모범 답변 보기") and question:
        answer = get_model_answer(question)
        st.markdown(answer)
        tts = gTTS(answer, lang="en")
        tts_path = "model_answer.mp3"
        tts.save(tts_path)
        st.audio(tts_path)

elif menu == "문장 변환 퀴즈":
    sentence = st.text_input("✏️ 변환하고 싶은 문장 입력")
    if st.button("자연스럽게 바꾸기") and sentence:
        result = transform_quiz(sentence)
        st.markdown(result)

elif menu == "오픽 문제은행":
    topic = st.text_input("📚 주제 입력 (예: 여행, 음악, 집안일 등)")
    if st.button("문제 생성") and topic:
        questions = generate_opic_questions(topic)
        st.markdown(questions)

elif menu == "자주 쓰는 단어 학습":
    if st.button("단어 목록 보기"):
        words = get_frequent_opic_words()
        st.markdown(words)
