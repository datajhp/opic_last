import streamlit as st
import whisper
import requests
import tempfile
from gtts import gTTS
import os
import json

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
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

@st.cache_resource
def load_script_library():
    with open("script_library.json", "r", encoding="utf-8") as f:
        return json.load(f)

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
st.title("🎧 오픽 & 영어 회화 피드백")

menu = st.radio("기능 선택", [
    "스크립트 학습","음성 피드백", "모범 답변 듣기", "문장 변환 퀴즈", "오픽 문제은행", "자주 쓰는 단어 학습"
], horizontal=True)

if menu == "음성 피드백":
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
        tts_path = media_path.replace(".mp4", "_feedback.mp3")
        tts.save(tts_path)
        st.audio(tts_path)

        os.remove(media_path)
        os.remove(tts_path)

elif menu == "모범 답변 듣기":
    st.subheader("📝 미리 준비된 질문으로 모범 답변 들어보기")
    script_library = load_script_library()

    # 주제 선택
    topic = st.selectbox("📚 주제를 선택하세요", list(script_library.keys()))

    if topic:
        questions = list(script_library[topic].keys())
        # 질문 선택
        question = st.selectbox("❓ 질문을 선택하세요", questions)

        if question:
            entry = script_library[topic][question]
            question_en = entry["question_en"]
            script_text = entry["script"]

            st.markdown(f"**🗨️ 질문 (한글):** {question}")
            st.markdown(f"**🗨️ 질문 (영어):** {question_en}")
            st.markdown(f"**📘 모범 답변:**\n\n{script_text}")

            if st.button("🎧 모범 답변 듣기"):
                from gtts import gTTS
                import tempfile

                # TTS 생성 → 임시 파일로 저장
                tts = gTTS(script_text, lang="en")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tts.save(tmp.name)
                    st.audio(tmp.name)


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

elif menu == "스크립트 학습":
    st.subheader("🎙️ 주제별 스크립트를 선택해 들어보세요")

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
            st.markdown(f"**📘 스크립트:** {script_text}")

            if st.button("🎧 질문과 스크립트 듣기"):

                gTTS(question_en, lang="en").save("질문.mp3")
                st.audio("질문.mp3")

                gTTS(script_text, lang="en").save("답변 스크립트.mp3")

st.markdown(
    """
    <style>
        /* 상단 여백 제거 */
        .block-container {
            padding-top: 0rem !important;
        }

        /* 헤더 Share / Edit / GitHub 아이콘 숨김 */
        header[data-testid="stHeader"] {
            visibility: hidden;
        }

        /* 좌우 여백 줄이기 (선택 사항) */
        .main .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* 파일 업로드 위 여백 제거 (탭이 있을 경우) */
        [data-baseweb="tab-list"] {
            margin-bottom: 0rem !important;
        }
        [data-baseweb="tab-list"] + div {
            margin-top: 0rem !important;
            padding-top: 0rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)
