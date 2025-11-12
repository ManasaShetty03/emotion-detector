import streamlit as st
from sentence_transformers import SentenceTransformer
import joblib
import time
import speech_recognition as sr
import requests
from googletrans import Translator

translator = Translator()

# -------------------- 🎙️ Voice Recognition --------------------
def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... please speak clearly...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language='kn-IN')
        st.success(f"🗣️ You said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I couldn’t understand that.")
        return None
    except sr.RequestError:
        st.error("Speech recognition service unavailable.")
        return None

# -------------------- 💡 LLM Query --------------------
def query_llm(user_input, emotion, severity=None):
    if emotion == "Unknown":
        emotion = "neutral"

    prompt = f"You are an empathetic assistant. The user feels **{emotion}**"
    if severity:
        prompt += f" with severity **{severity}**"
    prompt += f". Provide a short, caring, and motivating response.\n\nUser input: {user_input}"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "vicuna", "prompt": prompt, "stream": False},
            timeout=90
        )
        return response.json().get("response", "⚠️ LLM did not return a response.")
    except Exception as e:
        return f"❌ LLM connection error: {e}"

# -------------------- 🧩 Load Models --------------------
@st.cache_resource
def load_models():
    bert = SentenceTransformer('all-mpnet-base-v2')
    svm_emotion = joblib.load("svm_emotion.pkl")
    le_emotion = joblib.load("label_encoder_emotion.pkl")
    svm_severity = joblib.load("svm_severity.pkl")
    le_severity = joblib.load("label_encoder_severity.pkl")
    return bert, svm_emotion, le_emotion, svm_severity, le_severity

bert, svm_emotion, le_emotion, svm_severity, le_severity = load_models()

# -------------------- 💬 Streamlit Setup --------------------
st.set_page_config(page_title="💬 Human Mental Health Chat", page_icon="🧠", layout="centered")

st.markdown("""
    <h2 style='text-align:center; color:#6C63FF;'>💬 Human Mental Health Chat</h2>
    <p style='text-align:center;'>I’m here to listen and help — speak or type freely 💙</p>
    <hr style="border:1px solid #ddd;">
""", unsafe_allow_html=True)

# -------------------- 💭 Chat History --------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey there 👋 I'm here to listen. How are you feeling today?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- 🔽 Input Area --------------------
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("Type your message here...", key="user_text")
with col2:
    if st.button("🎤 Voice Input"):
        user_input = recognize_speech_from_mic()

# -------------------- 💬 Handle Inputs --------------------
def is_meaningful(text): return len(text.strip()) >= 2
def needs_severity(emotion): return emotion.lower() not in ["happy", "neutral"]

def translate_to_english(text):
    return translator.translate(text, src='kn', dest='en').text if text else ""

def translate_to_kannada(text):
    return translator.translate(text, src='en', dest='kn').text if text else ""

if user_input and is_meaningful(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    is_kannada_input = any("\u0C80" <= char <= "\u0CFF" for char in user_input)
    input_for_model = translate_to_english(user_input) if is_kannada_input else user_input

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your input..."):
            time.sleep(1)

            embedding = bert.encode([input_for_model])
            try:
                emotion_pred = svm_emotion.predict(embedding)[0]
                emotion = le_emotion.inverse_transform([emotion_pred])[0]
            except:
                emotion = "Unknown"

            severity = None
            if needs_severity(emotion):
                try:
                    severity_pred = svm_severity.predict(embedding)[0]
                    severity = le_severity.inverse_transform([severity_pred])[0].capitalize()
                except:
                    severity = "Unknown"

            llm_response = query_llm(input_for_model, emotion, severity)
            if llm_response.startswith("❌"):
                llm_response = "Stay calm and positive 💙. Take a deep breath or a short walk."

            if is_kannada_input:
                emotion_display = translate_to_kannada(emotion)
                llm_response = translate_to_kannada(llm_response)
            else:
                emotion_display = emotion

            st.markdown(f"**Emotion:** `{emotion_display}` | **Severity:** `{severity or '—'}`\n\n💡 **Suggestion:** {llm_response}")

    st.session_state.messages.append({"role": "assistant", "content": llm_response})
