import streamlit as st
from sentence_transformers import SentenceTransformer
import joblib
import re
import time
import speech_recognition as sr

# -------------------- 🎙️ Voice Recognition --------------------
def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... please speak clearly.")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.success(f"🗣️ You said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I couldn’t understand that.")
        return None
    except sr.RequestError:
        st.error("Speech recognition service is unavailable.")
        return None

# -------------------- 💡 Suggestion Generator --------------------
def get_suggestion(emotion: str):
    emotion = emotion.lower().strip()

    suggestions = {
        "sad": [
            "It’s okay to feel sad — allow yourself to rest and process your feelings.",
            "Talk to someone you trust about what’s weighing on your mind.",
            "Try writing or journaling your emotions to understand them better.",
            "Take a short walk outside — sunlight and fresh air can lift your mood.",
            "Listen to gentle or uplifting music to comfort yourself."
        ],
        "anger": [
            "Pause and take slow, deep breaths before reacting.",
            "Try channeling your energy into exercise or creativity.",
            "Identify what triggered your anger — awareness helps calm it.",
            "Take a short break to cool off before continuing a conversation.",
            "Remind yourself that staying calm keeps your power."
        ],
        "fear": [
            "Fear often exaggerates danger — remind yourself you are safe right now.",
            "Ground yourself with the 5-4-3-2-1 method (see, touch, hear, smell, taste).",
            "Focus on what you can control and release what you can’t.",
            "Talk about your fears with someone you trust — it lightens the burden.",
            "Breathe deeply and visualize a peaceful scene."
        ],
        "depression": [
            "Reach out to someone you trust — you don’t have to face this alone.",
            "Try small daily goals like eating on time or stepping outside.",
            "Engage in activities you once enjoyed, even for a few minutes.",
            "Maintain a routine with adequate sleep and self-care.",
            "Be kind to yourself — progress is made one step at a time."
        ],
        "anxiety": [
            "Take slow, deep breaths and focus on your heartbeat.",
            "Reduce caffeine and limit social media during stressful times.",
            "Write down your worries to separate thoughts from reality.",
            "Try mindfulness — stay present with what’s around you.",
            "Listen to soothing sounds or meditate to relax your mind."
        ]
    }

    return suggestions.get(emotion, ["Stay positive and take care of yourself 💙."])

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

# -------------------- 🧠 Helper Functions --------------------
def is_meaningful(text):
    return len(text.strip()) >= 2 and re.search(r"[a-zA-Z]", text)

def categorize_emotion(emotion):
    emotion = emotion.lower()
    if emotion == "happy":
        return "Positive"
    elif emotion == "neutral":
        return "Neutral"
    return "Negative"

# -------------------- 💬 Streamlit Setup --------------------
st.set_page_config(page_title="💬 Emotion Chatbot", page_icon="🧠", layout="centered")

st.markdown("""
    <h2 style='text-align:center; color:#6C63FF;'>🧠 Real-Time Emotion Chatbot</h2>
    <p style='text-align:center;'>Chat or speak with me — I’ll understand your emotions, analyze severity, and give helpful suggestions 💬</p>
""", unsafe_allow_html=True)

# -------------------- 💭 Initialize Chat History --------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey there 👋 I'm here to listen. How are you feeling today?"}
    ]

# -------------------- 💭 Display Chat --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- 🎤 Input Options (Text + Voice) --------------------
col1, col2 = st.columns(2)

with col1:
    prompt = st.chat_input("Type your message here...")

with col2:
    if st.button("🎙️ Speak"):
        spoken_text = recognize_speech_from_mic()
        if spoken_text:
            prompt = spoken_text
        else:
            prompt = None

# -------------------- 💬 Process Input --------------------
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not is_meaningful(prompt):
        response = "Please enter a meaningful message 🫶"
    else:
        embedding = bert.encode([prompt])

        # Predict Emotion
        emotion_pred = svm_emotion.predict(embedding)[0]
        emotion = le_emotion.inverse_transform([emotion_pred])[0]
        category = categorize_emotion(emotion)

        # Predict Severity (only for negative emotions)
        if category == "Negative":
            sev_pred = svm_severity.predict(embedding)[0]
            severity = le_severity.inverse_transform([sev_pred])[0].capitalize()
        else:
            severity = "None"

        # Typing animation
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your emotion..."):
                time.sleep(1.5)

            if emotion.lower() in ["happy", "neutral"]:
                response = f"You seem **{emotion.lower()}** 😊. That's wonderful! Keep spreading positivity!"
            else:
                suggestions = get_suggestion(emotion)
                suggestion_text = "\n".join([f"- {tip}" for tip in suggestions])
                response = (
                    f"**Emotion:** `{emotion}` | **Severity:** `{severity}`\n\n"
                    f"**💡 Suggestions:**\n{suggestion_text}"
                )

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
