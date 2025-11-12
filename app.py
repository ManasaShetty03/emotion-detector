import streamlit as st

# Page setup
st.set_page_config(page_title="🧠 Human Mental Health Analysis", page_icon="🧠", layout="wide")

# Custom background and styling
st.markdown("""
    <style>
        body {
            background-image: url('bg.png');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            height: 100vh;
            margin: 0;
        }

        .main-title {
            font-size: 48px;
            font-weight: bold;
            color: white;
            text-align: center;
            text-shadow: 2px 2px 6px rgba(0,0,0,0.6);
            margin-top: 100px;
        }

        .subtitle {
            font-size: 20px;
            color: white;
            text-align: center;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
            margin-top: 10px;
            margin-bottom: 60px;
        }

        .button-container {
            display: flex;
            justify-content: center;
            position: absolute;
            bottom: 120px;
            width: 100%;
        }

        .start-button {
            background-color: #6C63FF;
            color: white;
            font-size: 22px;
            font-weight: bold;
            padding: 15px 40px;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }

        .start-button:hover {
            background-color: #5a52e0;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# Content layout
st.markdown("<h1 class='main-title'>🧠 Human Mental Health Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Understand your emotions, speak freely, and get empathetic guidance.</p>", unsafe_allow_html=True)

# Centered button at the bottom
st.markdown("<div class='button-container'>", unsafe_allow_html=True)
if st.button("🚀 Start", key="start_button"):
    st.switch_page("pages/chat_page.py")
st.markdown("</div>", unsafe_allow_html=True)
