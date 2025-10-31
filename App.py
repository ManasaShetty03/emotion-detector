from flask import Flask, request, render_template
import joblib
import re
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load models and encoders
bert_model = SentenceTransformer('all-mpnet-base-v2')
svm_emotion = joblib.load("svm_emotion.pkl")
le_emotion = joblib.load("label_encoder_emotion.pkl")
svm_severity = joblib.load("svm_severity.pkl")
le_severity = joblib.load("label_encoder_severity.pkl")

# Helpers
def is_meaningful(text):
    return len(text.strip()) >= 3 and re.search(r'[a-zA-Z]', text)

def categorize_emotion(emotion):
    emotion = emotion.lower()
    if emotion == 'happy':
        return 'Positive'
    elif emotion == 'neutral':
        return 'Neutral'
    return 'Negative'

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    severity = None
    recommendation = None
    error = None

    if request.method == "POST":
        user_text = request.form["user_text"].strip()

        if not is_meaningful(user_text):
            error = "Please enter a meaningful sentence."
        else:
            # Predict Emotion
            embedding = bert_model.encode([user_text])
            emotion_pred = svm_emotion.predict(embedding)[0]
            emotion = le_emotion.inverse_transform([emotion_pred])[0]
            prediction = emotion

            category = categorize_emotion(emotion)

            if category == "Negative":
                # Predict Severity
                sev_pred = svm_severity.predict(embedding)[0]
                severity = le_severity.inverse_transform([sev_pred])[0].capitalize()

                # Generate Recommendations
                emotion = emotion.lower()
                recommendation = []

                if emotion == "anger":
                    if severity == "High":
                        recommendation = [
                            "Seek professional anger management counseling.",
                            "Practice mindfulness and relaxation techniques daily.",
                            "Avoid confrontation and take space to cool down."
                        ]
                    elif severity == "Moderate":
                        recommendation = [
                            "Use physical activity like walking to release tension.",
                            "Try journaling or speaking with a close friend.",
                            "Identify triggers and avoid them when possible."
                        ]
                    elif severity == "Low":
                        recommendation = [
                            "Take deep breaths when you feel irritated.",
                            "Engage in relaxing hobbies or music.",
                            "Keep a journal to track your emotional responses."
                        ]

                elif emotion == "fear":
                    if severity == "High":
                        recommendation = [
                            "Consider exposure therapy with a licensed therapist.",
                            "Avoid isolation—stay connected with a trusted support system.",
                            "Learn about grounding techniques to manage fear responses."
                        ]
                    elif severity == "Moderate":
                        recommendation = [
                            "Join a support group or talk to a counselor.",
                            "Practice deep breathing and progressive muscle relaxation.",
                            "Reduce media exposure to triggering content."
                        ]
                    elif severity == "Low":
                        recommendation = [
                            "Try writing down your fears and rational responses.",
                            "Talk to someone you trust about what's worrying you.",
                            "Get adequate rest and maintain healthy habits."
                        ]

                elif emotion == "depression":
                    if severity == "High":
                        recommendation = [
                            "Consult a licensed therapist or psychiatrist immediately.",
                            "Avoid isolation—engage with a trusted person daily.",
                            "Reach out to crisis support lines if you're feeling unsafe."
                        ]
                    elif severity == "Medium":
                        recommendation = [
                            "Consider talking to a mental health professional.",
                            "Try behavioral activation: engage in simple daily tasks.",
                            "Stay connected with people who uplift you."
                        ]
                    elif severity == "Low":
                        recommendation = [
                            "Keep a daily gratitude journal.",
                            "Incorporate short walks or light exercise into your routine.",
                            "Avoid negative self-talk and try affirmations."
                        ]

                elif emotion == "anxiety":
                    if severity == "High":
                        recommendation = [
                            "Speak with a therapist about cognitive behavioral therapy.",
                            "Practice grounding techniques (like 5-4-3-2-1 method).",
                            "Avoid stimulants like caffeine and maintain sleep hygiene."
                        ]
                    elif severity == "Moderate":
                        recommendation = [
                            "Use meditation apps or breathing exercises.",
                            "Try limiting your screen time or exposure to stressors.",
                            "Keep a worry journal to express thoughts."
                        ]
                    elif severity == "Low":
                        recommendation = [
                            "Take slow, deep breaths during anxious moments.",
                            "Maintain a regular schedule for meals and rest.",
                            "Engage in calming routines like reading or nature walks."
                        ]

                elif emotion == "sad":
                    if severity == "High":
                        recommendation = [
                            "Reach out to a therapist to explore persistent sadness.",
                            "Engage in social activities even if you don’t feel like it.",
                            "Listen to uplifting music or engage in creative outlets."
                        ]
                    elif severity == "Moderate":
                        recommendation = [
                            "Talk about your feelings with someone you trust.",
                            "Focus on small accomplishments throughout your day.",
                            "Get sunlight exposure and physical movement daily."
                        ]
                    elif severity == "Low":
                        recommendation = [
                            "Practice self-compassion and avoid negative self-talk.",
                            "Journal your thoughts and focus on solutions.",
                            "Allow yourself to feel emotions without judgment."
                        ]

            else:
                severity = ""
                recommendation = None

    return render_template(
        "index.html",
        prediction=prediction,
        severity=severity,
        error=error,
        recommendation=recommendation
    )

if __name__ == "__main__":
    app.run(debug=True)
