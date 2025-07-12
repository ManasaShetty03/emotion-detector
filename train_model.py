import pandas as pd
import joblib
import base64
import io
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ----------------- 1. Load & Clean Dataset -----------------
df = pd.read_csv("Cleaned_Emotion_Severity_Dataset.csv", encoding="ISO-8859-1")

# Drop rows with missing or invalid text/emotion
df = df[df['Text'].apply(lambda x: isinstance(x, str) and len(x.strip()) >= 3)]
df = df[df['Emotion'].apply(lambda x: isinstance(x, str) and x.strip() != '')]

# ----------------- 2. Add Category Column -----------------
def categorize_emotion(emotion):
    emotion = str(emotion).strip().lower()
    if emotion == 'happy':
        return 'Positive'
    elif emotion == 'neutral':
        return 'Neutral'
    return 'Negative'

df['Category'] = df['Emotion'].apply(categorize_emotion)

# Blank out severity for positive and neutral rows
df.loc[df['Category'].isin(['Positive', 'Neutral']), 'Severity'] = ''

# ----------------- 3. Load BERT Model -----------------
bert = SentenceTransformer("all-mpnet-base-v2")
X = bert.encode(df['Text'].tolist())

# ----------------- 4. Train Emotion Classification Model -----------------
le_emotion = LabelEncoder()
y_emotion = le_emotion.fit_transform(df['Emotion'])

X_train_emotion, X_test_emotion, y_train_emotion, y_test_emotion = train_test_split(
    X, y_emotion, test_size=0.2, random_state=22
)

svm_emotion = SVC(kernel='linear', probability=True)
svm_emotion.fit(X_train_emotion, y_train_emotion)

y_pred_emotion = svm_emotion.predict(X_test_emotion)
emotion_acc = accuracy_score(y_test_emotion, y_pred_emotion)
print("✅ Emotion Model Accuracy:", round(emotion_acc * 100, 2), "%")

emotion_targets = [str(cls) for cls in le_emotion.classes_]
print("\nEmotion Classification Report:\n",
      classification_report(y_test_emotion, y_pred_emotion, target_names=emotion_targets))

# Save Emotion Model and Encoder
joblib.dump(svm_emotion, "svm_emotion.pkl")
joblib.dump(le_emotion, "label_encoder_emotion.pkl")

# ----------------- 5. Emotion Label Distribution Chart -----------------
fig_em, ax_em = plt.subplots()
df['Emotion'].value_counts().reindex(le_emotion.classes_).plot(kind='bar', ax=ax_em, color='royalblue')
ax_em.set_title("Target Variables - Emotion Labels")
ax_em.set_ylabel("Count")
ax_em.set_xlabel("Emotion")
ax_em.set_xticklabels(ax_em.get_xticklabels(), rotation=45)

# Save the graph as base64 image (optional)
buf_em = io.BytesIO()
fig_em.savefig(buf_em, format='png')
buf_em.seek(0)
emotion_target_graph = base64.b64encode(buf_em.getvalue()).decode()

# ----------------- 6. Train Severity Model on Negative Data Only -----------------
df_sev = df[df['Category'] == 'Negative'].copy()
df_sev = df_sev[df_sev['Severity'].apply(lambda x: isinstance(x, str) and x.strip() != '')]

X_sev = bert.encode(df_sev['Text'].tolist())
le_sev = LabelEncoder()
y_sev = le_sev.fit_transform(df_sev['Severity'])

X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
    X_sev, y_sev, test_size=0.2, random_state=42
)

svm_severity = SVC(kernel='linear', probability=True)
svm_severity.fit(X_train_sev, y_train_sev)

y_pred_sev = svm_severity.predict(X_test_sev)
sev_acc = accuracy_score(y_test_sev, y_pred_sev)
print("✅ Severity Model Accuracy:", round(sev_acc * 100, 2), "%")

sev_targets = [str(cls) for cls in le_sev.classes_]
print("\nSeverity Classification Report:\n",
      classification_report(y_test_sev, y_pred_sev, target_names=sev_targets))

# Save Severity Model and Encoder
joblib.dump(svm_severity, "svm_severity.pkl")
joblib.dump(le_sev, "label_encoder_severity.pkl")

# ----------------- 7. Dataset Summary -----------------
print("\nEmotion Distribution:\n", df['Emotion'].value_counts())
print("\nCategory Distribution:\n", df['Category'].value_counts())

# Optional: Save emotion bar chart image
with open("emotion_target_graph.png", "wb") as f:
    f.write(base64.b64decode(emotion_target_graph))
print("\n📊 Emotion distribution graph saved as 'emotion_target_graph.png'")
