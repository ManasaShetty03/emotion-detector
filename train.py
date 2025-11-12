import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

# ------------------ Load Dataset ------------------
df = pd.read_csv("Dataset.csv", encoding="ISO-8859-1")

# Keep only valid rows
df = df[df['Text'].apply(lambda x: isinstance(x, str) and len(x.strip()) >= 3)]
df = df[df['Emotion'].apply(lambda x: isinstance(x, str) and x.strip() != '')]
df = df[df['Severity'].apply(lambda x: isinstance(x, str) and x.strip() != '')]

print(f"✅ Loaded {len(df)} rows from Dataset.csv")

# ------------------ Encode Text ------------------
bert = SentenceTransformer("all-mpnet-base-v2")
X = bert.encode(df['Text'].tolist())

# ------------------ Emotion Model ------------------
le_emotion = LabelEncoder()
y_emotion = le_emotion.fit_transform(df['Emotion'])
svm_emotion = SVC(kernel='linear', probability=True, class_weight='balanced')
svm_emotion.fit(X, y_emotion)

joblib.dump(svm_emotion, "svm_emotion.pkl")
joblib.dump(le_emotion, "label_encoder_emotion.pkl")
print("✅ Emotion model trained and saved!")

# ------------------ Severity Model ------------------
le_severity = LabelEncoder()
y_severity = le_severity.fit_transform(df['Severity'])
svm_severity = SVC(kernel='linear', probability=True, class_weight='balanced')
svm_severity.fit(X, y_severity)

joblib.dump(svm_severity, "svm_severity.pkl")
joblib.dump(le_severity, "label_encoder_severity.pkl")
print("✅ Severity model trained and saved!")
