import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# ------------------ 1. Load Dataset ------------------
df = pd.read_csv("Dataset.csv", encoding="ISO-8859-1")

# Basic cleaning
df = df[df['Text'].apply(lambda x: isinstance(x, str) and len(x.strip()) >= 3)]
df = df[df['Emotion'].apply(lambda x: isinstance(x, str) and x.strip() != '')]

# ------------------ 2. Add Category Column ------------------
def categorize_emotion(emotion):
    emotion = str(emotion).strip().lower()
    if emotion == 'happy':
        return 'Positive'
    elif emotion == 'neutral':
        return 'Neutral'
    return 'Negative'

df['Category'] = df['Emotion'].apply(categorize_emotion)
df.loc[df['Category'].isin(['Positive', 'Neutral']), 'Severity'] = ''

print(f"✅ Dataset Loaded: {len(df)} records")

# ------------------ 3. Sentence Embeddings ------------------
print("🔍 Generating sentence embeddings...")
bert = SentenceTransformer("all-mpnet-base-v2")
X = bert.encode(df['Text'].tolist())

# ------------------ 4. Train Emotion Classifier ------------------
le_emotion = LabelEncoder()
y_emotion = le_emotion.fit_transform(df['Emotion'])

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X, y_emotion, test_size=0.2, random_state=22
)

svm_emotion = SVC(kernel='linear', probability=True, class_weight='balanced')
svm_emotion.fit(X_train_e, y_train_e)

y_pred_e = svm_emotion.predict(X_test_e)
emotion_acc = accuracy_score(y_test_e, y_pred_e)
print(f"\n✅ Emotion Model Accuracy: {emotion_acc*100:.2f}%")
print("\nEmotion Classification Report:\n", classification_report(y_test_e, y_pred_e, target_names=le_emotion.classes_))

# Save models
joblib.dump(svm_emotion, "svm_emotion.pkl")
joblib.dump(le_emotion, "label_encoder_emotion.pkl")

# ------------------ 5. Train Severity Classifier ------------------
df_neg = df[df['Category'] == 'Negative'].copy()
df_neg = df_neg[df_neg['Severity'].apply(lambda x: isinstance(x, str) and x.strip() != '')]

if len(df_neg) > 0:
    X_sev = bert.encode(df_neg['Text'].tolist())
    le_sev = LabelEncoder()
    y_sev = le_sev.fit_transform(df_neg['Severity'])

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_sev, y_sev, test_size=0.2, random_state=5
    )

    svm_sev = SVC(kernel='linear', probability=True)
    svm_sev.fit(X_train_s, y_train_s)

    y_pred_s = svm_sev.predict(X_test_s)
    sev_acc = accuracy_score(y_test_s, y_pred_s)
    print(f"\n✅ Severity Model Accuracy: {sev_acc*100:.2f}%")
    print("\nSeverity Classification Report:\n", classification_report(y_test_s, y_pred_s, target_names=le_sev.classes_))

    # Save models
    joblib.dump(svm_sev, "svm_severity.pkl")
    joblib.dump(le_sev, "label_encoder_severity.pkl")
else:
    print("⚠️ No negative samples found with valid severity. Skipping severity model training.")

# ------------------ 6. Visualizations ------------------
os.makedirs("plots", exist_ok=True)

# Emotion Confusion Matrix
conf_e = confusion_matrix(y_test_e, y_pred_e)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_e, annot=True, fmt='d', cmap='Blues', xticklabels=le_emotion.classes_, yticklabels=le_emotion.classes_)
plt.title("Emotion Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/emotion_confusion_matrix.png")
plt.close()

# Emotion Distribution
plt.figure(figsize=(6, 4))
df['Emotion'].value_counts().plot(kind='bar', color='royalblue')
plt.title("Emotion Label Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/emotion_distribution.png")
plt.close()

# Category Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Category', data=df, palette='Set2')
plt.title("Emotion Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/category_distribution.png")
plt.close()

print("\n📊 Training Complete — Models and plots saved successfully!")
