import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("dataset.csv")

# Keep required columns
df = df[["text", "label"]]

# Remove null values
df.dropna(inplace=True)

# Convert to string
df["text"] = df["text"].astype(str)

print("Dataset Loaded:", len(df))

# ==============================
# FEATURES & LABELS
# ==============================

X = df["text"]
y = df["label"]

# ==============================
# TF-IDF VECTORIZER
# ==============================

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_vectorized = vectorizer.fit_transform(X)

# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================
# SVM MODEL
# ==============================

model = SVC(kernel='linear')

print("Training SVM Model...")

model.fit(X_train, y_train)

print("Training Complete!")

# ==============================
# EVALUATION
# ==============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# ==============================
# SAVE MODEL
# ==============================

pickle.dump(model, open("svm_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model saved successfully!")