import pandas as pd
import numpy as np
import torch
import pickle

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# LOAD DATA
# ==============================
def load_data():
    path = "dataset.csv"

    df = pd.read_csv(path)
    df = df[["text", "label"]].dropna()

    df["text"] = df["text"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna()

    print("Dataset loaded:", len(df))
    return df


# ==============================
# LOAD HINVEC MODEL
# ==============================
def load_hinvec():
    model_name = "Sailesh97/Hinvec"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model.to(device)
    model.eval()

    return tokenizer, model


# ==============================
# GET EMBEDDING
# ==============================
def get_hinvec_embedding(text, tokenizer, model):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=32
    )

    input_ids = tokens["input_ids"].to(device)

    with torch.no_grad():
        token_embeddings = model.get_input_embeddings()(input_ids)

    embedding = token_embeddings.mean(dim=1)

    return embedding.squeeze().cpu().numpy()


# ==============================
# GENERATE EMBEDDINGS
# ==============================
def generate_embeddings(df, tokenizer, model):
    X = []
    y = df["label"].values

    print("Generating embeddings...")

    for text in df["text"]:
        emb = get_hinvec_embedding(str(text), tokenizer, model)
        X.append(emb)

    X = np.array(X)

    print("Embedding Shape:", X.shape)

    return X, y


# ==============================
# TRAIN SVM
# ==============================
def train_and_save_model():
    df = load_data()
    tokenizer, hinvec_model = load_hinvec()

    X, y = generate_embeddings(df, tokenizer, hinvec_model)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # SVM MODEL
    svm_model = SVC(
        kernel='linear',
        probability=True,
        random_state=42
    )

    print("Training SVM...")
    svm_model.fit(X_train, y_train)

    # Prediction
    y_pred = svm_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)

    # Save model
    with open("svm_model.pkl", "wb") as f:
        pickle.dump(svm_model, f)

    print("Model saved as svm_model.pkl")


# ==============================
# LOAD SAVED MODEL
# ==============================
def load_trained_model():
    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)

    return model


# ==============================
# PREDICT FUNCTION
# ==============================
def predict(text, model, tokenizer, hinvec_model):

    emb = get_hinvec_embedding(
        text,
        tokenizer,
        hinvec_model
    )

    emb = emb.reshape(1, -1)

    prediction = model.predict(emb)[0]

    return int(prediction)


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    train_and_save_model()