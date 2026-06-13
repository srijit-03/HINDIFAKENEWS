from flask import Flask, render_template, request
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# LOAD SVM MODEL
# ==============================
with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

# ==============================
# LOAD HINVEC MODEL
# ==============================
model_name = "Sailesh97/Hinvec"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

hinvec_model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True
)

hinvec_model.to(device)
hinvec_model.eval()

print("HinVec loaded successfully.")

# ==============================
# EMBEDDING FUNCTION
# ==============================
def get_hinvec_embedding(text):

    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=32
    )

    input_ids = tokens["input_ids"].to(device)

    with torch.no_grad():
        token_embeddings = (
            hinvec_model.get_input_embeddings()(input_ids)
        )

    embedding = token_embeddings.mean(dim=1)

    return embedding.squeeze().cpu().numpy()


# ==============================
# FLASK APP
# ==============================
app = Flask(__name__)


# ==============================
# HOME ROUTE
# ==============================
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        news = request.form.get("news", "").strip()

        if news == "":
            prediction = "Please enter some news text."
        else:
            try:
                # Generate embedding
                emb = get_hinvec_embedding(news)

                # Reshape for SVM
                emb = emb.reshape(1, -1)

                # Prediction
                result = svm_model.predict(emb)[0]

                if int(result) == 1:
                    prediction = "🛑 Fake News"
                else:
                    prediction = "✅ Real News"

            except Exception as e:
                prediction = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction
    )


# ==============================
# RUN APP
# ==============================
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )