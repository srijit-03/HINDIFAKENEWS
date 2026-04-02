from flask import Flask, request
import numpy as np
from model import get_hinvec_embedding, CNNModel
import torch

app = Flask(__name__)

# Load CNN model
input_dim = 768  # HinVec embedding size

model = CNNModel(input_dim)
model.load_state_dict(torch.load("cnn_model.pth"))
model.eval()

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        news = request.form["news"]

        emb = get_hinvec_embedding(news)

        # ✅ FIXED INDENTATION
        emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)

        # ✅ inside POST block
        with torch.no_grad():
            output = model(emb_tensor)
            _, pred = torch.max(output, 1)

        pred = pred.item()

        prediction = "Fake News" if pred == 1 else "Real News"

    return f"""
    <h2>Fake News Detection</h2>
    <form method="post">
        <textarea name="news" rows="6" cols="60"></textarea><br><br>
        <button type="submit">Check</button>
    </form>
    <h3>{prediction}</h3>
    """

if __name__ == "__main__":
    app.run(debug=True)