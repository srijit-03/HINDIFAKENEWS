from flask import Flask, render_template, request
import pickle

# ==============================
# LOAD MODEL
# ==============================

model = pickle.load(open("svm_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

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

        news = request.form["news"]

        # Transform text
        news_vector = vectorizer.transform([news])

        # Predict
        result = model.predict(news_vector)[0]

        if result == 1:
            prediction = "🛑 Fake News"
        else:
            prediction = "✅ Real News"

    return render_template(
        "index.html",
        prediction=prediction
    )

# ==============================
# RUN APP
# ==============================

if __name__ == "__main__":
    import os

port = int(os.environ.get("PORT", 7860))

app.run(host="0.0.0.0", port=port)