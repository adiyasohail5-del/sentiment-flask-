from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import contractions

app = Flask(__name__)


# Load fine-tuned DistilBERT model


import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()


# Text preprocessing
# Same preprocessing used during training


def preprocess_text(text):
    text = str(text)

    # Expand contractions
    text = contractions.fix(text)

    # Handle emojis / non-ASCII characters
    text = text.encode("ascii", "ignore").decode("ascii")

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text



# Sentiment prediction


def predict_sentiment(review):

    cleaned_text = preprocess_text(review)

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[0][predicted_class].item()

    # Get label directly from model configuration
    sentiment = model.config.id2label[predicted_class]

    return sentiment, confidence, cleaned_text



# Home page


@app.route("/", methods=["GET", "POST"])
def home():

    sentiment = None
    confidence = None
    review = ""
    cleaned_text = ""

    if request.method == "POST":

        review = request.form.get("review", "").strip()

        if review:

            sentiment, confidence, cleaned_text = predict_sentiment(
                review
            )

            confidence = round(confidence * 100, 2)

    return render_template(
        "index.html",
        sentiment=sentiment,
        confidence=confidence,
        review=review,
        cleaned_text=cleaned_text
    )



# Run Flask


if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )