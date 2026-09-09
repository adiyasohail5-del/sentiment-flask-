# AI Sentiment Analysis using DistilBERT and Flask

This project is a sentiment analysis web application built using Flask and a fine-tuned DistilBERT Transformer model.

## Features

- Positive, Negative and Neutral sentiment classification
- Fine-tuned DistilBERT model
- Confidence score
- Text preprocessing
- Interactive Flask web interface
- Responsive UI

## Technologies

- Python
- Flask
- PyTorch
- Hugging Face Transformers
- DistilBERT
- HTML
- CSS

## Sentiment Classes

| Rating | Sentiment |
|--------|-----------|
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

## Project Structure

sentiment_flask/

├── app.py  
├── requirements.txt  
├── README.md  
├── sentiment_model/  
├── templates/  
│   └── index.html  
└── static/  
    └── style.css  

## Installation

Create a virtual environment:

```bash
python -m venv venv