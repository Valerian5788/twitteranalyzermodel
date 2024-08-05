# app.py
from flask import Flask, request, jsonify
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizerFast
import torch

app = Flask(__name__)

# Charger le modèle et le tokenizer
model = XLMRobertaForSequenceClassification.from_pretrained('./model')
tokenizer = XLMRobertaTokenizerFast.from_pretrained('./model')

def classify_tweet(tweet):
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = torch.argmax(probs, dim=1).item()
    return label

@app.route('/classify', methods=['POST'])
def classify():
    tweets = request.json['tweets']
    results = [classify_tweet(tweet) for tweet in tweets]
    return jsonify(results)

if __name__ == '__main__':
    app.run(port=5001)
