from flask import Flask, request, jsonify, render_template
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    MarianMTModel,
    MarianTokenizer,
    pipeline
)
import torch
from torch.nn.functional import softmax
import os
import gdown
import zipfile

app = Flask(__name__)

# --- Constants for Google Drive Download ---
MODEL_ZIP_ID = "1Wnu5n8t6hkrlpYHXrjRQzTSGzMl3mBsn"  # <- Keep your actual ID here
ZIP_PATH = "NLPProjectModels.zip"
EXTRACT_DIR = "NLPProjectModels"

def download_and_extract_models():
    if not os.path.exists(EXTRACT_DIR):
        print("Downloading model zip from Google Drive...")
        url = f"https://drive.google.com/uc?id={MODEL_ZIP_ID}"
        gdown.download(url, ZIP_PATH, quiet=False)

        print("Extracting models...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")

        print("Cleaning up...")
        os.remove(ZIP_PATH)
    else:
        print("Model folder already exists, skipping download.")

download_and_extract_models()

# --- Load Models & Tokenizers ---
sentiment_model_path = os.path.join(EXTRACT_DIR, "sentiment_model")
sentiment_tokenizer_path = os.path.join(EXTRACT_DIR, "sentiment_tokenizer")
translate_it_to_eng_model_path = os.path.join(EXTRACT_DIR, "translate_ita_to_eng_model")
translate_it_to_eng_tokenizer_path = os.path.join(EXTRACT_DIR, "translate_ita_to_eng_tokenizer")
translate_eng_to_it_model_path = os.path.join(EXTRACT_DIR, "translate_eng_to_it_model")
translate_eng_to_it_tokenizer_path = os.path.join(EXTRACT_DIR, "translate_eng_to_it_tokenizer")

# Load models and tokenizers
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_tokenizer_path)

translate_it_to_eng_model = MarianMTModel.from_pretrained(translate_it_to_eng_model_path)
translate_it_to_eng_tokenizer = MarianTokenizer.from_pretrained(translate_it_to_eng_tokenizer_path)

translate_eng_to_it_model = MarianMTModel.from_pretrained(translate_eng_to_it_model_path)
translate_eng_to_it_tokenizer = MarianTokenizer.from_pretrained(translate_eng_to_it_tokenizer_path)

# Load NER and POS pipelines (use pre-trained from HF hub)
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
pos_pipeline = pipeline("token-classification", model="vblagoje/bert-english-uncased-finetuned-pos", grouped_entities=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text')
    target_lang = data.get('target_lang')  # 'it' or 'en'

    if not text or not target_lang:
        return jsonify({"error": "Missing input text or target_lang"}), 400

    # Translation
    if target_lang == 'it':
        model = translate_eng_to_it_model
        tokenizer = translate_eng_to_it_tokenizer
    elif target_lang == 'en':
        model = translate_it_to_eng_model
        tokenizer = translate_it_to_eng_tokenizer
    else:
        return jsonify({"error": "Unsupported language"}), 400

    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    # Sentiment
    sentiment_inputs = sentiment_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = sentiment_model(**sentiment_inputs)
        logits = outputs.logits
        probs = softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    # NER and POS tagging
    ner_results = ner_pipeline(text)
    pos_results = pos_pipeline(text)

    ner_entities = [f"{ent['word']} ({ent['entity_group']})" for ent in ner_results]
    pos_tags = [f"{ent['word']} ({ent['entity']})" for ent in pos_results]

    return jsonify({
        "translated_text": translated_text,
        "sentiment_class": predicted_class,
        "confidence": round(probs[0][predicted_class].item(), 4),
        "ner": ner_entities,
        "pos": pos_tags
    })

if __name__ == '__main__':
    app.run(debug=True)
