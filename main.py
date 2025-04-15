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
import gdown
import zipfile
import os

app = Flask(__name__)

# ✅ Correct Google Drive zip file link for gdown (with id)
zip_file_link = 'https://drive.google.com/uc?id=1Wnu5n8t6hkrlpYHXrjRQzTSGzMl3mBsn'
model_dir = './NLPProjectModels'
zip_path = 'NLPProjectModels.zip'

# ✅ Download and unzip models (only if not already unzipped)
def download_and_unzip_model():
    if not os.path.exists(model_dir):
        print("Downloading .zip file...")
        gdown.download(zip_file_link, zip_path, quiet=False)

        print("Unzipping the .zip file...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
            print("Models unzipped successfully.")
        except zipfile.BadZipFile:
            print("❌ Error: File is not a valid zip file.")
        os.remove(zip_path)

download_and_unzip_model()

# Paths to model folders
sentiment_model_path = os.path.join(model_dir, 'sentiment_model')
sentiment_tokenizer_path = os.path.join(model_dir, 'sentiment_tokenizer')
translate_it_to_eng_model_path = os.path.join(model_dir, 'translation_ita_to_eng_model')
translate_it_to_eng_tokenizer_path = os.path.join(model_dir, 'translation_ita_to_eng_tokenizer')
translate_eng_to_it_model_path = os.path.join(model_dir, 'translation_model_en_to_it')
translate_eng_to_it_tokenizer_path = os.path.join(model_dir, 'translation_tokenizer_en_to_it')

# Load models and tokenizers
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_tokenizer_path)

translate_it_to_eng_model = MarianMTModel.from_pretrained(translate_it_to_eng_model_path)
translate_it_to_eng_tokenizer = MarianTokenizer.from_pretrained(translate_it_to_eng_tokenizer_path)

translate_eng_to_it_model = MarianMTModel.from_pretrained(translate_eng_to_it_model_path)
translate_eng_to_it_tokenizer = MarianTokenizer.from_pretrained(translate_eng_to_it_tokenizer_path)

# Load NER and POS pipelines
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
    inputs = sentiment_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
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

# ✅ DO NOT USE app.run() on Railway – use this instead:
if __name__ == "__main__":
    from os import getenv
    app.run(host="0.0.0.0", port=int(getenv("PORT", 5000)))
