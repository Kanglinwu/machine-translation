"""
Translation Service API

Purpose:
This Flask-based web application provides a translation service using the Flask, FastText language detection model,
and NLLB-200 translation model from Facebook's transformers library.
It is designed to detect the source language of a given input text, 
and translate it into a target language using a specified machine translation model.

Naming Convention:
Snake Case (e.g., model_ft, model_name)
"""

from flask import Flask, request, jsonify
import logging
from logging.handlers import RotatingFileHandler
from flask_cors import cross_origin
import fasttext
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import json
from flask import Flask, request, jsonify, Response
import os

# Configure Logging
# ------------------------------------------------------------------------------------
log_file_path = 'logs/api.log'  # Adjust the path based on your Docker volume setup
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Create the logs directory if it doesn't exist

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# ------------------------------------------------------------------------------------

# Create a rotating file handler
# ------------------------------------------------------------------------------------
handler = RotatingFileHandler(log_file_path, maxBytes=5*1024*1024, backupCount=5)  # 5 MB per log file, keep 5 backups
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
# ------------------------------------------------------------------------------------


app = Flask(__name__)

# TODO
# Post Training Quantization
# 防呆

# Load FastText language detection model
model_ft = fasttext.load_model('models/lid.176.bin')

# Load translation model and tokenizer
model_name = "facebook/nllb-200-distilled-600M"
cache_dir = "./models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trans_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16)
trans_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16)
trans_model.to(device)

torch.backends.cuda.enable_flash_sdp = True  # This enables Flash Attention (SDPA)
torch.backends.cuda.enable_mem_efficient_sdp = True  # Use memory efficient SDPA (if supported)
torch.backends.cuda.enable_math_sdp = False  # Use this flag to revert to math-based SDPA if needed (less efficient)

# Define translation pipeline
translator = pipeline(
    'translation',
    model=trans_model,
    tokenizer=trans_tokenizer,
    max_length=512,
    device=0 if torch.cuda.is_available() else -1
)

# Retrieve the value of an environment variable named "det_conf"
det_conf = os.getenv("det_conf")

# Language code translation for the Translation model
target_languages = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "es": "spa_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "ko": "kor_Hang",
    "ja": "jpn_Jpan",
    "hi": "hin_Deva",
    "km": "khm_Khmr",
    "pt-br": "por_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "tr": "tur_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "no": "nob_Latn",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "hu": "hun_Latn",
}

@app.route('/translate', methods=['POST'])
@cross_origin()
def translate():
    """
    Function: Translate

    Purpose:
    This function handles POST requests to the '/translate' endpoint. It takes input text and a target language,
    detects the source language of the input text using the FastText model, and translates the text into the target language.

    Parameters:
    - request: A Flask request object containing the input text and target language in the JSON payload.
    {
        "msg": "Text to be translated.",
        "target_lang": "en"
    }

    Return:
    - A Flask response object containing the source language, a flag indicating whether translation occurred,
    and the translated text in the target language.
    {
        "source_lang": "es",
        "is_trans": true,
        "target_msg": "Translated text here."
        "model": "Model used."
    }
    """

    # Check if `det_conf` is valid or not
    # ------------------------------------------------------------------------------------
    if not det_conf:
        error_msg = "det_conf is not set."
        logger.error(error_msg)  # Log the error
        return jsonify({"error": error_msg}), 400  # Return a 400 Bad Request response

    try:
        det_conf_float = float(det_conf)  # Attempt to convert to float
    except ValueError:
        error_msg = f"Invalid det_conf value: {det_conf}. Must be a float between 0 and 1."
        logger.error(error_msg)  # Log the error
        return jsonify({"error": error_msg}), 400  # Return a 400 Bad Request response

    # Check if det_conf is within the valid range
    if not (0 <= det_conf_float <= 1):
        error_msg = "Valid det_conf required (must be a float between 0 and 1)."
        logger.error(error_msg)  # Log the error
        return jsonify({"error": error_msg}), 400  # Return a 400 Bad Request response

    # ------------------------------------------------------------------------------------

    data = request.get_json()  # 從 JSON 請求中提取數據
    logger.info(f'Received request data: {data}')

    msg = data.get("msg") # Text to be translated
    target_lang = data.get("target_lang") # Language to be translated

    if target_lang in target_languages.keys():
        target_lang = target_languages[target_lang]

    else:
        if not target_lang:
            error_msg = "No target language provided."
            logger.error(error_msg)  # Log the error
            return jsonify({"error": error_msg}), 500

        else:
            error_msg = "Invalid taraget language."
            logger.error(error_msg)  # Log the error
            return jsonify({"error": error_msg}), 500

    # Initialize response object with default values for source language, translation flag, and translated text
    response_text = {
        "source_lang": "",
        "is_trans": False,
        "target_msg": msg,
        "model": "None"
    }

    # Check if input text is provided
    if not msg:
        error_msg = "No input text provided."
        logger.error(error_msg)  # Log the error
        return jsonify({"error": error_msg}), 500

    # Clean input text
    clean_text = msg.replace('\n', ' ').replace('\r', ' ')
    
    # Language Detection
    prediction = model_ft.predict(clean_text, k=1)
    source_lang = prediction[0][0].replace('__label__', '') # Predicted language
    confidence = prediction[1][0] # Language confidence
            
    response_text["source_lang"] = source_lang
    logger.info(f'Detected language: {source_lang}, Confidence: {confidence}')  # Log detected language and confidence

    # Check if source language detected is the same as the target language
    if target_languages[source_lang] == target_lang:
        return Response(json.dumps(response_text, ensure_ascii=False))

    translated_texts: str = ""

    # Check translation confidence and perform translation if confidence is high enough
    if confidence > det_conf_float:
        try:
            # Perform translation using the translation model
            translation = translator(clean_text, src_lang=target_languages[source_lang], tgt_lang=target_lang)
            translated_texts = translation[0]['translation_text']
            response_text["is_trans"] = True
            response_text["target_msg"] = translated_texts
            response_text["model"] = model_name
            logger.info(f'Successful translation: {translated_texts}')  # Log the successful translation
            return Response(json.dumps(response_text, ensure_ascii=False))

        except Exception as e:
            error_msg = f"翻譯過程中出錯：{e}"
            logger.error(error_msg)  # Log the error
            return jsonify({"error": error_msg}), 500
        
    else:
        logger.warning("Confidence level too low for translation.")  # Log warning
        return Response(json.dumps(response_text, ensure_ascii=False)) # 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)