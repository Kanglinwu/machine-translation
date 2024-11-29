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
from gevent import pywsgi
import logging
from logging.handlers import RotatingFileHandler
from flask_cors import cross_origin
import fasttext
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

import torch
import json
from flask import Flask, request, jsonify, Response
import os

from utils.text import normalize_string
from utils.lang_dict import langCode_convert
from utils.merge_words import interleave_lists

DEFAULT_LANGUAGES = ["en", "ko", "th", "vi", "zh"]


# Configure Logging
# ------------------------------------------------------------------------------------
log_file_path = "logs/api.log"  # Adjust the path based on your Docker volume setup
os.makedirs(
    os.path.dirname(log_file_path), exist_ok=True
)  # Create the logs directory if it doesn't exist

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# ------------------------------------------------------------------------------------

# Create a rotating file handler
# ------------------------------------------------------------------------------------
handler = RotatingFileHandler(
    log_file_path, maxBytes=5 * 1024 * 1024, backupCount=5
)  # 5 MB per log file, keep 5 backups
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)
# -----------------------------------------------------------------------------------

app = Flask(__name__)

# Load FastText language detection model
model_ft = fasttext.load_model("models/lid.176.bin")

# Load translation model and tokenizer
model_name = "facebook/nllb-200-distilled-600M"
cache_dir = "./models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO
# Post Training Quantization
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------

model_mt = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16
)
tokenizer_mt = AutoTokenizer.from_pretrained(
    model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16
)

torch.backends.cuda.enable_flash_sdp = True  # This enables Flash Attention (SDPA)
torch.backends.cuda.enable_mem_efficient_sdp = (
    True  # Use memory efficient SDPA (if supported)
)
torch.backends.cuda.enable_math_sdp = (
    False  # Use this flag to revert to math-based SDPA if needed (less efficient)
)

# Define translation pipeline
translator = pipeline(
    "translation",
    model=model_mt,
    tokenizer=tokenizer_mt,
    max_length=512,
    device=device,
)


@app.route("/translate", methods=["POST"])
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
        "source_lang": "Detected Source Language",
        "is_trans": True,
        "target_msg": "Translated text here."
        "all_emoji": "If the message is full of emojis.,
        "model": "Model used."
        "lang_undefined": "If the language is the limited language."
    }
    """
    det_conf = os.getenv("det_conf")

    # Check if `det_conf` is valid or not
    # ------------------------------------------------------------------------------------
    if not det_conf:
        error_msg = "det_conf is not set."
        logger.error(error_msg)  # Log the error
        return jsonify({"error": error_msg}), 400  # Return a 400 Bad Request response

    try:
        det_conf_float = float(det_conf)  # Attempt to convert to float
    except ValueError:
        error_msg = (
            f"Invalid det_conf value: {det_conf}. Must be a float between 0 and 1."
        )
        logger.error(error_msg)  # Log the error
        return jsonify({"error": error_msg}), 400  # Return a 400 Bad Request response

    # Check if det_conf is within the valid range
    if not (0 <= det_conf_float <= 1):
        error_msg = "Valid det_conf required (must be a float between 0 and 1)."
        logger.error(error_msg)  # Log the error
        return jsonify({"error": error_msg}), 400  # Return a 400 Bad Request response

    # ------------------------------------------------------------------------------------

    data = request.get_json()  # 從 JSON 請求中提取數據
    logger.info(f"Received request data: {data}")

    msg = data.get("msg").strip()  # Text to be translated

    # Check if input text is provided
    if not msg:
        error_msg = "Invalid input text."
        logger.error(error_msg)  # Log the error
        return jsonify({"error": error_msg}), 500

    # Initialize response object with default values for source language, translation flag, and translated text
    response_text = {
        "source_lang": "",
        "is_trans": False,
        "target_msg": msg,
        "all_emoji": False,
        "model": "None",
        "lang_undefined": False,
    }

    msgs = normalize_string(msg)
    if not msg:
        error_msg = "Invalid input text."
        logger.error(error_msg)  # Log the error
        return jsonify({"error": error_msg}), 500

    if len(msgs) == 0:
        logger.info("All emojis, meaningless data.")
        response_text["all_emoji"] = True
        return jsonify(response_text)

    logger.info(f"Normalized text: {msgs}")

    # Check target language
    target_lang = data.get("target_lang")  # Language to be translated

    if not target_lang:
        error_msg = "No target language provided."
        logger.error(error_msg)  # Log the error
        return jsonify({"error": error_msg}), 500

    if target_lang not in langCode_convert.keys():
        error_msg = "Invalid taraget language."
        logger.error(error_msg)  # Log the error
        return jsonify({"error": error_msg}), 500

    target_lang = langCode_convert[target_lang]

    # Clean input text
    clean_text = msgs

    # Language Detection
    predicted_source_languages, confidence_scores = model_ft.predict(clean_text, k=5)
    predicted_source_languages = [
        lang.replace("__label__", "") for lang in predicted_source_languages
    ]

    for i, lang in enumerate(predicted_source_languages):
        if lang in DEFAULT_LANGUAGES:
            confidence_score = confidence_scores[i]
            predicted_source_language = lang
            break
    else:
        predicted_source_language = predicted_source_languages[0]
        confidence_score = confidence_scores[0]

    response_text["source_lang"] = predicted_source_language
    logger.info(
        f"Detected language: {predicted_source_language}, Confidence: {confidence_score}"
    )  # Log detected language and confidence

    # if source_lang not in language_options:
    #     logger.info(f"Unidentified language.")
    #     response_text["lang_unidentified"] = True
    #     return Response(json.dumps(response_text, ensure_ascii=False))

    # Check if source language detected is the same as the target language
    if langCode_convert[predicted_source_language] == target_lang:
        logger.info("The source language is same as the target language.")
        return Response(json.dumps(response_text, ensure_ascii=False))

    translated_texts: str = ""

    # Check translation confidence and perform translation if confidence is high enough
    if confidence_score > det_conf_float:
        try:
            # Perform translation using the translation model
            translation = translator(
                clean_text,
                src_lang=langCode_convert[predicted_source_language],
                tgt_lang=target_lang,
            )

            translated_texts = [
                translated["translation_text"] for translated in translation
            ]
            translated_texts = (
                interleave_lists(emojis, translated_texts)
                if firstIndex
                else interleave_lists(translated_texts, emojis)
            )

            response_text["is_trans"] = True
            response_text["target_msg"] = translated_texts
            response_text["model"] = model_name
            logger.info(
                f"Successful translation: {translated_texts}"
            )  # Log the successful translation
            return Response(json.dumps(response_text, ensure_ascii=False))

        except Exception as e:
            error_msg = f"翻譯過程中出錯：{e}"
            logger.error(error_msg)  # Log the error
            return jsonify({"error": error_msg}), 500

    else:
        logger.warning("Confidence level too low for translation.")  # Log warning
        return Response(json.dumps(response_text, ensure_ascii=False))  #


if __name__ == "__main__":
    server = pywsgi.WSGIServer(("0.0.0.0", 5050), app)
    server.serve_forever()
