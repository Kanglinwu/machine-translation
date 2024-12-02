import os
import sys
import json
import logging
from typing import Tuple
from pathlib import Path
from logging.handlers import RotatingFileHandler

import yaml
import torch
import fasttext
from gevent import pywsgi
from flask_cors import cross_origin
from flask import Flask, request, jsonify, Response
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from utils.text import normalize_text
from utils.merge_words import interleave_lists


with open("conf/api.yml", "r", encoding="utf-8") as file:
    cfg = yaml.safe_load(file)

MODELS_DIR = Path(cfg["models_dir"])
MODEL_LID_NAME = cfg["model_lid_name"]
MODEL_MT_NAME = cfg["model_mt_name"]
DEFAULT_LANGUAGES = cfg["default_languages"]

CODE_MAP_FILE = Path(cfg["code_map_file"])
with Path(CODE_MAP_FILE).open() as file:
    CODE_MAP = json.load(file)

LOG_FILE = Path(cfg["log_file"])

if sys.platform.startswith("darwin"):
    DEVICE = torch.device("mps" if torch.cuda.is_available() else "cpu")
elif sys.platform.startswith("linux"):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cpu")

# Configure Logging
# ------------------------------------------------------------------------------------
(LOG_FILE.parent).mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# ------------------------------------------------------------------------------------

# Create a rotating file handler
# ------------------------------------------------------------------------------------
handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
# -----------------------------------------------------------------------------------

app = Flask(__name__)


model_lid = fasttext.load_model(str(MODELS_DIR / MODEL_LID_NAME))
model_mt = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_MT_NAME, cache_dir=MODELS_DIR, torch_dtype=torch.bfloat16
)
tokenizer_mt = AutoTokenizer.from_pretrained(
    MODEL_MT_NAME, cache_dir=MODELS_DIR, torch_dtype=torch.bfloat16
)
translator = pipeline(
    "translation",
    model=model_mt,
    tokenizer=tokenizer_mt,
    max_length=512,
    device=DEVICE,
)

torch.backends.cuda.enable_flash_sdp = True
torch.backends.cuda.enable_mem_efficient_sdp = True
torch.backends.cuda.enable_math_sdp = False


def identify_language(text: str) -> Tuple[str, float]:
    predicted_source_languages, confidence_scores = model_lid.predict(text, k=5)
    print(predicted_source_languages, confidence_scores)
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

    return predicted_source_language, confidence_score


@app.route("/translate", methods=["POST"])
@cross_origin()
def translate():
    data = request.get_json()
    logger.info(f"Received request data: {data}")

    raw_msg = data.get("msg").strip()
    if not raw_msg:
        error_msg = "Invalid input text."
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

    response_text = {
        "source_lang": "",
        "is_trans": False,
        "target_msg": raw_msg,
        "all_emoji": False,
        "model": f"{MODEL_LID_NAME}+{MODEL_MT_NAME}",
        "lang_undefined": False,
    }

    normalized_msgs = normalize_text(raw_msg)
    if not normalized_msgs:
        error_msg = "Invalid input text."
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

    logger.info(f"Normalized text: {normalized_msgs}")

    target_lang = data.get("target_lang").strip()

    if not target_lang:
        error_msg = "No target language provided."
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

    if target_lang not in CODE_MAP.values():
        error_msg = "Invalid taraget language."
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

    # target_lang = CODE_MAP[target_lang]

    predicted_source_language, confidence_score = identify_language(normalized_msgs)

    response_text["source_lang"] = predicted_source_language
    logger.info(
        f"Detected language: {predicted_source_language}, Confidence: {confidence_score}"
    )

    translated_texts: str = ""

    try:
        translation = translator(
            normalized_msgs,
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


if __name__ == "__main__":
    server = pywsgi.WSGIServer(("0.0.0.0", 5050), app)
    server.serve_forever()
