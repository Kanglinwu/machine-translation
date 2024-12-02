import sys
import json
from pathlib import Path
from typing import Tuple

import yaml
import torch
import fasttext
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from utils.text import normalize_text


PROJECT_ROOT = Path(__file__).parent.parent.resolve()

with (PROJECT_ROOT / "conf/api.yml").open("r", encoding="utf-8") as file:
    cfg = yaml.safe_load(file)

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_LID_NAME = cfg["model_lid_name"]
MODEL_MT_NAME = "facebook/nllb-200-distilled-600M"
CODE_MAP_FILE = PROJECT_ROOT / "docs/iso-639_to_flores-200.json"
DEFAULT_LANGUAGES = ["en", "ko", "th", "vi", "zh"]

if sys.platform.startswith("darwin"):
    DEVICE = torch.device("mps" if torch.cuda.is_available() else "cpu")
elif sys.platform.startswith("linux"):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cpu")

with Path(CODE_MAP_FILE).open() as file:
    CODE_MAP = json.load(file)

assert len(CODE_MAP.values()) == len(set(CODE_MAP.values()))
exit()

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


def translate(text: str, source_language: str = "en", target_language: str = "en"):
    translated = translator(
        text,
        src_lang=source_language,
        tgt_lang=target_language,
    )
    return translated


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


if __name__ == "__main__":
    raw_text = "Cô gái này đẹp quá!"
    # raw_text = "Cậu bé này đẹp trai quá!"

    target_language = "en"
    target_language = CODE_MAP[target_language]

    normalized_text = normalize_text(raw_text)

    predicted_source_language, confidence_score = identify_language(normalized_text)

    translated = translate(normalized_text, predicted_source_language, target_language)
    print(type(translated))

    print(normalized_text)

    print(predicted_source_language, confidence_score)

    print(translated)
