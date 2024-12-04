import sys
import json
from pathlib import Path
from typing import Tuple

import yaml
import torch
import fasttext
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from utils.text import normalize_text


def translate(text: str, source_language: str = "en", target_language: str = "en"):
    translated = translator(
        text,
        src_lang=source_language,
        tgt_lang=target_language,
    )
    return translated


project_root = Path(__file__).parent.parent.resolve()

with (project_root / "conf/api.yml").open("r", encoding="utf-8") as file:
    cfg = yaml.safe_load(file)

models_dir = Path(project_root / cfg["models_dir"])
model_lid_name = cfg["model_lid_name"]
model_mt_name = cfg["model_mt_name"]
default_languages = cfg["default_languages"]

lang_to_flores_200_file = Path(project_root / cfg["lang_to_flores_200_file"])
with Path(lang_to_flores_200_file).open() as file:
    lang_to_flores_200 = json.load(file)

flores_200_to_lang_file = Path(project_root / cfg["flores_200_to_lang_file"])
with Path(flores_200_to_lang_file).open() as file:
    flores_200_to_lang = json.load(file)

if sys.platform.startswith("darwin"):
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
elif sys.platform.startswith("linux"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# 語言識別模型
model_lid_name = cfg["model_lid_name"]
model_lid_file = hf_hub_download(model_lid_name, "model.bin", cache_dir=models_dir)
model_lid = fasttext.load_model(str(model_lid_file))

# 翻譯模型
model_mt_name = cfg["model_mt_name"]
model_mt = AutoModelForSeq2SeqLM.from_pretrained(
    model_mt_name, cache_dir=models_dir, torch_dtype=torch.bfloat16
)
tokenizer_mt = AutoTokenizer.from_pretrained(
    model_mt_name, cache_dir=models_dir, torch_dtype=torch.bfloat16
)

translator = pipeline(
    "translation",
    model=model_mt,
    tokenizer=tokenizer_mt,
    max_length=512,
    device=device,
)

# 設置 CUDA 優化
torch.backends.cuda.enable_flash_sdp = True
torch.backends.cuda.enable_mem_efficient_sdp = True
torch.backends.cuda.enable_math_sdp = False

if __name__ == "__main__":
    threshold = 0.0

    raw_text = "Cô gái này đẹp quá!"
    # raw_text = "Cậu bé này đẹp trai quá!"

    target_language = "English"
    target_code = lang_to_flores_200[target_language]

    normalized_text = normalize_text(raw_text)

    print(f"Input: {raw_text}")
    print(f"Normalized: {normalized_text}")

    predicted_labels, confidence_scores = model_lid.predict(normalized_text, k=5)

    predicted_codes = [label.replace("__label__", "") for label in predicted_labels]
    predicted_languages = [flores_200_to_lang[code] for code in predicted_codes]

    for i, lang in enumerate(predicted_languages):
        if lang in default_languages and confidence_scores[i] >= threshold:
            predicted_code = predicted_codes[i]
            predicted_language = lang
            confidence_score = confidence_scores[i]
            break
    else:
        predicted_code = predicted_codes[0]
        predicted_language = predicted_languages[0]
        confidence_score = confidence_scores[0]

    print(f"Predicted: {predicted_language} ({predicted_code})")
    print(f"Confidence: {confidence_score}")

    translated_text = translate(normalized_text, predicted_code, target_code)
    # print(type(translated))

    # print(normalized_text)

    # print(predicted_source_language, confidence_score)

    print(translated_text)
