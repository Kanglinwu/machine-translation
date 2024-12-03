import sys
import csv
import json
from pathlib import Path
from typing import Tuple

import yaml
import torch
import fasttext
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from utils.text import normalize_text


def read_csv_to_dict(file_path: Path):
    try:
        with file_path.open(encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)
            result_dict = {}
            for row in reader:
                if len(row) >= 2:
                    key = row[0]
                    value = row[1]
                    result_dict[key] = value
                else:
                    print(f"Skipping row due to insufficient columns: {row}")
            return result_dict
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


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


project_root = Path(__file__).parent.parent.resolve()

with (project_root / "conf/api.yml").open("r", encoding="utf-8") as file:
    cfg = yaml.safe_load(file)

models_dir = Path(project_root / cfg["models_dir"])
model_lid_name = cfg["model_lid_name"]
model_mt_name = cfg["model_mt_name"]
default_languages = cfg["default_languages"]

iso_639_to_flores_200_file = Path(project_root / cfg["iso_639_to_flores_200_file"])
with Path(iso_639_to_flores_200_file).open() as file:
    iso_639_to_flores_200 = json.load(file)

flores_200_to_iso_639 = {value: key for key, value in iso_639_to_flores_200.items()}

flores_200_codes = set()
for iso_639_code in iso_639_to_flores_200:
    if iso_639_to_flores_200[iso_639_code] not in flores_200_codes:
        flores_200_codes.add(iso_639_to_flores_200[iso_639_code])
    else:
        print(f"Duplicate code: {iso_639_code}, {iso_639_to_flores_200[iso_639_code]}")

lang_to_flores_200_file = Path(project_root / cfg["lang_to_flores_200_file"])
lang_to_flores_200 = read_csv_to_dict(lang_to_flores_200_file)
flores_200_to_lang = {value: key for key, value in lang_to_flores_200.items()}

flores_200_to_lang_file = Path(project_root / cfg["flores_200_to_lang_file"])
with Path(flores_200_to_lang_file).open("w") as file:
    json.dump(flores_200_to_lang, file, indent=4, ensure_ascii=False)

exit()


DEFAULT_LANGUAGES = ["en", "ko", "th", "vi", "zh"]

if sys.platform.startswith("darwin"):
    DEVICE = torch.device("mps" if torch.cuda.is_available() else "cpu")
elif sys.platform.startswith("linux"):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cpu")


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


if __name__ == "__main__":
    raw_text = "Cô gái này đẹp quá!"
    # raw_text = "Cậu bé này đẹp trai quá!"

    # target_language = "en"
    # target_language = CODE_MAP[target_language]

    # normalized_text = normalize_text(raw_text)

    # predicted_source_language, confidence_score = identify_language(normalized_text)

    # translated = translate(normalized_text, predicted_source_language, target_language)
    # print(type(translated))

    # print(normalized_text)

    # print(predicted_source_language, confidence_score)

    # print(translated)
