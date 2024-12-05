import sys
import json
import uuid
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
from logging.handlers import RotatingFileHandler

import yaml
import torch
import fasttext
from gevent import pywsgi
from flask_cors import cross_origin
from huggingface_hub import hf_hub_download
from flask import Flask, request, jsonify, Response
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from utils.text import split_string_by_emoji, normalize_text


app = Flask(__name__)


class ModelConfig:
    """
    統一管理模型配置和初始化的類
    """

    def __init__(self, config_path: str = "conf/api.yml"):
        """
        初始化模型配置

        Args:
            config_path (str): 配置文件路徑
        """
        self.project_root = Path(__file__).parent.resolve()
        self.cfg = self._load_config(config_path)
        self._validate_config()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        載入配置文件

        Args:
            config_path (str): 配置文件路徑

        Returns:
            Dict[str, Any]: 配置字典
        """
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logging.error(f"配置文件未找到: {config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"配置文件解析錯誤: {e}")
            raise

    def _validate_config(self):
        """
        驗證配置文件的必要欄位
        """
        required_keys = [
            "models_dir",
            "model_lid_name",
            "model_mt_name",
            "log_file",
            "lang_to_flores_200_file",
            "flores_200_to_lang_file",
            "default_languages",
        ]
        for key in required_keys:
            if key not in self.cfg:
                raise KeyError(f"配置缺少必要的鍵: {key}")

    def setup_logging(self) -> logging.Logger:
        """
        設置日誌系統

        Returns:
            logging.Logger: 配置好的日誌記錄器
        """
        log_file = self.project_root / self.cfg["log_file"]
        log_dir = log_file.parent
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # 迴轉日誌處理器
        handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=5,
        )

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def get_device(self) -> torch.device:
        """
        根據當前平台和可用硬體選擇運算設備

        Returns:
            torch.device: 選定的運算設備
        """
        if sys.platform.startswith("darwin"):
            return torch.device("mps" if torch.cuda.is_available() else "cpu")
        elif sys.platform.startswith("linux"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device("cpu")

    def load_translation_models(self):
        """
        載入語言識別和機器翻譯模型

        Returns:
            Tuple: (語言識別模型, 翻譯模型, 翻譯tokenizer)
        """
        models_dir = self.project_root / self.cfg["models_dir"]

        # 語言識別模型
        model_lid_name = self.cfg["model_lid_name"]
        model_lid_file = hf_hub_download(
            model_lid_name, "model.bin", cache_dir=models_dir
        )
        model_lid = fasttext.load_model(str(model_lid_file))

        # 翻譯模型
        model_mt_name = self.cfg["model_mt_name"]

        model_mt = AutoModelForSeq2SeqLM.from_pretrained(
            model_mt_name, cache_dir=models_dir, torch_dtype=torch.bfloat16
        )

        tokenizer_mt = AutoTokenizer.from_pretrained(
            model_mt_name, cache_dir=models_dir, torch_dtype=torch.bfloat16
        )

        device = self.get_device()

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

        return model_lid, translator

    def load_language_mappings(self):
        """
        載入語言對映表

        Returns:
            Tuple: (lang_to_flores_200, flores_200_to_lang)
        """
        lang_to_flores_200_file = (
            self.project_root / self.cfg["lang_to_flores_200_file"]
        )
        with Path(lang_to_flores_200_file).open() as file:
            lang_to_flores_200 = json.load(file)

        flores_200_to_lang_file = (
            self.project_root / self.cfg["flores_200_to_lang_file"]
        )
        with Path(flores_200_to_lang_file).open() as file:
            flores_200_to_lang = json.load(file)

        return lang_to_flores_200, flores_200_to_lang


def initialize_app(app):
    """
    初始化 Flask 應用程式的全域變數和模型

    Args:
        app (Flask): Flask 應用程式實例
    """
    # 創建模型配置實例
    model_config = ModelConfig()

    # 設置日誌
    global logger
    logger = model_config.setup_logging()

    # 載入模型
    global model_lid, translator
    model_lid, translator = model_config.load_translation_models()

    # 載入語言對映
    global lang_to_flores_200, flores_200_to_lang
    lang_to_flores_200, flores_200_to_lang = model_config.load_language_mappings()

    # 設定預設語言列表
    global default_languages
    default_languages = model_config.cfg.get("default_languages", [])

    # 生成模型名稱
    global model_lid_name, model_mt_name
    model_lid_name = model_config.cfg["model_lid_name"]
    model_mt_name = model_config.cfg["model_mt_name"]


def identify_language(text: str, threshold: float = 0.0) -> Tuple[str, str, float]:
    """
    識別輸入文本的語言

    Args:
        text (str): 待識別語言的文本
        threshold (float, optional): 語言confidence的最小閾值. Defaults to 0.7.

    Returns:
        Tuple[str, str, float]: 識別出的語言代碼、語言和confidence分數
    """
    # 預測前 5 種可能的語言
    predicted_labels, confidence_scores = model_lid.predict(text, k=5)

    # 移除 __label__ 前綴
    predicted_codes = [label.replace("__label__", "") for label in predicted_labels]
    predicted_languages = [
        flores_200_to_lang.get(code, "N/A") for code in predicted_codes
    ]

    # 日誌記錄預測結果（可選）
    logger.debug(f"Predicted codes: {predicted_codes}")
    logger.debug(f"Predicted languages: {predicted_languages}")
    logger.debug(f"Confidence scores: {confidence_scores}")

    # 優先處理預設語言且confidence夠高的情況
    for i, lang in enumerate(predicted_languages):
        if lang in default_languages and confidence_scores[i] >= threshold:
            return predicted_codes[i], lang, confidence_scores[i]
    else:
        # 如果沒有預設語言滿足閾值，返回第一個預測結果
        predicted_code = predicted_codes[0]
        predicted_language = predicted_languages[0]
        confidence_score = confidence_scores[0]

    # 額外的安全檢查：如果confidence太低，可以拋出警告或返回特殊值
    if confidence_score < threshold:
        logger.warning(
            f"Low confidence language detection: {predicted_language} "
            f"(score: {confidence_score})"
        )

    return predicted_code, predicted_language, confidence_score


def validate_data(data: Dict) -> Tuple[str, str]:
    """
    驗證輸入數據是否符合預期

    Args:
        data (Dict): 輸入數據

    Returns:
        bool: 是否通過驗證
    """
    if "raw_text" not in data:
        raise Exception("raw_text is missing")

    if "target_language" not in data:
        raise Exception("target_language is missing")

    raw_text = data["raw_text"].strip()
    target_language = data["target_language"].strip()

    if not raw_text:
        raise Exception("raw_text is empty")

    if not target_language:
        raise Exception("target_language is empty")

    if target_language not in default_languages:
        raise Exception("target_language does not belong to default_languages")

    return raw_text, target_language


@app.route("/translate", methods=["POST"])
@cross_origin()
def translate():
    """
    機器翻譯 API 端點

    處理翻譯請求，包括輸入驗證、語言識別和翻譯

    Returns:
        JSON 響應，包含翻譯結果或錯誤信息
    """

    # 驗證輸入
    try:
        data = request.get_json()
        raw_text, target_language = validate_data(data)
        target_code = lang_to_flores_200[target_language]
    except Exception as e:
        logger.error(f"Input validation error: {e}")
        return jsonify({"error": "Invalid input", "details": f"{e}"}), 400

    # 日誌追蹤
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Raw text: {raw_text}")
    logger.info(f"[{request_id}] Target language: {target_language}")

    response = {
        "request_id": request_id,
        "raw_text": raw_text,
        "translated_text": "",
        "predicted_language": "",
        "target_language": target_language,
        "model_lid": model_lid_name,
        "model_mt": model_mt_name,
    }

    try:
        # 分割輸入字串
        split_strings, is_emoji = split_string_by_emoji(raw_text)

        translated_text = []
        predicted_languages = set()
        confidence_scores = []
        for i, split_string in enumerate(split_strings):
            normalized_string = normalize_text(split_string)
            if not normalized_string:
                continue

            if not is_emoji[i]:
                try:
                    predicted_code, predicted_language, confidence_score = (
                        identify_language(
                            normalized_string,
                            threshold=0.0,  # 可調整的信心閾值
                        )
                    )
                    predicted_languages.add(predicted_language)
                    confidence_scores.append(confidence_score)
                except Exception as e:
                    logger.error(f"[{request_id}] LID Failed: {e}")
                    return jsonify({"error": "LID Failed", "details": str(e)}), 500

                try:
                    translated = translator(
                        normalized_string,
                        src_lang=predicted_code,
                        tgt_lang=target_code,
                    )
                    translated_text.append(translated[0]["translation_text"])  # type: ignore
                except Exception as e:
                    logger.error(f"[{request_id}] MT Failed: {e}")
                    return jsonify({"error": "MT Failed", "details": str(e)}), 500
            else:
                translated_text.append(split_string)

        response["predicted_languages"] = list(predicted_languages)
        response["translated_text"] = " ".join(translated_text)

        logger.debug(f"[{request_id}] Predicted Languages: {list(predicted_languages)}")
        logger.debug(f"[{request_id}] Confidence Scores: {confidence_scores}")
        logger.debug(f"[{request_id}] Translated Text: {translated_text}")

        return Response(
            json.dumps(response, ensure_ascii=False),
            content_type="application/json; charset=utf-8",
        )

    except Exception as e:
        logger.error(f"[{request_id}] Unexpected Error: {e}")
        return jsonify({"error": "Unexpected Error", "details": str(e)}), 500


if __name__ == "__main__":
    initialize_app(app)

    server = pywsgi.WSGIServer(("0.0.0.0", 5050), app)
    server.serve_forever()
