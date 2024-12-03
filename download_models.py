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
from flask import Flask, request, jsonify, Response
from marshmallow import Schema, fields, ValidationError, validate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from utils.text import normalize_text


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
            "iso_639_to_flores_200_file",
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
        model_lid_file = models_dir / model_lid_name
        model_lid = fasttext.load_model(str(model_lid_file))

        # 翻譯模型
        model_mt_name = self.cfg["model_mt_name"]
        device = self.get_device()

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

        return model_lid, translator

    def load_language_mappings(self):
        """
        載入語言對映表

        Returns:
            Tuple: (iso_639_to_flores_200, flores_200_to_iso_639)
        """
        iso_639_to_flores_200_file = (
            self.project_root / self.cfg["iso_639_to_flores_200_file"]
        )

        with Path(iso_639_to_flores_200_file).open() as file:
            iso_639_to_flores_200 = json.load(file)

        flores_200_to_iso_639 = {
            value: key for key, value in iso_639_to_flores_200.items()
        }

        flores_200_to_lang_file = (
            self.project_root / self.cfg["flores_200_to_lang_file"]
        )
        with Path(flores_200_to_lang_file).open() as file:
            flores_200_to_lang = json.load(file)

        return iso_639_to_flores_200, flores_200_to_iso_639, flores_200_to_lang


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
    global iso_639_to_flores_200, flores_200_to_iso_639, flores_200_to_lang
    iso_639_to_flores_200, flores_200_to_iso_639, flores_200_to_lang = (
        model_config.load_language_mappings()
    )

    # 設定預設語言列表
    global default_languages
    default_languages = model_config.cfg.get("default_languages", [])

    # 生成模型名稱
    global model_name
    model_name = (
        f"{model_config.cfg['model_lid_name']}+{model_config.cfg['model_mt_name']}"
    )


class TranslationRequestSchema(Schema):
    """
    定義翻譯請求的校驗模式
    """

    @staticmethod
    def validate_not_empty_after_strip(value):
        """
        驗證字串在 strip() 後不為空

        Args:
            value (str): 待驗證的字串

        Raises:
            ValidationError: 如果字串在 strip() 後為空
        """
        if not value or not str(value).strip():
            raise ValidationError(
                "Target language cannot be an empty string after stripping whitespace."
            )

    msg = fields.String(
        required=True,
        validate=[
            validate.Length(
                min=1,
                max=280,
                error="Text length must be between 1 and 280 characters",
            )
        ],
    )
    target_lang = fields.String(
        required=True, validate=[validate_not_empty_after_strip]
    )


def identify_language(text: str, threshold: float = 0.0) -> Tuple[str, float]:
    """
    識別輸入文本的語言

    Args:
        text (str): 待識別語言的文本
        threshold (float, optional): 語言confidence的最小閾值. Defaults to 0.7.

    Returns:
        Tuple[str, float]: 識別出的語言代碼和confidence分數
    """
    # 檢查輸入文本是否為空
    if not text or len(text.strip()) == 0:
        raise ValueError("Input text cannot be empty")

    # 預測前 5 種可能的語言
    predicted_source_languages, confidence_scores = model_lid.predict(text, k=5)

    # 移除 __label__ 前綴
    predicted_source_languages = [
        lang.replace("__label__", "") for lang in predicted_source_languages
    ]

    # 日誌記錄預測結果（可選）
    logger.debug(f"Predicted language: {predicted_source_languages}")
    logger.debug(f"Confidence scores: {confidence_scores}")

    # 優先處理預設語言且confidence夠高的情況
    for i, lang in enumerate(predicted_source_languages):
        if lang in default_languages and confidence_scores[i] >= threshold:
            return lang, confidence_scores[i]

    # 如果沒有預設語言滿足閾值，返回第一個預測結果
    predicted_source_language = predicted_source_languages[0]
    confidence_score = confidence_scores[0]

    # 額外的安全檢查：如果confidence太低，可以拋出警告或返回特殊值
    if confidence_score < threshold:
        logger.warning(
            f"Low confidence language detection: {predicted_source_language} "
            f"(score: {confidence_score})"
        )

    return predicted_source_language, confidence_score


@app.route("/translate", methods=["POST"])
@cross_origin()
def translate():
    """
    機器翻譯 API 端點

    處理翻譯請求，包括輸入驗證、語言識別和翻譯

    Returns:
        JSON 響應，包含翻譯結果或錯誤信息
    """
    # 使用模式驗證輸入
    schema = TranslationRequestSchema()
    try:
        data = schema.load(request.get_json())
    except ValidationError as err:
        logger.error(f"Input validation error: {err.messages}")
        return jsonify({"error": "Invalid input", "details": err.messages}), 400

    # 提取並規範化輸入
    raw_msg = data["msg"]  # type: ignore
    target_lang = data["target_lang"]  # type: ignore

    # 日誌追蹤
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Received translation request")
    logger.info(f"[{request_id}] Target language: {target_lang}")

    try:
        # 文本規範化
        normalized_msgs = normalize_text(raw_msg)
        if not normalized_msgs:
            raise ValueError("Normalized text is empty")

        logger.info(f"[{request_id}] Normalized text: {normalized_msgs}")

        # 語言偵測
        try:
            predicted_source_language, confidence_score = identify_language(
                normalized_msgs,
                threshold=0.0,  # 可調整的信心閾值
            )
            predicted_source_language = flores_200_to_iso_639[predicted_source_language]
        except ValueError as lang_error:
            logger.warning(f"[{request_id}] Language detection failed: {lang_error}")
            predicted_source_language = "en"  # 預設語言
            confidence_score = 0.0

        # 構建響應基礎結構
        response_text = {
            "request_id": request_id,
            "source_msg": raw_msg,
            "source_lang": predicted_source_language,
            "target_lang": target_lang,
            "confidence": confidence_score,
            "is_trans": True,
            "model": model_name,
        }

        # 執行翻譯
        try:
            translated = translator(
                normalized_msgs,
                src_lang=predicted_source_language,
                tgt_lang=iso_639_to_flores_200[target_lang],
            )
            translated = translated[0]["translation_text"]  # type: ignore
            print(translated)

            response_text["target_msg"] = translated

            logger.info(f"[{request_id}] Translation successful")
            logger.debug(f"[{request_id}] Translated text: {translated}")

            return Response(
                json.dumps(response_text, ensure_ascii=False),
                content_type="application/json; charset=utf-8",
            )

        except Exception as translation_error:
            logger.error(f"[{request_id}] Translation error: {translation_error}")
            return jsonify(
                {"error": "Translation failed", "details": str(translation_error)}
            ), 500

    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {e}")
        return jsonify(
            {"error": "An unexpected error occurred", "details": str(e)}
        ), 500


if __name__ == "__main__":
    initialize_app(app)

    server = pywsgi.WSGIServer(("0.0.0.0", 5050), app)
    server.serve_forever()
