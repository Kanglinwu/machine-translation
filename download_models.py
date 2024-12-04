import logging
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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
        ]
        for key in required_keys:
            if key not in self.cfg:
                raise KeyError(f"配置缺少必要的鍵: {key}")

    def load_translation_models(self):
        """
        載入語言識別和機器翻譯模型

        Returns:
            Tuple: (語言識別模型, 翻譯模型, 翻譯tokenizer)
        """
        models_dir = self.project_root / self.cfg["models_dir"]

        # 語言識別模型
        model_lid_name = self.cfg["model_lid_name"]
        hf_hub_download(model_lid_name, "model.bin", cache_dir=models_dir)

        # # 翻譯模型
        model_mt_name = self.cfg["model_mt_name"]

        AutoModelForSeq2SeqLM.from_pretrained(
            model_mt_name, cache_dir=models_dir, torch_dtype=torch.bfloat16
        )

        AutoTokenizer.from_pretrained(
            model_mt_name, cache_dir=models_dir, torch_dtype=torch.bfloat16
        )


if __name__ == "__main__":
    model_config = ModelConfig()
    model_config.load_translation_models()
