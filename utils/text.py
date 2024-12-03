import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional


def normalize_text(text):
    """
    文字正規化函式，具有以下功能：
    1. 移除 emoji
    2. 保留中文、英文、泰文、越南文、韓文字符
    3. 去除與語言無關的特殊符號
    4. 轉換為小寫

    Args:
        text (str): 輸入的原始文字

    Returns:
        str: 正規化後的文字
    """
    # 定義要保留的 Unicode 區塊
    unicode_ranges = {
        "CJK": (0x4E00, 0x9FFF),  # 中文漢字
        "Latin": (0x0000, 0x007F),  # 基本拉丁字母（英文）
        "Thai": (0x0E00, 0x0E7F),  # 泰文
        "Vietnamese": (0x00C0, 0x1EF9),  # 越南文音調字母
        "Korean": (0xAC00, 0xD7A3),  # 韓文音節
    }

    def is_allowed_char(char):
        """檢查字元是否在允許的 Unicode 區塊中"""
        code = ord(char)
        return any(start <= code <= end for start, end in unicode_ranges.values())

    # 去除 emoji 和特殊符號，保留允許的字元
    normalized_text = "".join(
        char if is_allowed_char(char) or char.isspace() else " "
        for char in unicodedata.normalize("NFKD", text)
    )

    # 去除多餘空白並轉為小寫
    return " ".join(normalized_text.lower().split())


def read_csv_to_dict(
    file_path: Path,
    key_column: int = 1,
    value_column: int = 0,
    skip_header: bool = True,
) -> Optional[Dict[str, str]]:
    """
    從 CSV 檔案讀取數據並轉換為字典

    Args:
        file_path (Path): CSV 檔案路徑
        key_column (int, optional): 作為字典鍵的列索引. Defaults to 1.
        value_column (int, optional): 作為字典值的列索引. Defaults to 0.
        skip_header (bool, optional): 是否跳過第一行. Defaults to True.

    Returns:
        Optional[Dict[str, str]]: 轉換後的字典，讀取失敗時返回 None
    """
    try:
        # 檢查文件是否存在且可讀
        if not file_path.is_file():
            logger.error(f"File not found: {file_path}")
            return None

        result_dict = {}

        with file_path.open(encoding="utf-8") as file:
            reader = csv.reader(file)

            # 根據 skip_header 參數決定是否跳過標題列
            if skip_header:
                next(reader, None)

            for row_num, row in enumerate(reader, 1):
                # 檢查行是否有足夠的列
                if len(row) < max(key_column, value_column) + 1:
                    logger.warning(
                        f"Skipping row {row_num} due to insufficient columns: {row}"
                    )
                    continue

                # 處理可能的空值或異常值
                key = str(row[key_column]).strip()
                value = str(row[value_column]).strip()

                if not key:
                    logger.warning(f"Skipping row {row_num} due to empty key")
                    continue

                # 檢查是否有重複鍵
                if key in result_dict:
                    logger.warning(
                        f"Duplicate key '{key}' found. Overwriting previous value."
                    )

                result_dict[key] = value

        return result_dict

    except PermissionError:
        logger.error(f"Permission denied when accessing file: {file_path}")
        return None
    except csv.Error as e:
        logger.error(f"CSV parsing error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error reading CSV: {e}")
        return None
