import unicodedata


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
