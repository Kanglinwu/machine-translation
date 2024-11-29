import re
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


def remove_punctuations_symbols_emojis(text):
    # Emoji pattern (targeting emojis)
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f100-\U0001f1ff"  # flags (iOS)
        "\U0001f200-\U0001f2ff"  # additional emojis
        "\U00002702-\U000027b0"  # Dingbats and other symbols
        "\U0001f780-\U0001f7ff"
        "\U0001f900-\U0001f9ff"
        "]+",
        flags=re.UNICODE,
    )

    segments = []  # To store text segments
    emojis = []  # To store emojis and their positions

    last_end = 0  # Track the end of the last match to segment the text

    firstIndex = False

    # Iterate over each emoji match
    for match in emoji_pattern.finditer(text):
        start, end = match.span()

        if start == 0:
            firstIndex = True

        # Append text segment before the emoji
        if last_end != start:
            segments.append(text[last_end:start])

        # Store emoji info (position and emoji)
        emojis.append(text[start:end])

        # Update last_end to the end of the current emoji
        last_end = end

    # Append any remaining text after the last emoji
    if last_end < len(text):
        segments.append(text[last_end:])

    return segments, emojis, firstIndex


# Example usage
if __name__ == "__main__":
    text = "😊😊😊😊😊😊😊😊 😊😊😊😊"
    segments, emojis, firstIndex = remove_punctuations_symbols_emojis(text)

    print("Text Segments:", segments)
    print("Emojis Info:", emojis)
    print("Is first: ", firstIndex)
