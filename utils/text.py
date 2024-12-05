# import unicodedata
# import regex as re

import emoji


def split_string_by_emoji(text: str) -> tuple:
    """
    將字串分割成連續的 non-emoji 與連續的 emoji 的部分。

    Args:
        text: 輸入字串。

    Returns:
        一個包含分割後字串的列表，以及一個指示每個部分是否為 emoji 的布林值列表。
    """

    if not text:
        return [], []

    result_strings = []
    is_emoji_list = []
    start_index = 0
    current_is_emoji = None
    i = 0

    while i < len(text):
        possible_emoji = ""
        j = i
        is_char_emoji = None
        emoji_length = 0

        while j < len(text):
            possible_emoji += text[j]
            if possible_emoji in emoji.EMOJI_DATA:
                is_char_emoji = True
                emoji_length = j - i + 1
                j += 1
            elif any(char in emoji.EMOJI_DATA for char in possible_emoji):
                j += 1
            else:
                if is_char_emoji is None:
                    is_char_emoji = False
                break

        if current_is_emoji is None:
            current_is_emoji = is_char_emoji
        elif current_is_emoji != is_char_emoji:
            result_strings.append(text[start_index:i])
            is_emoji_list.append(current_is_emoji)
            start_index = i
            current_is_emoji = is_char_emoji

        i += emoji_length or 1

    result_strings.append(text[start_index:])
    is_emoji_list.append(current_is_emoji)

    return result_strings, is_emoji_list


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


if __name__ == "__main__":
    test_string = " Cô gái này đẹp quá! ❤️❤️❤️❤️❤️ Anh yêu em! "
    strings, is_emoji = split_string_by_emoji(test_string)
    print("分割後的字串列表:", strings)
    print("是否為 emoji 的列表:", is_emoji)


# def normalize_text(text):
#     """
#     文字正規化函式，具有以下功能：
#     1. 移除 emoji
#     2. 保留中文、英文、泰文、越南文、韓文字符
#     3. 去除與語言無關的特殊符號
#     4. 轉換為小寫

#     Args:
#         text (str): 輸入的原始文字

#     Returns:
#         str: 正規化後的文字
#     """
#     # 定義要保留的 Unicode 區塊
#     unicode_ranges = {
#         "CJK": (0x4E00, 0x9FFF),  # 中文漢字
#         "Latin": (0x0000, 0x007F),  # 基本拉丁字母（英文）
#         "Thai": (0x0E00, 0x0E7F),  # 泰文
#         "Vietnamese": (0x00C0, 0x1EF9),  # 越南文音調字母
#         "Korean": (0xAC00, 0xD7A3),  # 韓文音節
#     }

#     def is_allowed_char(char):
#         """檢查字元是否在允許的 Unicode 區塊中"""
#         code = ord(char)
#         return any(start <= code <= end for start, end in unicode_ranges.values())

#     # 去除 emoji 和特殊符號，保留允許的字元
#     normalized_text = "".join(
#         char if is_allowed_char(char) or char.isspace() else " "
#         for char in unicodedata.normalize("NFKD", text)
#     )

#     # 去除多餘空白並轉為小寫
#     return " ".join(normalized_text.lower().split())
