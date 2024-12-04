import json

# 讀取原始 JSON 檔案
with open("flores-200_to_lang.json", "r", encoding="utf-8") as f:
    flores_to_lang = json.load(f)

# 建立新的字典，將 key 和 value 互換
lang_to_flores = {value: key for key, value in flores_to_lang.items()}

# 將新的字典存成 JSON 檔案
with open("lang_to_flores-200.json", "w", encoding="utf-8") as f:
    json.dump(lang_to_flores, f, indent=4, ensure_ascii=False)

print("已成功將 key 和 value 互換並存成 lang_to_flores-200.json 檔案")
