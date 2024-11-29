import re
import csv
import timeit
from pathlib import Path
import subprocess

import fasttext
import pandas as pd
from huggingface_hub import hf_hub_download


URL_MODEL_176 = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
MODELS_DIR = Path("../models/").resolve()


def download_model_file(url, output_dir: Path, filename: str):
    output_path = output_dir / filename

    if not output_path.exists():
        try:
            subprocess.run(["wget", url, "-O", f"{output_path}"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"下载 {url} 失败: {e}")
        except FileNotFoundError:
            print("wget 命令未找到，请确保已安装 wget 并添加到 PATH 环境变量中。")

    return output_path


def read_csv_to_custom_dict(csv_file: Path):
    try:
        with csv_file.open(encoding="utf-8") as file:
            reader = csv.reader(file)  # 使用 reader 以獲取列表形式的行
            header = next(reader)  # 獲取第一行 (標題行)
            key_index = 1  # 第二列的索引
            value_index = 0  # 第一列的索引

            if len(header) <= max(key_index, value_index):
                print("CSV 檔案列數不足。")
                return None

            code2name = []
            for row in reader:
                key = row[key_index]
                value = row[value_index]
                code2name.append((key, value))
            return dict(code2name)
    except FileNotFoundError:
        print(f"找不到檔案: {csv_file}")
        return None
    except Exception as e:
        print(f"讀取 CSV 時發生錯誤: {e}")
        return None


# def get_lang_list():


if __name__ == "__main__":
    text = "PLAYER"

    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    model_176_path = MODELS_DIR / "lid.176.bin"
    if not model_176_path.exists():
        model_176_path = download_model_file(URL_MODEL_176, MODELS_DIR, "lid.176.bin")
    model_176 = fasttext.load_model(str(model_176_path))
    prediction_176 = model_176.predict("PLAYER", k=5)

    # execution_time = timeit.timeit(
    #     lambda: model_176.predict("PLAYER", k=10), number=1000
    # )
    # print(f"Execution time: {execution_time:.6f} seconds")

    df = pd.read_csv("iso-639-3.tab", sep="\t")

    code2name = {}
    lang_codes = [str(code).replace("__label__", "") for code in model_176.get_labels()]
    for code in lang_codes:
        name = df[(df["Id"].str.strip() == code) | (df["Part1"].str.strip() == code)]

        if not name.empty:
            lang_name = re.sub(r"\s*\(.*?\)\s*", "", name["Ref_Name"].iloc[0])
            code2name[code] = lang_name
        else:
            print("unknown:", code)

    # for code in sorted(code2name.keys()):
    #     print(code, code2name[code])

    print(prediction_176)
    # print(model_176_path.stat().st_size)
    # print(model_176.get_labels())

    # model_217_path = MODELS_DIR / "lid.217.bin"
    # if not model_217_path.exists():
    #     model_217_path = Path(
    #         hf_hub_download(
    #             repo_id="facebook/fasttext-language-identification",
    #             local_dir=f"{MODELS_DIR}",
    #             filename="model.bin",
    #         )
    #     )
    #     model_217_path.rename(MODELS_DIR / "lid.217.bin")
    # model_217 = fasttext.load_model(str(model_217_path))
    # prediction_217 = model_176.predict("PLAYER", k=1)

    # print(prediction_217)
    # print(Path(model_217_path).stat().st_size)
    # print(model_217.get_labels())

    # code2name = read_csv_to_custom_dict(Path("FLORES-200_code.csv"))
    # if code2name:
    #     for code in sorted(code2name.keys()):
    #         print(code, code2name[code])
