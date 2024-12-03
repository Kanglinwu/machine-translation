# 使用官方 PyTorch 映像作為基礎
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 設置工作目錄
WORKDIR /app

# 安裝必要的系統依賴
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 依賴
RUN pip3 install --no-cache-dir \
    transformers \
    torch \
    sentencepiece \
    protobuf \
    huggingface_hub \
    flask \
    flask-cors \
    gevent \
    marshmallow \
    pyyaml \
    fasttext \
    logging \
    pathlib \
    typing

# 複製項目文件
COPY api.py .
COPY models/ .
COPY conf/ .
COPY docs/ .