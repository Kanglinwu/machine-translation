# 使用 Ubuntu 作為基礎映像以便更好地控制 Python 版本
FROM ubuntu:22.04

# 避免交互式安裝過程
ENV DEBIAN_FRONTEND=noninteractive

# 安裝系統依賴和 Python 3.10
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 設置 Python 3.10 為默認 Python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10

# 設置工作目錄
WORKDIR /app

# 創建虛擬環境
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 安裝 PyTorch 和其他依賴
RUN pip install --no-cache-dir \
    torch \
    transformers \
    sentencepiece \
    protobuf \
    huggingface_hub \
    flask \
    flask-cors \
    gevent \
    marshmallow \
    pyyaml \
    fasttext

# 複製項目文件
COPY api.py /app
COPY ./models ./conf ./docs /app/