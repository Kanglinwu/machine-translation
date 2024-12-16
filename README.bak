# Machine Translation

Features:
* Language detection for over 170 languages.
* Translation pipeline using the NLLB-200 Distilled 600M model.
* Uses GPU (CUDA) if available for fast inference.


## Installation & Setup

### Prerequisites:

1. Python 3.10.12
2. Flask
3. Hugging Face Transformers library
4. FastText model
5. CUDA-enabled GPU (optional, for faster computation)

### Setup Instructions

Follow these steps to set up the machine translation API:

1. **Clone the Repository**

   Clone the machine translation repository from GitHub:

   ```bash
   git clone https://github.com/United-Link/machine-translation.git
    ```

2. **Install Dependencies**

    Navigate to the project directory and install the required dependencies using pip:

    ```bash
    cd machine-translation
    pip install -r requirements.txt
    ```

3. **Download the FastText Language Detection Model**
    
    Download the FastText language detection model and place it in the models folder:
    
    ```bash
    wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P models/
    ```

## Run Single Server

This guide outlines the steps to set up single servers.

### Steps:

1. **Set the Environment Variable**

    Configure the environment variable `det_conf` to specify a suitable confidence threshold for language detection (e.g., 0.3):
    
    ```bash
    export det_conf=0.3
    ```

2. **Run the API**

    Start the API using the following command:

    ```bash
    python api.py
    ```
    or if you didn't setup the environment variable `det_conf`:

   ```bash
   det_conf=0.3 python api.py
   ```

## Components Overview

### FastText Language Detection

The FastText model (`lid.176.bin`) is pre-trained on a variety of languages. This model is used to predict the source language of the input text with a confidence score.

### NLLB-200 Translation Model

The `NLLB-200 Distilled 600M` model is a multilingual machine translation model capable of translating between numerous languages. The model is loaded using Hugging Face’s transformers library and fine-tuned for efficient GPU usage through memory-efficient configurations.

### Environment Variables

`det_conf`: The minimum confidence score (a float between 0 and 1) required for the detected language to be considered valid for translation. This prevents low-confidence predictions from triggering a translation attempt.

## Endpoints

### POST 10.10.10.48/translate

Description:
This endpoint accepts a JSON payload containing the message to be translated and the target language. It detects the source language of the message and translates it to the target language if they differ.

#### Request Payload

```json
{
    "msg": "Text to be translated.",
    "target_lang": "en"
}
```
* `msg`: (string) The input text to translate.
* `target_lang`: (string) The target language code (e.g., "en" for English, "zh" for Chinese).

#### Response

```json
{
    "source_lang": "es",
    "is_trans": true,
    "target_msg": "Translated text here.",
    "all_emoji": false,
    "model": "facebook/nllb-200-distilled-600M",
    "lang_undefined": false
}
```
* `source_lang`: Detected language of the input text.
* `is_trans`: Boolean indicating whether translation occurred.
* `target_msg`: Either the translated message or the original if no translation was needed.
* `all_emoji`: Boolean indicating whether message is all emojis.
* `model`: The model used for translation.
* `lang_undefined`: Boolean indicating whether language is out of service scope.

#### Error Handling:

Returns a 500 error if no input text is provided.
Returns the original message if the detected source language and target language are the same.

### Example

#### Request

To send a translation request to the API, you can use the following `curl` command:

```bash
curl -X POST http://10.10.10.48:2486/translate \
    -H "Content-Type: application/json" \
    -d '{"msg": "Where are you", "target_lang": "zh"}'
```

#### Expected Response

Upon making the request, you will receive a response in JSON format similar to the following:

```bash
{
   "source_lang": "en",
   "is_trans": true,
   "target_msg": "你在哪里?",
   "all_emoji": false,
   "model": "facebook/nllb-200-distilled-600M",
   "lang_undefined": false
}
```

## Run Multiple Servers with Load Balancing Mechanism

This guide outlines the steps to set up multiple servers using Docker and Docker Compose, incorporating a load balancing mechanism.

### Steps:

1. **Build the Server Docker Image**

   To build the Docker image for the server, use the following command:
   
   ```bash
   docker build -t server .
   ```
   
   This command creates an image named `server` from the current directory (`.`), using the Dockerfile in the root of your server project.

2. **Build the Nginx Docker Image**
   
   Similarly, to build the Docker image for the Nginx load balancer, use the following command:
   
   ```bash
   cd nginx
   docker build -t nginx_load_balancer .
   ```
   
   This will generate an image named `nginx_load_balancer` from the current directory, using the Dockerfile configured for Nginx.

3. **Run Docker Compose**

    Start the server with Docker Compose while setting the environment variable det_conf:
    
    ```bash
    det_conf=0.3 docker compose up -d
    ```
 4. **Stop the Server**

    To stop and remove the running server, execute:
    
    ```bash
    det_conf=0.3 docker compose down
    ```

## Components Overview

### Nginx

Nginx serves as a reverse proxy and load balancer, distributing incoming traffic to multiple server instances for improved performance and reliability.

### Docker Image

A Docker image is a lightweight, stand-alone, executable package that includes everything needed to run a piece of software, including the code, runtime, libraries, and environment variables.

### Docker Container

A Docker container is a runnable instance of a Docker image. Containers are isolated from each other and share the OS kernel, enabling efficient resource utilization.

### Docker Compose

Docker Compose is a tool for defining and managing multi-container Docker applications. With Compose, you can define all your services, networks, and volumes in a single docker-compose.yml file and manage them with simple commands.

## FAQ

1. **What should I do if I can't leverage my GPU?**

   To ensure that you can utilize your GPU effectively, make sure you install the correct version of PyTorch with CUDA support.


   For more detailed information, visit the [PyTorch Previous Versions page](https://pytorch.org/get-started/previous-versions/).
   For Linux environments with GPUs and CUDA version 12.4 or above, follow these installation instructions:
   
   **Using Conda**
   
   ```bash
   conda install pytorch==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
   ```
   
   **Using Pip**
   
   ```bash
   pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
   ```
