# Machine Translation


Features:
* Language detection for over 170 languages.
* Translation pipeline using the NLLB-200 Distilled 600M model.
* Uses GPU (CUDA) if available for fast inference.


## Installation & Setup
### Prerequisites:
1. Python 3.8 or higher
2. Flask
3. Hugging Face Transformers library
4. FastText model
5. CUDA-enabled GPU (optional, for faster computation)

### Steps:
1. Clone the repository.
2. Install dependencies via pip:
```console
pip install -r requirements.txt
```
3. (Optional) Download the FastText language detection model and place it in the models folder:
```console
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P models/
```
4. Set the environment variable `det_conf` to a suitable confidence threshold for language detection (e.g., 0.5).
5. Run the API:
```console
det_conf=0.5 python api.py
```
## Components Overview
### FastText Language Detection
The FastText model (`lid.176.bin`) is pre-trained on a variety of languages. This model is used to predict the source language of the input text with a confidence score.

### NLLB-200 Translation Model
The `NLLB-200 Distilled 600M` model is a multilingual machine translation model capable of translating between numerous languages. The model is loaded using Hugging Face’s transformers library and fine-tuned for efficient GPU usage through memory-efficient configurations.

### Environment Variables
`det_conf`: The minimum confidence score (a float between 0 and 1) required for the detected language to be considered valid for translation. This prevents low-confidence predictions from triggering a translation attempt.

## Endpoints
### POST /translate
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
    "target_msg": "Translated text here."
}
```
* `source_lang`: Detected language of the input text.
* `is_trans`: Boolean indicating whether translation occurred.
* `target_msg`: Either the translated message or the original if no translation was needed.

#### Error Handling:
Returns a 500 error if no input text is provided.
Returns the original message if the detected source language and target language are the same.



