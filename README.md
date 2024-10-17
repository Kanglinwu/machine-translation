# machine-translation


Features:
Language detection for over 170 languages.
Translation pipeline using the NLLB-200 Distilled 600M model.
Uses GPU (CUDA) if available for fast inference.

1. Installation & Setup
Prerequisites:
Python 3.8 or higher
Flask
Hugging Face Transformers library
FastText model
CUDA-enabled GPU (optional, for faster computation)
Steps:
Clone the repository.
Install dependencies via pip:
pip install flask torch transformers fasttext
Download the FastText language detection model and place it in the models folder:
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P models/
Set the environment variable det_conf to a suitable confidence threshold for language detection (e.g., 0.5).
Run the Flask application:
```
python api.py
```
