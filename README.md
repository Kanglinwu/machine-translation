# machine-translation


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
5. Run the Flask application:
```console
python api.py
```
