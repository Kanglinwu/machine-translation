# Models
## FastText Language Detection
The FastText model (`lid.176.bin`) is pre-trained on a variety of languages. This model is used to predict the source language of the input text with a confidence score.

## NLLB-200 Translation Model
The `NLLB-200 Distilled 600M` model is a multilingual machine translation model capable of translating between numerous languages. The model is loaded using Hugging Face’s transformers library and fine-tuned for efficient GPU usage through memory-efficient configurations.
