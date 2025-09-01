## LUGANDA-ASR

This repository demonstrates how to fine-tune popular Automatic Speech Recognition (ASR) models for Luganda using the Hugging Face Trainer API. The provided notebooks walk through data loading, preprocessing, and tokenizer creation, followed by model training. The ASR models explored include:
* Wav2Vec-XLS-R
* Wav2Vec2-Bert
* Whisper 
* MMS

Below is a table with details of open-source Luganda speech recognition datasets that can be used for training and evaluation:

| Dataset | Number of Hours | Type of Speech |
|---------|-----------------| ---------------|
|[Mozilla Common Voice Dataset Luganda Dataset](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | 560 | Read |
| [Fleurs Dataset](https://huggingface.co/datasets/google/fleurs) | 11 | Spontaneous |
| [The Makerere Radio Speech Corpus: A Luganda Radio Corpus for Automatic Speech Recognition](https://doi.org/10.5281/zenodo.5855016) | 155 | Spontaneous |
| [Yogera](https://github.com/AI-Lab-Makerere/Yogera-Dataset-Metadata) | 251 | Read |



The code was inpired by several HuggingFace tutorials
* [Fine-Tune W2V2-Bert for low-resource ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-w2v2-bert)
* [Fine-tuning XLS-R for Multi-Lingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2)
* [Boosting Wav2Vec2 with n-grams in 🤗 Transformers](https://huggingface.co/blog/wav2vec2-with-ngram)
* [Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper)


You can access some of the finetuned models here
* [dmusingu/WHISPER-SMALL-LUGANDA-ASR-CV-14](https://huggingface.co/dmusingu/WHISPER-SMALL-LUGANDA-ASR-CV-14)
    This model obtained a WER of 29.9 on the test set of the CommonVoice 14 test dataset.
* [dmusingu/XLS-R-SWAHILI-ASR-CV-14-1B](https://huggingface.co/dmusingu/XLS-R-SWAHILI-ASR-CV-14-1B)
    This model obtained a WER of 27.94 on the test set of the CommonVoice 14 test dataset.
* [dmusingu/w2v-bert-2.0-luganda-CV-train-validation-7.0](https://huggingface.co/dmusingu/w2v-bert-2.0-luganda-CV-train-validation-7.0)
    This model obtained a WER of 19.33 on the test set of the CommonVoice 7 test dataset.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/neubig/starter-repo/blob/main/LICENSE) file for details.

## Citation
This was created by Denis Musinguzi to demonstrate how to finetune ASR models.

```@misc{musinguzi2025asr,
  author = {Denis Musinguzi},
  title = {Luganda ASR models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/MusinguziDenis/Luganda-ASR}}
}
```



