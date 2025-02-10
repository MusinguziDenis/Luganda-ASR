### Finetune ASR Models

This repo contains code for finetuning popular ASR models
* Wav2Vec-XLS-R
* Wav2Vec2-Bert
* Whisper 
* MMS

Luganda ASR Datasets
* [Mozilla Common Voice Dataset Luganda Dataset](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)
* [Fleurs Dataset](https://huggingface.co/datasets/google/fleurs)
* [The Makerere Radio Speech Corpus: A Luganda Radio Corpus for Automatic Speech Recognition](https://doi.org/10.5281/zenodo.5855016)
* [Yogera](https://github.com/AI-Lab-Makerere/Yogera-Dataset-Metadata)


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

