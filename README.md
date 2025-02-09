### Finetune ASR Models

This repo contains code for finetuning popular ASR models such as Wav2Vec-XLS-R, Wav2Vec2-Bert, Whisper and MMS on Mozilla Common Voice dataset.

We use the [Mozilla Common Voice Dataset Luganda Dataset](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) but this can be replaced by other datasets that contain Luganda data for example
* [Fleurs Dataset](https://huggingface.co/datasets/google/fleurs)


The code was inpired by several HuggingFace tutorials
* [Fine-Tune Wav2Vec Bert for low resource ASR with transformers](https://huggingface.co/blog/fine-tune-w2v2-bert)
* [Fine-tuning XLS-R for Multi-Lingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2)
* [Boosting Wav2Vec2 with n-grams in 🤗 Transformers](https://huggingface.co/blog/wav2vec2-with-ngram)
* [Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper)

The model is Wav2Vec Bert model with a WER of 19.33.

### Model Available

[HUGGINGFACE](https://huggingface.co/dmusingu/w2v-bert-2.0-luganda-CV-train-validation-7.0)

You can download the model from Huggingface and use it directly to produce the same results.