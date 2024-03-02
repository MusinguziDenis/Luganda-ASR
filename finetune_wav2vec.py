#!/usr/bin/env/ python
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import string

import datasets
import numpy as np
import torch
from datasets import load_dataset, load_metric
from packaging import version
from torch import nn
from pathlib import Path
from datasets import Audio


import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    is_apex_available,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint, is_main_process


if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) < version.parse("1.8"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    freezwav2vec: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to freeze the wav2vec2 model during training."}
    )

    attention_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout ratio for the attention probabilities."}
    )

    activation_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout ratio for activations inside the fully connected layer."}
    )

    hidden_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout probabilitiy for the hidden states."}
    )

    feat_proj_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout probabilitiy for the feature extractor."}
    )

    mask_time_prob: Optional[float] = field(
        default=0.05,
        metadata={"help": "The probability with which we mask entire feature vectors."}
    )

    layerdrop: Optional[float] = field(
        default=0.1,
        metadata={"help": "The probability to drop entire layers."}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using HfArgumentParser we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name : str = field(
        default= None,
        metadanamta={"help": "The name of the dataset to use (via the datasets library)."} 
    )

    dataset_config_name : str = field(
        default= None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."} 
    )

    dataset_data_dir: Optional['str']   = field(
        default=None,
        metadata={"help": "The directory where the dataset is stored."}
    )

    dataset_data_test_size: Optional[float] = field(
        default=0.2,
        metadata={"help": "The size of the test dataset."}
    )

    dataset_data_seed: Optional[int] = field(
        default=42,
        metadata={"help": "The seed to use for the test dataset."}
    )

    dataset_min_filesize: Optional[int] = field(
        default=0,
        metadata={"help": "The minimum filesize of the audio files to use."}
    )

    dataset_min_textlength: Optional[int] = field(
        default=0,
        metadata={"help": "The minimum text length of the audio files to use."}
    )

    train_split_name: Optional[str] = field(
        default="train+validation",
        metadata = {
            "help": "The name of the train split to use from the dataset."
        }
    )

    validation_split_name: Optional[str] = field(
        default = "validation",
        metadata = {
            "help": "The name of the validation split to use from the dataset."
        }
    )

    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum number of samples to use from the train split."}
    )

    max_val_samples: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum number of samples to use from the validation split."}
    )

    chars_to_ignore: Optional[str] = field(
        default="",
        metadata={"help": "A string of characters to remove from the transcripts."}
    )


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    
def min_filesize_and_textlength(row, filesize, textlength):
    return (Path(row['path']).stat().st_size >= filesize) and (len(row['sentence']) >= textlength)
    
def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith("json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Set --overwrite_output_dir to train anyway."
            )
        elif last_checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming training from {last_checkpoint}")
            
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers= [logging.StreamHandler(sys.stdout)]
    )

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")

    # Set the verbosity to info of the Transformers logger
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets:
    train_dataset = datasets.load_dataset(
        data_args.dataset_name, data_args.dataset_config_name, data_dir=data_args.dataset_data_dir,
        split=data_args.train_split_name, cache_dir=model_args.cache_dir
    )

    eval_dataset = datasets.load_dataset(
        data_args.dataset_name, data_args.dataset_config_name, data_dir=data_args.dataset_data_dir,
        split="test", cache_dir=model_args.cache_dir
    )

    if data_args.dataset_min_file != 0 or data_args.dataset_min_textlength != 0:
        train_dataset = train_dataset.filter(min_filesize_and_textlength,
                                             fn_kwargs = {
                                                 "filesize" : data_args.dataset_min_filesize,
                                                    "textlength" : data_args.dataset_min_textlength
                                             })
        
        eval_dataset = eval_dataset.filter(min_filesize_and_textlength,
                                                fn_kwargs = {
                                                    "filesize" : data_args.dataset_min_filesize,
                                                        "textlength" : data_args.dataset_min_textlength
                                                })
        
    # Create and save tokenizer
    chars_to_ignore_regex = f'[''.join(data_args.chars_to_ignore)}]'

    def remove_special_characters(batch):
        batch["text"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
        return batch
    
    train_dataset = train_dataset.map(remove_special_characters)
    eval_dataset = eval_dataset.map(remove_special_characters)

    def extract_all_chars(batch):
        all_text = " ".join(batch["text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}
    
    vocab_train = train_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=train_dataset.column_names)
    vocab_test = eval_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=eval_dataset.column_names)    

    vocab_list = sorted(list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0])))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict['[UNK]'] = len(vocab_dict)
    vocab_dict['[PAD]'] = len(vocab_dict)

    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    # load pretrained model and tokenizer
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # load pretrained model
    model = Wav2Vec2ForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        activation_dropout=model_args.activation_dropout,
        attention_dropout=model_args.attention_dropout,
        hidden_dropout=model_args.hidden_dropout,
        feat_proj_dropout=model_args.feat_proj_dropout,
        mask_time_prob=model_args.mask_time_prob,
        layerdrop=model_args.layerdrop,
        pad_tooken_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )


    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if data_args.max_val_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    # Preprocess the dataset
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000)).cast_column("audio", Audio(sampling_rate=16000))
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=16000)).cast_column("audio", Audio(sampling_rate=16000))

    def prepare_dataset(batch):
        audio = batch["audio"]

        # batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = processor(audio["array"], sampling_rate=16000).input_values[0]

        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch
    
    train_dataset = train_dataset.map(
        prepare_dataset,
        remove_columns=train_dataset.column_names["train"],
        batched = True,
        batch_size = training_args.per_device_train_batch_size,
        num_proc= data_args.preprocessing_num_workers
        )
    
    eval_dataset = eval_dataset.map(
        prepare_dataset,
        remove_columns=eval_dataset.column_names["train"],
        batched = True,
        batch_size = training_args.per_device_train_batch_size,
        num_proc= data_args.preprocessing_num_workers
        )
    
    # Metric
    wer_metric = load_metric("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    # Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # Freeze the feature extractor
    if model_args.freezwav2vec:
        model.freeze_feature_extractor()

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        # save the feature extractor and the tokenizer
        if is_main_process(training_args.local_rank):
            processor.save_pretrained(training_args.output_dir)

        training_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = training_result.metrics
        max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)

        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        results = metrics

        return results
    
if __name__ == "__main__":
    main()
