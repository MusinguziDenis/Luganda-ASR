#!/bin/sh
# Run multi parallel the fine-tuning using 4 GPUs.
# All options are writen in finetuning_common_voice.json.

python -m torch.distributed.launch --nproc_per_node=4 finetuning_wav2vec.py  finetuning_wav2vec.json