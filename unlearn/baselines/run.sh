#!/bin/bash


algo="tv"
CORPUS="wmdp"

FORGET="../data/forget.txt"
RETAIN="../data/retain.txt"

TARGET_DIR='/hy-tmp/mistral'
LLAMA_DIR='/hy-tmp/mistral'

MAX_LEN=2048
EPOCHS=1
LR='1e-5'
PER_DEVICE_BATCH_SIZE=1 # 8 GPUs
FT_EPOCHS=1
FT_LR='1e-5'


python unlearn.py \
        --algo $algo \
        --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
        --data_file $FORGET --retain_data_file $RETAIN \
        --out_dir "./ckpt/$CORPUS/$algo" \
        --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
        --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
        --alpha 1000.0
