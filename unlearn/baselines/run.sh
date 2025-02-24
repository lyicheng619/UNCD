# #!/bin/bash


# algo="ga"
# CORPUS="wmdp"

# FORGET="../data/forget.txt"
# RETAIN="../data/retain.txt"

# TARGET_DIR='/hy-tmp/mistral'
# LLAMA_DIR='/hy-tmp/mistral'

# MAX_LEN=2048
# EPOCHS=1
# LR='1e-5'
# PER_DEVICE_BATCH_SIZE=1 # 8 GPUs
# FT_EPOCHS=1
# FT_LR='1e-5'


# python unlearn.py \
#         --algo $algo \
#         --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
#         --data_file $FORGET --retain_data_file $RETAIN \
#         --out_dir "./ckpt/$CORPUS/$algo" \
#         --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
#         --per_device_batch_size $PER_DEVICE_BATCH_SIZE 




#!/bin/bash

algo="ga"  # Change this to 'gd', 'tv', or 'rmu' as needed
CORPUS="wmdp"

FORGET="../data/forget.txt"
RETAIN="../data/retain.txt"

TARGET_DIR='/hy-tmp/mistral'
LLAMA_DIR='/hy-tmp/mistral'

MAX_LEN=2048
EPOCHS=1
LR='1e-5'
PER_DEVICE_BATCH_SIZE=1  # 8 GPUs
FT_EPOCHS=1
FT_LR='1e-5'
N_STEP=3  # Adjust the number of checkpoints
ALPHA=1.0  # Required for Task Vector (TV)

# Base command
CMD="python unlearn.py \
        --algo $algo \
        --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
        --data_file $FORGET --retain_data_file $RETAIN \
        --out_dir \"./ckpt/$CORPUS/$algo\" \
        --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
        --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
        --n_step $N_STEP"

# Optionally add resume flag
if [ "$RESUME" = "true" ]; then
    CMD="$CMD --resume_from_checkpoint"
fi

# Handle RMU-specific arguments
if [ "$algo" = "rmu" ]; then
    CMD="$CMD --model_name_or_path HuggingFaceH4/zephyr-7b-beta \
              --steering_coeffs '20,20' \
              --checkpoint_path None \
              --retain_corpora wikitext,wikitext \
              --forget_corpora bio-forget-corpus,cyber-forget-corpus \
              --batch_size 4 --max_num_batches 80 \
              --layer_id 7 --layer_ids '5,6,7' --param_ids '6'"
fi

# Handle Task Vector (TV) specific arguments
if [ "$algo" = "tv" ]; then
    CMD="$CMD --alpha $ALPHA"
fi

# Run the command
echo "Running: $CMD"
eval $CMD