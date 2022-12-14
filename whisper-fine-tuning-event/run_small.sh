#!/usr/bin/env bash
# Copyright 2022 Bofeng Huang

# export TRANSFORMERS_CACHE=/rd_storage/<user>/.cache/huggingface/transformers/
# export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

export WANDB_PROJECT=hf-whisper-sprint-v2

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
export OMP_NUM_THREADS=1

# export CUDA_VISIBLE_DEVICES=1,3
export CUDA_VISIBLE_DEVICES=0

# https://pytorch.org/docs/stable/elastic/run.html
# export HOST_NODE_ADDR="localhost:29000"

# python run_speech_recognition_seq2seq_streaming.py \
# python -m torch.distributed.launch \
# torchrun \
#     --rdzv_endpoint=$HOST_NODE_ADDR \
# 	--nproc_per_node 2 run_speech_recognition_seq2seq_streaming.py \
python run_speech_recognition_seq2seq_streaming.py \
    --dataset_name="mozilla-foundation/common_voice_11_0" \
	--dataset_config_name="fr" \
    --train_split_name="train+validation" \
    --eval_split_name="test" \
    --text_column_name="sentence" \
	--use_auth_token \
	--max_duration_in_seconds="30" \
	--language="french" \
	--task="transcribe" \
    --model_name_or_path="openai/whisper-small" \
	--output_dir="./outputs/hf_event_fr/whisper-small-ft-lr6e6-bs256-steps4k-adamw_bnb_8bit" \
    --overwrite_output_dir \
    --max_steps="4000" \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
	--gradient_accumulation_steps="4" \
    --learning_rate="6.25e-6" \
    --optim="adamw_bnb_8bit" \
    --warmup_steps="200" \
	--weight_decay "0.01" \
    --logging_steps="25" \
    --evaluation_strategy="steps" \
    --eval_steps="500" \
    --save_strategy="steps" \
    --save_steps="500" \
	--save_total_limit="5" \
	--metric_for_best_model="wer" \
	--greater_is_better="False" \
	--load_best_model_at_end \
    --freeze_feature_encoder="False" \
	--use_cache="False" \
    --fp16 \
    --gradient_checkpointing \
    --predict_with_generate \
    --generation_max_length="225" \
    --generation_num_beams="1" \
    --do_train \
    --do_eval

# --push_to_hub
# todo: --dataloader_num_workers="8" \ got error "Sharding a CyclingMultiSourcesExamplesIterable is not implemented"
# --max_steps="5000" \  from 5k to 10k
# mcv fr train+split 501123

# debug
# --max_train_samples="100" \
# --max_eval_samples="100" \
# --max_steps="10" \
# --eval_steps="5" \
