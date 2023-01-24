#!/usr/bin/env bash

export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

export CUDA_VISIBLE_DEVICES=4

# model_name_or_path="./outputs/whisper-medium-cv11-german-punct"
# model_name_or_path="./outputs/whisper-medium-ft-lr6e6-bs256-steps2k-dropout005-de"
# model_name_or_path="./outputs/whisper-small-ft-lr6e6-bs256-steps2k-dropout005"
# model_name_or_path="./outputs/whisper-large-v2-ft-lr4e6-bs256-steps3k-dropout005-casepunc-ds"
# outdir="$model_name_or_path/results"

# model_name_or_path="openai/whisper-small"
model_name_or_path="openai/whisper-medium"
# model_name_or_path="bofenghuang/whisper-medium-cv11-french"

tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"
outdir="./outputs/$tmp_model_id/results"

python run_eval_whisper_streaming.py \
    --model_id $model_name_or_path \
    --language "fr" \
    --task "transcribe" \
    --dataset "mozilla-foundation/common_voice_11_0" \
    --config "fr" \
    --split "test" \
    --device "0" \
    --fp16 \
    --batch_size 16 \
    --log_outputs \
    --outdir ${outdir}_cv11_fr_greedy


model_name_or_path="openai/whisper-small"
tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"
outdir="./outputs/$tmp_model_id/results"
python run_eval_whisper_streaming.py \
    --model_id $model_name_or_path \
    --language "fr" \
    --task "transcribe" \
    --dataset "mozilla-foundation/common_voice_11_0" \
    --config "fr" \
    --split "test" \
    --device "0" \
    --fp16 \
    --batch_size 16 \
    --log_outputs \
    --outdir ${outdir}_cv11_fr_greedy


model_name_or_path="openai/whisper-large-v2"
tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"
outdir="./outputs/$tmp_model_id/results"
python run_eval_whisper_streaming.py \
    --model_id $model_name_or_path \
    --language "fr" \
    --task "transcribe" \
    --dataset "mozilla-foundation/common_voice_11_0" \
    --config "fr" \
    --split "test" \
    --device "0" \
    --fp16 \
    --batch_size 8 \
    --log_outputs \
    --outdir ${outdir}_cv11_fr_greedy
