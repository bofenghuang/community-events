#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

# model_name_or_path="openai/whisper-small"
model_name_or_path="bhuang/whisper-small-cv11-french-case-punctuation"
# model_name_or_path="bhuang/whisper-medium-cv11-french-case-punctuation"

tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"

outdir="./outputs/$tmp_model_id/results_cv11"


# todo: beam, lm, normalizer into tokenizer, suppress_tokens, audio normalization

# python run_eval_whisper_streaming_low_api_nbest.py \
#     --model_id $model_name_or_path \
#     --language "fr" \
#     --task "transcribe" \
#     --dataset "mozilla-foundation/common_voice_11_0" \
# 	--config "fr" \
# 	--split "test" \
#     --device "0" \
#     --log_outputs \
#     --max_eval_samples 10 \
#     --outdir ${outdir}_greedysampling_nbest10

python run_eval_whisper_streaming.py \
    --model_id $model_name_or_path \
    --language "fr" \
    --task "transcribe" \
    --dataset "mozilla-foundation/common_voice_11_0" \
	--config "fr" \
	--split "test" \
    --device "0" \
    --batch_size 16 \
    --log_outputs \
    --max_eval_samples 10 \
    --outdir ${outdir}_greedy
