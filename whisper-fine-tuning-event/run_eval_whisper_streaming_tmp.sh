#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

# model_name_or_path="openai/whisper-small"
# model_name_or_path="bhuang/whisper-small-cv11-french-case-punctuation"
# model_name_or_path="bhuang/whisper-medium-cv11-french-case-punctuation"
# model_name_or_path="bofenghuang/whisper-medium-cv11-german-punct"

# tmp_model_id="$(echo "${model_name_or_path}" | sed -e "s/-/\_/g" -e "s/[ |=/]/-/g")"
# outdir="./outputs/$tmp_model_id/results_cv11"

# model_name_or_path="./outputs/whisper-medium-cv11-german-punct"
# model_name_or_path="./outputs/whisper-medium-ft-lr6e6-bs256-steps2k-dropout005-de"
# model_name_or_path="./outputs/whisper-small-ft-lr6e6-bs256-steps2k-dropout005"
model_name_or_path="./outputs/whisper-large-v2-ft-lr4e6-bs256-steps3k-dropout005-casepunc-ds"
outdir="$model_name_or_path/results"


# todo: beam, lm, normalizer into tokenizer, suppress_tokens, audio normalization

    # --max_eval_samples 10 \


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
#     --outdir ${outdir}_cv11_greedysampling_nbest10

# python run_eval_whisper_streaming.py \
#     --model_id $model_name_or_path \
#     --language "fr" \
#     --task "transcribe" \
#     --dataset "mozilla-foundation/common_voice_11_0" \
# 	--config "fr" \
# 	--split "test" \
#     --device "0" \
#     --batch_size 16 \
#     --log_outputs \
#     --max_eval_samples 10 \
#     --outdir ${outdir}_cv11_greedy

# # medium
# python run_eval_whisper_streaming.py \
#     --model_id $model_name_or_path \
#     --language "de" \
#     --task "transcribe" \
#     --dataset "mozilla-foundation/common_voice_11_0" \
# 	--config "de" \
# 	--split "test" \
#     --device "0" \
#     --batch_size 32 \
#     --log_outputs \
#     --outdir ${outdir}_cv11_greedy

python run_eval_whisper_streaming.py \
    --model_id $model_name_or_path \
    --language "de" \
    --task "transcribe" \
    --dataset "mozilla-foundation/common_voice_11_0" \
	--config "de" \
	--split "test" \
    --device "0" \
    --fp16 \
    --batch_size 8 \
    --log_outputs \
    --outdir ${outdir}_cv11_greedy