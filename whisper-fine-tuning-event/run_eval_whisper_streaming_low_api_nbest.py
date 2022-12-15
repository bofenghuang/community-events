import argparse
import os
from typing import Dict

import evaluate
import torch
from datasets import Audio, Dataset, load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from normalize_text_hf_sprint import FrenchTextNormalizer

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# normalizer = BasicTextNormalizer()
normalizer = FrenchTextNormalizer()


def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )


def normalise(batch):
    batch["norm_text"] = normalizer(get_text(batch))
    return batch


def data(dataset):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "target": item["norm_text"]}


def main(args):

    processor = AutoProcessor.from_pretrained(args.model_id, language=args.language, task=args.task)

    # model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id)
    model_args = {}
    if args.fp16:
        model_args["torch_dtype"] = torch.float16
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, **model_args)
    model.eval()
    model = model.to(args.device)

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    print(f"Model `forced_decoder_ids`: {model.config.forced_decoder_ids}")

    # batch_size = args.batch_size
    # pipe = pipeline(
    #     "automatic-speech-recognition", model=args.model_id, device=args.device
    # )

    # pipe.model.config.forced_decoder_ids = (
    #     pipe.tokenizer.get_decoder_prompt_ids(
    #         language=args.language, task="transcribe"
    #     )
    # )

    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=args.streaming,
        use_auth_token=True,
    )

    # Only uncomment for debugging
    dataset = dataset.take(args.max_eval_samples)

    # todo: max audio length, text token length
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(normalise)
    dataset = dataset.filter(is_target_text_in_range, input_columns=["norm_text"])

    # predictions = []
    # references = []

    # for out in pipe(data(dataset), batch_size=batch_size):
    #     predictions.append(normalizer(out["text"]))
    #     references.append(out["target"][0])

    # wer = wer_metric.compute(references=references, predictions=predictions)
    # # wer = round(100 * wer, 2)

    # print("WER:", wer)

    gen_greedy = {"do_sample": False, "num_beams": 1}
    gen_greedy_sampling = {"do_sample": True, "num_beams": 1}
    # While nucleus sampling can generate text free of repetitions, the semantic coherence of the generated text is not well-maintained.
    gen_nucleus = {"do_sample": True, "num_beams": 1, "top_p": 0.95, "top_k": 0}
    # reducing the temperature brings nucleus sampling closer to greedy search, which can be seen as a trade-off between greedy search and nucleus sampling.
    gen_nucleus_temperature = {"do_sample": True, "num_beams": 1, "top_p": 0.95, "top_k": 0, "temperature": 0.7}

    gen_beam = {"do_sample": False, "num_beams": 5}
    gen_beam_sampling = {"do_sample": True, "num_beams": 5}
    gen_beam_group = {"do_sample": False, "num_beams": 10, "num_beam_groups": 2}

    # When generating output, contrastive search jointly considers
    # (i) the probability predicted by the language model to maintain the semantic coherence
    # between the generated text and the prefix text
    # (ii) the similarity with respect to the previous context to avoid model degeneration.
    gen_contrastive_search = {"top_k": 6, "penalty_alpha": 0.6}


    # !
    # nbest = 10
    nbest = 5

    gen_greedy_sampling_nbest = {**gen_greedy_sampling, "return_dict_in_generate": True, "output_scores": True, "num_return_sequences": nbest}
    gen_beam_nbest = {**gen_beam, "return_dict_in_generate": True, "output_scores": True, "num_return_sequences": nbest}

    gen_kwargs = {
        "max_new_tokens": 225,
        # "max_new_tokens": 40,
        # **gen_greedy_sampling_nbest,
        **gen_beam_nbest,
        # "repetition_penalty"
        # "length_penalty"
        # "no_repeat_ngram_size"
        # "bad_words_ids"
        # "num_return_sequences"
    }

    results = []

    # run streamed inference
    for example_idx, example in enumerate(data(dataset)):
        if not example["target"]:
            raise ValueError(example)
            # continue

        example_output = {
            args.id_column_name: example.get(args.id_column_name, f"{example_idx:09d}"),
            "target": example["target"],
        }
    
        # bh: synchronised process and forward, this can be improved by dataloader
        inputs = processor(example["array"], sampling_rate=16_000, return_tensors="pt")
        input_features = inputs.input_features
        input_features = input_features.to(args.device)

        if args.fp16:
            input_features = input_features.half()

        # generated_ids = model.generate(inputs=input_features, **gen_kwargs)
        generated_out = model.generate(inputs=input_features, **gen_kwargs)
        # print(generated_out)
        generated_ids = generated_out.sequences  # shape BS x NBEST
        # ! beam
        generated_scores = generated_out.sequences_scores.cpu().tolist()

        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # normalize prediction
        predictions = [normalizer(prediction) for prediction in transcriptions]

        for i in range(nbest):
            example_output[f"prediction_{i}"] = predictions[i]
            example_output[f"score_{i}"] = generated_scores[i]

        results.append(example_output)

        # batch["target"] = batch[args.text_column_name]
        # normalize target
        # batch["target"] = normalize_text(batch[args.text_column_name], invalid_chars_regex)
        # batch["target"] = [normalizer(target) for target in batch[args.text_column_name]]

    # compute metrics
    references = [result_["target"] for result_ in results]
    predictions = [result_["prediction_0"] for result_ in results]
    wer_result = wer_metric.compute(references=references, predictions=predictions)
    cer_result = cer_metric.compute(references=references, predictions=predictions)

    # print & log results
    result_str = f"WER: {wer_result}\n" f"CER: {cer_result}"
    print(result_str)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    with open(f"{args.outdir}/eval_results.txt", "w") as f:
        f.write(result_str)

    if args.log_outputs is not None:
        pred_file = f"{args.outdir}/log_predictions.txt"
        target_file = f"{args.outdir}/log_targets.txt"

        # ! to adjust for your cases
        with open(pred_file, "w") as p, open(target_file, "w") as t:            
            for result in results:
                # p.write(batch[args.id_column_name] + "\t" + str(batch[args.start_column_name]) + "\t" + str(batch[args.end_column_name]) + "\t" + batch["prediction"] + "\n")
                # t.write(batch[args.id_column_name] + "\t" + str(batch[args.start_column_name]) + "\t" + str(batch[args.end_column_name]) + "\t" + batch["target"] + "\n")
                # p.write(batch[args.id_column_name] + "\t" + batch["prediction"] + "\n")
                p.write(result[args.id_column_name] + "\t" + "\t".join([result[f"prediction_{i}"] for i in range(nbest)]) + "\t" + "\t".join([str(result[f"score_{i}"]) for i in range(nbest)]) + "\n")
                t.write(result[args.id_column_name] + "\t" + result["target"] + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mozilla-foundation/common_voice_11_0",
        help="Dataset name to evaluate the `model_id`. Should be loadable with 🤗 Datasets",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config of the dataset. *E.g.* `'en'` for the English split of Common Voice",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'test'`",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--streaming",
        type=bool,
        default=True,
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Two letter language code for the transcription language, e.g. use 'en' for English.",
    )
    parser.add_argument("--task", type=str, default="transcribe", help="Task token")
    parser.add_argument("--outdir", type=str, required=True, help="Save path.")
    parser.add_argument("--log_outputs", action="store_true", help="If defined, write outputs to log file for analysis.")
    parser.add_argument("--id_column_name", type=str, default="ID")
    parser.add_argument("--fp16", action="store_true", help="Downcast model and data to fp16")

    args = parser.parse_args()

    main(args)
