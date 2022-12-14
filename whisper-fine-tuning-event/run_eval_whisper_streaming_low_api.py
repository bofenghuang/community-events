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
    # batch["norm_text"] = normalizer(get_text(batch))
    batch["target"] = normalizer(get_text(batch))
    return batch


def data(dataset):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": item["norm_text"]}

def log_results(result: Dataset, args: Dict[str, str]):
    """DO NOT CHANGE. This function computes and logs the result metrics."""

    log_outputs = args.log_outputs
    # dataset_id = "_".join(args.dataset.split("/") + [args.config, args.split])
    # dataset_id = Path(args.test_csv_file).stem

    # compute metrics
    # wer_result = wer.compute(references=result["target"], predictions=result["prediction"])
    # cer_result = cer.compute(references=result["target"], predictions=result["prediction"])
    wer_result = wer_metric.compute(references=result["target"], predictions=result["prediction_0"])
    cer_result = cer_metric.compute(references=result["target"], predictions=result["prediction_0"])

    # print & log results
    result_str = f"WER: {wer_result}\n" f"CER: {cer_result}"
    print(result_str)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # with open(f"{args.outdir}/{dataset_id}_eval_results.txt", "w") as f:
    with open(f"{args.outdir}/eval_results.txt", "w") as f:
        f.write(result_str)

    # log all results in text file. Possibly interesting for analysis
    if log_outputs is not None:
        # pred_file = f"{args.outdir}/log_{dataset_id}_predictions.txt"
        # target_file = f"{args.outdir}/log_{dataset_id}_targets.txt"
        pred_file = f"{args.outdir}/log_predictions.txt"
        target_file = f"{args.outdir}/log_targets.txt"

        with open(pred_file, "w") as p, open(target_file, "w") as t:

            # mapping function to write output
            # def write_to_file(batch, i):
            #     p.write(f"{i}" + "\n")
            #     p.write(batch["prediction"] + "\n")
            #     t.write(f"{i}" + "\n")
            #     t.write(batch["target"] + "\n")

            # ! to adjust for your cases
            def write_to_file(batch, i):
                # p.write(batch[args.id_column_name] + "\t" + str(batch[args.start_column_name]) + "\t" + str(batch[args.end_column_name]) + "\t" + batch["prediction"] + "\n")
                # t.write(batch[args.id_column_name] + "\t" + str(batch[args.start_column_name]) + "\t" + str(batch[args.end_column_name]) + "\t" + batch["target"] + "\n")
                # p.write(batch[args.id_column_name] + "\t" + batch["prediction"] + "\n")
                p.write(batch[args.id_column_name] + "\t" + "\t".join([batch[f"prediction_{nbest_i}"] for nbest_i in range(10)]) + "\n")
                t.write(batch[args.id_column_name] + "\t" + batch["target"] + "\n")

            result.map(write_to_file, with_indices=True)

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

    gen_greedy = {"do_sample": False, "num_beams": 1}
    gen_greedy_sampling = {"do_sample": True, "num_beams": 1}
    gen_greedy_sampling_nbest = {**gen_greedy_sampling, "return_dict_in_generate": True, "output_scores": True, "num_return_sequences": 10}
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

    gen_kwargs = {
        "max_new_tokens": 225,
        # "max_new_tokens": 40,
        # "repetition_penalty"
        # "length_penalty"
        # "no_repeat_ngram_size"
        # "bad_words_ids"
        # "num_return_sequences"
    }

    def map_to_pred(batch):
        # bh: synchronised process and forward, this can be improved by dataloader
        inputs = processor(
            [example["array"] for example in batch["audio"]], sampling_rate=16_000, return_tensors="pt"
        )
        input_features = inputs.input_features
        input_features = input_features.to(args.device)

        if args.fp16:
            input_features = input_features.half()

        # generated_ids = model.generate(inputs=input_features, **gen_kwargs)
        outputs_ = model.generate(inputs=input_features, **gen_kwargs)
        generated_ids = outputs_.sequences  # shape BS x NBEST

        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # batch["prediction"] = transcriptions
        # normalize prediction
        predictions = [normalizer(prediction) for prediction in transcriptions]

        nbest = gen_greedy_sampling_nbest["num_return_sequences"]
        for nbest_i in range(nbest):
            batch[f"prediction_{nbest_i}"] = [predictions[i + nbest_i] for i in range(0, len(predictions), nbest)]

        # batch["target"] = batch[args.text_column_name]
        # normalize target
        # batch["target"] = normalize_text(batch[args.text_column_name], invalid_chars_regex)
        # batch["target"] = [normalizer(target) for target in batch[args.text_column_name]]

        return batch

    result = dataset.map(map_to_pred, batched=True, batch_size=args.batch_size)

    # fake ID if not exists
    if args.id_column_name not in result.features.keys():
        result = result.map(lambda example, idx: {**example, args.id_column_name: f"{idx:09d}"}, with_indices=True)

    # filtering out empty targets
    # result = result.filter(lambda example: example["target"] != "")

    # compute and log_results
    # do not change function below
    log_results(result, args)

    # predictions = []
    # references = []

    # # run streamed inference
    # for out in pipe(data(dataset), batch_size=batch_size):
    #     predictions.append(normalizer(out["text"]))
    #     references.append(out["reference"][0])

    # wer = wer_metric.compute(references=references, predictions=predictions)
    # # wer = round(100 * wer, 2)

    # print("WER:", wer)


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
    parser.add_argument("--id_column_name", type=str, default="ID")
    parser.add_argument("--fp16", action="store_true", help="Downcast model and data to fp16")

    args = parser.parse_args()

    main(args)
