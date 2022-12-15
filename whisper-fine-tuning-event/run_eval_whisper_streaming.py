import argparse
import os

import evaluate
from datasets import Audio, load_dataset
from transformers import pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# from normalize_text_hf_sprint import FrenchTextNormalizer

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

normalizer = BasicTextNormalizer()
# normalizer = FrenchTextNormalizer()


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
    batch_size = args.batch_size
    # todo: add fp16
    pipe = pipeline(
        "automatic-speech-recognition", model=args.model_id, device=args.device
    )

    pipe.model.config.forced_decoder_ids = (
        pipe.tokenizer.get_decoder_prompt_ids(
            language=args.language, task="transcribe"
        )
    )

    # ! decoding option
    pipe.model.config.max_length = 225 + 1
    # beam search
    # pipe.model.config.beam = 5

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

    predictions = []
    references = []

    # run streamed inference
    for out in pipe(data(dataset), batch_size=batch_size):
        predictions.append(normalizer(out["text"]))
        references.append(out["target"][0])

    wer_result = wer_metric.compute(references=references, predictions=predictions)
    cer_result = cer_metric.compute(references=references, predictions=predictions)

    # print & log results
    result_str = f"WER: {wer_result}\n" f"CER: {cer_result}"
    print(result_str)

    if args.log_outputs is not None:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

        with open(f"{args.outdir}/eval_results.txt", "w") as f:
            f.write(result_str)

        pred_file = f"{args.outdir}/log_predictions.txt"
        target_file = f"{args.outdir}/log_targets.txt"

        with open(pred_file, "w") as p, open(target_file, "w") as t:            
            for idx, (reference, prediction) in enumerate(zip(references, predictions)):
                p.write(str(idx) + "\t" + prediction + "\n")
                t.write(str(idx) + "\t" + reference + "\n")


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
    parser.add_argument("--fp16", action="store_true", help="Downcast model and data to fp16")

    args = parser.parse_args()

    main(args)
