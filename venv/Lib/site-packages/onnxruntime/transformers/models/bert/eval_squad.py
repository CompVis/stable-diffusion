# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
#
# This script evaluates accuracy of ONNX models for question-answering task on SQuAD data set.
# Example to evaluate raw and optimized model for CUDA in Linux:
#   pip3 install datasets evaluate optimum transformers onnxruntime-gpu
#   python3 eval_squad.py -m distilbert-base-cased-distilled-squad
#   python3 -m onnxruntime.transformers.optimizer --output optimized_fp16.onnx --num_heads 12 --hidden_size 768 \
#           --input /home/$USER/.cache/huggingface/hub/distilbert-base-cased-distilled-squad/model.onnx \
#           --use_mask_index --float16
#   python3 eval_squad.py -m distilbert-base-cased-distilled-squad --onnx optimized_fp16.onnx

import argparse
import csv
import os

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from evaluate import evaluator
from optimum.onnxruntime import ORTModelForQuestionAnswering
from optimum.onnxruntime.modeling_ort import ORTModel
from transformers import AutoTokenizer, pipeline

PRETRAINED_SQUAD_MODELS = [
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "deepset/roberta-base-squad2",
    "distilbert-base-cased-distilled-squad",
]


def get_package_version(package_name: str):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def load_onnx_model(
    model_id: str, onnx_path: Optional[str] = None, provider="CUDAExecutionProvider", use_io_binding: bool = False
):
    """Load onnx model given pretrained model name and optional ONNX model path. If onnx_path is None,
    the default onnx model from optimum will be used.

    Args:
        model_id (str): pretrained model name or checkpoint path
        onnx_path (Optional[str], optional): path of onnx model to evaluate. Defaults to None.

    Returns:
        model: ORTModel for the onnx model
        onnx_path: the path of onnx model
    """
    model = ORTModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)

    if onnx_path is not None:
        model.latest_model_name = Path(onnx_path).name

        if provider != "CPUExecutionProvider":
            model.device = torch.device("cuda:0")
            model.model = ORTModel.load_model(onnx_path, provider)
        else:
            model.device = torch.device("cpu")
            model.model = ORTModel.load_model(onnx_path)
    else:
        onnx_path = os.path.join(model.model_save_dir.as_posix(), model.latest_model_name)
        if provider != "CPUExecutionProvider":
            model.to("cuda")

    model.use_io_binding = use_io_binding

    return model, onnx_path


def output_details(results: List[Dict[str, Any]], csv_filename: str):
    """Output a CSV file with detail of each test results.

    Args:
        results (List[Dict[str, Any]]): list of JSON results.
        csv_filename (str): path of output CSV file
    """
    with open(csv_filename, mode="a", newline="", encoding="ascii") as csv_file:
        column_names = [
            "pretrained_model_name",
            "onnx_path",
            "provider",
            "disable_fused_attention",
            "batch_size",
            "sequence_length",
            "use_io_binding",
            "exact",
            "f1",
            "total",
            "HasAns_exact",
            "HasAns_f1",
            "HasAns_total",
            "best_exact",
            "best_exact_thresh",
            "best_f1",
            "best_f1_thresh",
            "total_time_in_seconds",
            "samples_per_second",
            "latency_in_seconds",
        ]

        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()
        for result in results:
            csv_writer.writerow(result)

        csv_file.flush()

    print(f"Detail results are saved to csv file: {csv_filename}")


def output_summary(results: List[Dict[str, Any]], csv_filename: str, metric_name: str):
    """Output a CSV file with summary of a metric on combinations of batch_size and sequence_length.

    Args:
        results (List[Dict[str, Any]]): list of JSON results.
        csv_filename (str): path of output CSV file
        metric_name (str): the metric to summarize
    """
    with open(csv_filename, mode="a", newline="", encoding="ascii") as csv_file:
        header_names = [
            "pretrained_model_name",
            "onnx_path",
            "provider",
            "disable_fused_attention",
            "use_io_binding",
        ]

        model_list = list({result["onnx_path"] for result in results})
        model_list.sort()

        batch_sizes = list({result["batch_size"] for result in results})
        batch_sizes.sort()

        sequence_lengths = list({result["sequence_length"] for result in results})
        sequence_lengths.sort()

        key_names = []
        for sequence_length in sequence_lengths:
            for batch_size in batch_sizes:
                key_names.append(f"b{batch_size}_s{sequence_length}")

        csv_writer = csv.DictWriter(csv_file, fieldnames=header_names + key_names)
        csv_writer.writeheader()

        for model in model_list:
            row = {}

            # Metric value for given pair of batch_size and sequence_length.
            # Assume that (onnx_path, batch_size and sequence_length) are unique so keep first occurrence only.
            values = {}
            values.update({k: "" for k in key_names})

            for result in results:
                if result["onnx_path"] == model and result[metric_name]:
                    headers = {k: v for k, v in result.items() if k in header_names}
                    if not row:
                        row.update(headers)

                    batch_size = result["batch_size"]
                    sequence_length = result["sequence_length"]
                    key = f"b{batch_size}_s{sequence_length}"

                    if key in key_names:
                        values[key] = result[metric_name]

            if row:
                for key in key_names:
                    row[key] = values[key] if key in values else ""
                csv_writer.writerow(row)

        csv_file.flush()

    print(f"Summary results for {metric_name} are saved to csv file: {csv_filename}")


def main():
    args = parse_arguments()
    print(args)

    for name in ["onnxruntime-gpu", "onnxruntime", "onnx", "torch", "transformers", "optimum", "datasets", "evaluate"]:
        package_version = get_package_version(name)
        if package_version:
            print(f"{name} version", package_version)

    pretrained_model_name = args.model_name
    if args.onnx and not os.path.exists(args.onnx):
        raise RuntimeError(f"Onnx model path does not exist: {args.onnx}")

    disable_fused_attention = os.environ.get("ORT_DISABLE_FUSED_ATTENTION", "0") == "1"

    all_results = []
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    for sequence_length in args.sequence_lengths:
        tokenizer.model_max_length = sequence_length
        tokenizer.doc_stride = min(sequence_length // 2, 128)
        ort_model, onnx_path = load_onnx_model(pretrained_model_name, args.onnx, args.provider, args.use_io_binding)

        print(ort_model.config)
        if sequence_length > ort_model.config.max_position_embeddings:
            raise RuntimeError("sequence length should not be larger than {ort_model.config.max_position_embeddings}")

        qa_pipeline = pipeline(
            "question-answering", model=ort_model, tokenizer=tokenizer, question_first=True, batch_size=args.batch_size
        )

        task_evaluator = evaluator("question-answering")
        squad_dataset = load_dataset("squad", split=f"validation[:{args.total}]" if args.total > 0 else "validation")

        result = task_evaluator.compute(
            model_or_pipeline=qa_pipeline,
            data=squad_dataset,
            metric="squad_v2",
            squad_v2_format=True,
        )

        result["provider"] = args.provider
        result["disable_fused_attention"] = disable_fused_attention
        result["pretrained_model_name"] = pretrained_model_name
        result["onnx_path"] = onnx_path
        result["batch_size"] = args.batch_size
        result["sequence_length"] = sequence_length
        result["use_io_binding"] = args.use_io_binding
        print(result)

        all_results.append(result)

    output_details(all_results, "detail.csv")

    for metric_name in ["f1", "exact", "samples_per_second"]:
        output_summary(all_results, f"{metric_name}.csv", metric_name)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_name",
        required=False,
        type=str,
        default=PRETRAINED_SQUAD_MODELS[0],
        help=f"Checkpoint directory or pre-trained model names in the list: {PRETRAINED_SQUAD_MODELS}",
    )

    parser.add_argument(
        "-s",
        "--sequence_lengths",
        nargs="+",
        type=int,
        default=[384],
        help="Sequence lengths for onnx model inputs. It could have multiple values.",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="batch size for inference.",
    )

    parser.add_argument("-t", "--total", type=int, default=0, help="Total samples to test. 0 means all samples.")

    parser.add_argument(
        "--onnx",
        required=False,
        type=str,
        default=None,
        help="Optional onnx model path. If not specified, optimum will be used to export onnx model for testing.",
    )

    parser.add_argument(
        "--provider",
        required=False,
        default="CUDAExecutionProvider",
        help="Select which Execution Provider to use for runs. Default is CUDAExecutionProvider.",
    )

    parser.add_argument("--use_io_binding", required=False, action="store_true", help="Use IO Binding for GPU.")
    parser.set_defaults(use_io_binding=False)

    args = parser.parse_args(argv)

    return args


if __name__ == "__main__":
    main()
