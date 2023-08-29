# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
#
# This script converts stable diffusion onnx models from float to half (mixed) precision for GPU inference.
#
# Before running this script, follow README.md to setup python environment and convert stable diffusion checkpoint to float32 onnx models.
#
# For example, the float32 ONNX pipeline is saved to ./sd-v1-5 directory, you can optimize and convert it to float16 like the following:
#    python optimize_pipeline.py -i ./sd-v1-5 -o ./sd-v1-5-fp16 --float16
#
# Note that the optimizations are carried out for CUDA Execution Provider at first, other EPs may not have the support for the fused opeartors.
# In this case, the users should disable the operator fusion manually to workaround.
#
# Stable diffusion 2.1 model will get black images using float16 Attention. A walkaround is to force Attention to run in float32 like the following:
#    python optimize_pipeline.py -i ./sd-v2-1 -o ./sd-v2-1-fp16 --float16 --force_fp32_ops unet:Attention
#
# If you are using nightly package (or built from source), you can force MultiHeadAttention to run in float32:
#    python optimize_pipeline.py -i ./sd-v2-1 -o ./sd-v2-1-fp16 --float16 --force_fp32_ops unet:MultiHeadAttention

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

import coloredlogs
import onnx
from packaging import version

import onnxruntime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from fusion_options import FusionOptions  # noqa: E402
from onnx_model_clip import ClipOnnxModel  # noqa: E402
from onnx_model_unet import UnetOnnxModel  # noqa: E402
from onnx_model_vae import VaeOnnxModel  # noqa: E402
from optimizer import optimize_by_onnxruntime, optimize_model  # noqa: E402

logger = logging.getLogger(__name__)


def optimize_sd_pipeline(
    source_dir: Path,
    target_dir: Path,
    overwrite: bool,
    use_external_data_format: bool,
    float16: bool,
    force_fp32_ops: List[str],
    enable_runtime_optimization: bool,
    args,
):
    """Optimize onnx models used in stable diffusion onnx pipeline and optionally convert to float16.

    Args:
        source_dir (Path): Root of input directory of stable diffusion onnx pipeline with float32 models.
        target_dir (Path): Root of output directory of stable diffusion onnx pipeline with optimized models.
        overwrite (bool): Overwrite files if exists.
        use_external_data_format (bool): save onnx model to two files: one for onnx graph, another for weights
        float16 (bool): use half precision
        force_fp32_ops(List[str]): operators that are forced to run in float32.
        enable_runtime_optimization(bool): run graph optimization using Onnx Runtime.

    Raises:
        RuntimeError: input onnx model does not exist
        RuntimeError: output onnx model path existed
    """
    model_type_mapping = {
        "unet": "unet",
        "vae_encoder": "vae",
        "vae_decoder": "vae",
        "text_encoder": "clip",
        "safety_checker": "unet",
    }

    model_type_class_mapping = {
        "unet": UnetOnnxModel,
        "vae": VaeOnnxModel,
        "clip": ClipOnnxModel,
    }

    force_fp32_operators = {
        "unet": [],
        "vae_encoder": [],
        "vae_decoder": [],
        "text_encoder": [],
        "safety_checker": [],
    }

    if force_fp32_ops:
        for fp32_operator in force_fp32_ops:
            parts = fp32_operator.split(":")
            if len(parts) == 2 and parts[0] in force_fp32_operators and (parts[1] and parts[1][0].isupper()):
                force_fp32_operators[parts[0]].append(parts[1])
            else:
                raise ValueError(
                    f"--force_fp32_ops shall be in the format of module:operator like unet:Attention, got {fp32_operator}"
                )

    for name, model_type in model_type_mapping.items():
        onnx_model_path = source_dir / name / "model.onnx"

        if not os.path.exists(onnx_model_path):
            message = f"input onnx model does not exist: {onnx_model_path}."
            if name not in ["safety_checker"]:
                raise RuntimeError(message)
            continue

        # Prepare output directory
        optimized_model_path = target_dir / name / "model.onnx"
        output_dir = optimized_model_path.parent
        if optimized_model_path.exists():
            if not overwrite:
                raise RuntimeError(f"output onnx model path existed: {optimized_model_path}")

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Graph fusion before fp16 conversion, otherwise they cannot be fused later.
        # Right now, onnxruntime does not save >2GB model so we use script to optimize unet instead.
        logger.info(f"Optimize {onnx_model_path}...")

        args.model_type = model_type
        fusion_options = FusionOptions.parse(args)

        if model_type in ["unet"]:
            # Some optimizations are not available in v1.14 or older version: packed QKV and BiasAdd
            has_all_optimizations = version.parse(onnxruntime.__version__) >= version.parse("1.15.0")
            fusion_options.enable_packed_kv = float16
            fusion_options.enable_packed_qkv = float16 and has_all_optimizations
            fusion_options.enable_bias_add = has_all_optimizations

        m = optimize_model(
            str(onnx_model_path),
            model_type=model_type,
            num_heads=0,  # will be deduced from graph
            hidden_size=0,  # will be deduced from graph
            opt_level=0,
            optimization_options=fusion_options,
            use_gpu=True,
        )

        if float16:
            logger.info("Convert %s to float16 ...", name)
            op_block_list = ["RandomNormalLike"]
            m.convert_float_to_float16(
                keep_io_types=False,
                op_block_list=op_block_list + force_fp32_operators[name],
            )

        if enable_runtime_optimization and (float16 or (name not in ["unet"])):
            # Use this step to see the final graph that executed by Onnx Runtime.
            # Note that ORT cannot save model larger than 2GB so we exclude unet float32 model.
            # This step is optional since it has no impact on performance except model loading time.
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Save to a temporary file so that we can load it with Onnx Runtime.
                logger.info("Saving a temporary model to run OnnxRuntime graph optimizations...")
                tmp_model_path = Path(tmp_dir) / "model.onnx"
                m.save_model_to_file(str(tmp_model_path))
                ort_optimized_model_path = tmp_model_path
                optimize_by_onnxruntime(
                    str(tmp_model_path), use_gpu=True, optimized_model_path=str(ort_optimized_model_path)
                )
                model = onnx.load(str(ort_optimized_model_path), load_external_data=True)
                m = model_type_class_mapping[model_type](model)

        m.get_operator_statistics()
        m.get_fused_operator_statistics()
        m.save_model_to_file(str(optimized_model_path), use_external_data_format=use_external_data_format)
        logger.info("%s is optimized", name)
        logger.info("*" * 20)


def copy_extra_directory(source_dir: Path, target_dir: Path, overwrite: bool):
    """Copy extra directory that does not have onnx model

    Args:
        source_dir (Path): source directory
        target_dir (Path): target directory
        overwrite (bool): overwrite if exists

    Raises:
        RuntimeError: source path does not exist
        RuntimeError: output path exists but overwrite is false.
    """
    extra_dirs = ["scheduler", "tokenizer", "feature_extractor"]

    for name in extra_dirs:
        source_path = source_dir / name

        if not os.path.exists(source_path):
            message = f"source path does not exist: {source_path}"
            if name not in ["feature_extractor"]:
                raise RuntimeError(message)
            continue

        target_path = target_dir / name
        if target_path.exists():
            if not overwrite:
                raise RuntimeError(f"output path existed: {target_path}")
            shutil.rmtree(target_path)

        shutil.copytree(source_path, target_path)
        logger.info("%s => %s", source_path, target_path)

    extra_files = ["model_index.json"]
    for name in extra_files:
        source_path = source_dir / name
        if not os.path.exists(source_path):
            raise RuntimeError(f"source path does not exist: {source_path}")

        target_path = target_dir / name
        if target_path.exists():
            if not overwrite:
                raise RuntimeError(f"output path existed: {target_path}")
            os.remove(target_path)
        shutil.copyfile(source_path, target_path)
        logger.info("%s => %s", source_path, target_path)


def parse_arguments():
    """Parse arguments

    Returns:
        Namespace: arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Root of input directory of stable diffusion onnx pipeline with float32 models.",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Root of output directory of stable diffusion onnx pipeline with optimized models.",
    )

    parser.add_argument(
        "--float16",
        required=False,
        action="store_true",
        help="Output models of half or mixed precision.",
    )
    parser.set_defaults(float16=False)

    parser.add_argument(
        "--force_fp32_ops",
        required=False,
        nargs="+",
        type=str,
        help="Force given operators (like unet:Attention) to run in float32. It is case sensitive!",
    )

    parser.add_argument(
        "--inspect",
        required=False,
        action="store_true",
        help="Inspect the optimized graph from Onnx Runtime for debugging purpose. This option has no impact on model performance.",
    )
    parser.set_defaults(inspect=False)

    parser.add_argument(
        "--overwrite",
        required=False,
        action="store_true",
        help="Overwrite exists files.",
    )
    parser.set_defaults(overwrite=False)

    parser.add_argument(
        "-e",
        "--use_external_data_format",
        required=False,
        action="store_true",
        help="Onnx model larger than 2GB need to use external data format. "
        "Save onnx model to two files: one for onnx graph, another for large weights.",
    )
    parser.set_defaults(use_external_data_format=False)

    FusionOptions.add_arguments(parser)

    args = parser.parse_args()
    return args


def main():
    coloredlogs.install(fmt="%(funcName)20s: %(message)s")
    args = parse_arguments()
    logger.info("Arguments: %s", str(args))
    copy_extra_directory(Path(args.input), Path(args.output), args.overwrite)
    optimize_sd_pipeline(
        Path(args.input),
        Path(args.output),
        args.overwrite,
        args.use_external_data_format,
        args.float16,
        args.force_fp32_ops,
        args.inspect,
        args,
    )


main()
