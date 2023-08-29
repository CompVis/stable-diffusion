# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy
import onnx
import torch
from transformers import WhisperConfig

from onnxruntime import InferenceSession

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from onnx_model import OnnxModel  # noqa: E402
from torch_onnx_export_helper import torch_onnx_export  # noqa: E402

logger = logging.getLogger(__name__)


class WhisperEncoder(torch.nn.Module):
    """Whisper encoder outputs only the last hidden state"""

    def __init__(self, encoder, config: WhisperConfig):
        super().__init__()
        self.encoder = encoder
        self.config = config

    def forward(self, input_features, attention_mask):
        return self.encoder.model.encoder(input_features)[0]


class WhisperEncoderInputs:
    def __init__(self, input_features, attention_mask):
        self.input_ids: torch.LongTensor = input_features
        # HF Whisper model doesn't support Attention Mask functionality

    @staticmethod
    def create_dummy(
        batch_size: int, sequence_length: int, feature_size: int, device: torch.device, use_int32_inputs: bool
    ):
        """Create dummy inputs for Whisper encoder.

        Args:
            batch_size (int): batch size
            sequence_length (int): sequence length
            feature_size (int): feature size for spectrogram input
            device (torch.device): device of output tensors

        Returns:
            WhisperEncoderInputs: dummy inputs for encoder
        """
        dtype = torch.float32

        input_features = torch.randn(
            size=(batch_size, feature_size, sequence_length),
            device=device,
        )
        attention_mask = torch.ones([batch_size, feature_size, sequence_length], dtype=dtype, device=device)
        return WhisperEncoderInputs(input_features, attention_mask)

    def to_list(self) -> List:
        if self.input_features is None:
            return []
        return [self.input_features]


class WhisperEncoderHelper:
    @staticmethod
    def export_onnx(
        encoder,
        device: torch.device,
        onnx_model_path: str,
        verbose: bool = True,
        use_external_data_format: bool = False,
    ):
        """Export encoder to ONNX

        Args:
            encoder (WhisperEncoder): encoder object
            device (torch.device): device of encoder object
            onnx_model_path (str): onnx path
            verbose (bool, optional): print verbose information. Defaults to True.
            use_external_data_format (bool, optional): use external data format or not. Defaults to False.
        """
        config = encoder.config
        encoder_inputs = WhisperEncoderInputs.create_dummy(
            batch_size=2,
            sequence_length=3000,
            feature_size=config.num_mel_bins,
            device=device,
        )

        Path(onnx_model_path).parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            temp_onnx_model_path = os.path.join(tmp_dir_name, "encoder.onnx")
            Path(temp_onnx_model_path).parent.mkdir(parents=True, exist_ok=True)
            torch_onnx_export(
                encoder,
                args=tuple(encoder_inputs.to_list()),
                f=temp_onnx_model_path if use_external_data_format else onnx_model_path,
                export_params=True,
                input_names=["input_features"],
                output_names=["hidden_states"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "feature_size", 2: "sequence_length"},
                    "hidden_states": {0: "batch_size", 1: "sequence_length"},
                },
                opset_version=17,
                do_constant_folding=True,
                use_external_data_format=use_external_data_format,
                verbose=verbose,
            )

            if use_external_data_format:
                model = onnx.load_model(temp_onnx_model_path, load_external_data=True)
                OnnxModel.save(
                    model,
                    onnx_model_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                )

    @staticmethod
    def onnxruntime_inference(ort_session, inputs: WhisperEncoderInputs):
        """Run inference of ONNX model."""
        ort_inputs = {
            "input_ids": numpy.ascontiguousarray(inputs.input_ids.cpu().numpy()),
            "attention_mask": numpy.ascontiguousarray(inputs.attention_mask.cpu().numpy()),
        }

        return ort_session.run(None, ort_inputs)

    @staticmethod
    def verify_onnx(
        model: WhisperEncoder, ort_session: InferenceSession, device: torch.device, use_int32_inputs: bool = False
    ):
        """Compare the result from PyTorch and OnnxRuntime to verify the ONNX model is good."""
        inputs = WhisperEncoderInputs.create_dummy(
            batch_size=4,
            sequence_length=11,
            device=device,
            use_int32_inputs=use_int32_inputs,
        )
        input_list = inputs.to_list()
        torch_outputs = model(*input_list)

        ort_outputs = WhisperEncoderHelper.onnxruntime_inference(ort_session, inputs)

        max_diff = numpy.amax(numpy.abs(torch_outputs.cpu().numpy() - ort_outputs[0]))

        logger.info(f"max_diff={max_diff}")

        return max_diff
