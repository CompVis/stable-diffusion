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
from typing import List, Optional

import numpy
import onnx
import torch
from transformers import WhisperConfig
from whisper_decoder import WhisperDecoderInit
from whisper_encoder import WhisperEncoder, WhisperEncoderInputs

from onnxruntime import InferenceSession

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.t5.past_helper import PastKeyValuesHelper  # noqa: E402
from onnx_model import OnnxModel  # noqa: E402
from torch_onnx_export_helper import torch_onnx_export  # noqa: E402

logger = logging.getLogger(__name__)


class WhisperEncoderDecoderInit(torch.nn.Module):
    """A combination of WhisperEncoder and WhisperDecoderInit."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        lm_head: torch.nn.Module,
        config: WhisperConfig,
        decoder_start_token_id: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.whisper_encoder = WhisperEncoder(encoder, config)
        self.whisper_decoder_init = WhisperDecoderInit(decoder, lm_head, config, decoder_start_token_id)

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor = None,
    ):
        encoder_hidden_states: torch.FloatTensor = self.whisper_encoder(encoder_input_ids, None)
        # Decoder out: (logits, past_key_values, encoder_hidden_state)
        decinit_out = self.whisper_decoder_init(decoder_input_ids, encoder_attention_mask, encoder_hidden_states)
        present_self, present_cross = PastKeyValuesHelper.group_by_self_and_cross(decinit_out[1])
        present = present_self + present_cross
        return decinit_out[0], encoder_hidden_states, present


class WhisperEncoderDecoderInitInputs:
    def __init__(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids=None):
        self.encoder_input_ids: torch.LongTensor = encoder_input_ids
        self.encoder_attention_mask: torch.LongTensor = encoder_attention_mask
        self.decoder_input_ids: torch.LongTensor = decoder_input_ids

    @staticmethod
    def create_dummy(
        config: WhisperConfig,
        batch_size: int,
        encode_sequence_length: int,
        use_decoder_input_ids: int,
        device: torch.device,
        use_int32_inputs: bool = False,
    ):  # -> WhisperEncoderDecoderInitInputs:
        encoder_inputs: WhisperEncoderInputs = WhisperEncoderInputs.create_dummy(
            batch_size,
            sequence_length=3000,
            feature_size=config.num_mel_bins,
            device=device,
            use_int32_inputs=use_int32_inputs,
        )
        decoder_input_ids = None
        encoder_attention_mask = torch.zeros(
            (encoder_inputs.input_ids.shape[0], 1, encoder_inputs.input_ids.shape[1], encoder_inputs.input_ids.shape[1])
        ).type(torch.int8)
        if use_decoder_input_ids:
            dtype = torch.int32 if use_int32_inputs else torch.int64
            decoder_input_ids = torch.ones((batch_size, 1), dtype=dtype, device=device) * config.decoder_start_token_id

        return WhisperEncoderDecoderInitInputs(encoder_inputs.input_ids, encoder_attention_mask, decoder_input_ids)

    def to_list(self) -> List:
        input_list = [self.encoder_input_ids, self.encoder_attention_mask]
        if self.decoder_input_ids is not None:
            input_list.append(self.decoder_input_ids)
        return input_list


class WhisperEncoderDecoderInitHelper:
    @staticmethod
    def export_onnx(
        model: WhisperEncoderDecoderInit,
        device: torch.device,
        onnx_model_path: str,
        use_decoder_input_ids: bool = True,
        verbose: bool = True,
        use_external_data_format: bool = False,
        use_int32_inputs: bool = False,
    ):
        """Export decoder to ONNX

        Args:
            model (WhisperEncoderDecoderInit): the model to export
            device (torch.device): device of decoder object
            onnx_model_path (str): onnx path
            verbose (bool, optional): print verbose information. Defaults to True.
            use_external_data_format (bool, optional): use external data format or not. Defaults to False.
        """
        assert isinstance(model, WhisperEncoderDecoderInit)

        inputs = WhisperEncoderDecoderInitInputs.create_dummy(
            model.config,
            batch_size=2,
            encode_sequence_length=3000,
            use_decoder_input_ids=use_decoder_input_ids,
            device=device,
            use_int32_inputs=use_int32_inputs,
        )
        input_list = inputs.to_list()

        out = model(inputs.encoder_input_ids, inputs.encoder_attention_mask, inputs.decoder_input_ids)
        present = out[2]
        # pdb.set_trace()
        present_names = PastKeyValuesHelper.get_input_names(present, encoder=True)
        # present_names = PastKeyValuesHelper.get_past_names(model.config.num_layers, present=True)

        output_names = ["logits", "encoder_hidden_states", *present_names]

        # Shape of input tensors (sequence_length==1):
        #    input_ids: (batch_size, sequence_length)
        #    encoder_attention_mask: (batch_size, encode_sequence_length)
        #    encoder_hidden_states: (batch_size, encode_sequence_length, hidden_size)
        #    past_self_*: (batch_size, num_heads, past_decode_sequence_length, head_size)
        #    past_cross_*: (batch_size, num_heads, encode_sequence_length, head_size)

        # Shape of output tensors:
        #    logits: (batch_size, sequence_length, vocab_size)
        #    past_self_*: (batch_size, num_heads, past_decode_sequence_length + sequence_length, head_size)
        #    past_cross_*: (batch_size, num_heads, encode_sequence_length, head_size)

        input_names = ["encoder_input_ids", "encoder_attention_mask"]

        # ONNX exporter might mark dimension like 'Transposepresent_value_self_1_dim_2' in shape inference.
        # We use a workaround here: first use dim_param "1" for sequence_length, and later change to dim_value.
        sequence_length = "1"
        num_heads = str(model.config.encoder_attention_heads)
        hidden_size = str(model.config.d_model)
        head_size = str(model.config.d_model // model.config.encoder_attention_heads)
        dynamic_axes = {
            "encoder_input_ids": {0: "batch_size", 1: "encode_sequence_length"},
            "encoder_attention_mask": {0: "batch_size", 1: "encode_sequence_length"},
            "encoder_hidden_states": {
                0: "batch_size",
                1: "encode_sequence_length",
                2: hidden_size,
            },
            "logits": {
                0: "batch_size",
                1: sequence_length,
            },
        }

        if use_decoder_input_ids:
            input_names.append("decoder_input_ids")
            dynamic_axes["decoder_input_ids"] = {
                0: "batch_size",
                1: sequence_length,
            }

        for name in present_names:
            if "cross" in name:
                dynamic_axes[name] = {
                    0: "batch_size",
                    1: num_heads,
                    2: "encode_sequence_length",
                    3: head_size,
                }

            else:  # self attention past state
                dynamic_axes[name] = {
                    0: "batch_size",
                    1: num_heads,
                    2: sequence_length,
                    3: head_size,
                }

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            temp_onnx_model_path = os.path.join(tmp_dir_name, "encoder_decoder_init.onnx")
            Path(temp_onnx_model_path).parent.mkdir(parents=True, exist_ok=True)
            torch_onnx_export(
                model,
                args=tuple(input_list),
                f=temp_onnx_model_path,
                export_params=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=17,
                do_constant_folding=True,
                use_external_data_format=use_external_data_format,
                verbose=verbose,
            )

            # Workaround as mentioned earlier: change numeric dim_param to dim_value
            model = onnx.load(temp_onnx_model_path)
            for tensor in model.graph.output:
                for dim_proto in tensor.type.tensor_type.shape.dim:
                    if dim_proto.HasField("dim_param") and dim_proto.dim_param in [
                        sequence_length,
                        num_heads,
                        hidden_size,
                        head_size,
                    ]:
                        dim_value = int(dim_proto.dim_param)
                        dim_proto.Clear()
                        dim_proto.dim_value = dim_value

            OnnxModel.save(
                model,
                onnx_model_path,
                save_as_external_data=use_external_data_format,
                all_tensors_to_one_file=True,
            )

    @staticmethod
    def onnxruntime_inference(ort_session, inputs: WhisperEncoderDecoderInitInputs):
        """Run inference of ONNX model."""
        logger.debug("start onnxruntime_inference")

        ort_inputs = {
            "encoder_input_ids": numpy.ascontiguousarray(inputs.encoder_input_ids.cpu().numpy()),
            "encoder_attention_mask": numpy.ascontiguousarray(inputs.encoder_attention_mask.cpu().numpy()),
        }
        if inputs.decoder_input_ids is not None:
            ort_inputs["decoder_input_ids"] = numpy.ascontiguousarray(inputs.decoder_input_ids.cpu().numpy())

        ort_outputs = ort_session.run(None, ort_inputs)
        return ort_outputs

    @staticmethod
    def verify_onnx(
        model: WhisperEncoderDecoderInit,
        ort_session: InferenceSession,
        device: torch.device,
        use_int32_inputs: bool,
        max_cases: int = 4,
    ):
        """Compare the result from PyTorch and OnnxRuntime to verify the ONNX model is good."""
        ort_inputs = ort_session.get_inputs()
        use_decoder_input_ids = len(ort_inputs) == 3

        test_cases = [(4, 11), (1, 2), (3, 1), (8, 5)]
        test_cases_max_diff = []
        for batch_size, encode_sequence_length in test_cases[:max_cases]:
            inputs = WhisperEncoderDecoderInitInputs.create_dummy(
                model.config,
                batch_size,
                encode_sequence_length,
                use_decoder_input_ids=use_decoder_input_ids,
                device=device,
                use_int32_inputs=use_int32_inputs,
            )

            ort_outputs = WhisperEncoderDecoderInitHelper.onnxruntime_inference(ort_session, inputs)

            # Run inference of PyTorch model
            input_list = inputs.to_list()
            torch_outputs = model(*input_list)

            assert torch_outputs[0].cpu().numpy().shape == ort_outputs[0].shape
            max_diff = numpy.amax(numpy.abs(torch_outputs[0].cpu().numpy() - ort_outputs[0]))
            logger.debug(f"logits max_diff={max_diff}")
            max_diff_all = max_diff

            assert torch_outputs[1].cpu().numpy().shape == ort_outputs[1].shape
            max_diff = numpy.amax(numpy.abs(torch_outputs[1].cpu().numpy() - ort_outputs[1]))
            logger.debug(f"encoder_hidden_states max_diff={max_diff}")
            max_diff_all = max(max_diff_all, max_diff)

            for i in range(2 * model.config.num_layers):
                max_diff = numpy.amax(numpy.abs(torch_outputs[2][i].cpu().numpy() - ort_outputs[2 + i]))
                logger.debug(f"self attention past state {i} max_diff={max_diff}")

            for i in range(2 * model.config.num_layers):
                max_diff = numpy.amax(
                    numpy.abs(torch_outputs[3][i].cpu().numpy() - ort_outputs[2 + 2 * model.config.num_layers + i])
                )
                logger.debug(f"cross attention past state {i} max_diff={max_diff}")
                max_diff_all = max(max_diff_all, max_diff)

            test_cases_max_diff.append(max_diff_all)
            logger.info(
                f"batch_size={batch_size} encode_sequence_length={encode_sequence_length}, max_diff={max_diff_all}"
            )

        return max(test_cases_max_diff)
