import os
import sys

import onnx
from onnx import TensorProto, helper
from transformers import WhisperConfig

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from convert_generation import get_shared_initializers  # noqa: E402


def add_attention_mask(model):
    # Add attention mask - required by BeamSearch but unused in Pytorch
    mask = helper.make_tensor_value_info(
        "encoder_attention_mask", TensorProto.INT32, shape=["batch", "feature_size", "sequence"]
    )
    model.graph.input.insert(1, mask)


def chain_model(args):
    # Load encoder/decoder and insert necessary (but unused) graph inputs expected by BeamSearch op
    encoder_model = onnx.load_model(args.encoder_path, load_external_data=True)
    encoder_model.graph.name = "encoderdecoderinit subgraph"
    add_attention_mask(encoder_model)

    decoder_model = onnx.load_model(args.decoder_path, load_external_data=True)
    decoder_model.graph.name = "decoder subgraph"
    add_attention_mask(decoder_model)

    config = WhisperConfig.from_pretrained(args.model_name_or_path)

    beam_inputs = [
        "input_features",
        "max_length",
        "min_length",
        "num_beams",
        "num_return_sequences",
        "length_penalty",
        "repetition_penalty",
        "",
        "",
        "attention_mask",
    ]
    beam_outputs = ["sequences"]

    node = helper.make_node("BeamSearch", inputs=beam_inputs, outputs=beam_outputs, name="BeamSearch_zcode")
    node.domain = "com.microsoft"
    node.attribute.extend(
        [
            helper.make_attribute("eos_token_id", config.eos_token_id),
            helper.make_attribute("pad_token_id", config.pad_token_id),
            helper.make_attribute("decoder_start_token_id", config.decoder_start_token_id),
            helper.make_attribute("no_repeat_ngram_size", args.no_repeat_ngram_size),
            helper.make_attribute("early_stopping", True),
            helper.make_attribute("model_type", 2),
        ]
    )

    # beam graph inputs
    input_features = helper.make_tensor_value_info(
        "input_features", TensorProto.FLOAT, ["batch_size", "feature_size", "sequence_length"]
    )
    max_length = helper.make_tensor_value_info("max_length", TensorProto.INT32, [1])
    min_length = helper.make_tensor_value_info("min_length", TensorProto.INT32, [1])
    num_beams = helper.make_tensor_value_info("num_beams", TensorProto.INT32, [1])
    num_return_sequences = helper.make_tensor_value_info("num_return_sequences", TensorProto.INT32, [1])
    length_penalty = helper.make_tensor_value_info("length_penalty", TensorProto.FLOAT, [1])
    repetition_penalty = helper.make_tensor_value_info("repetition_penalty", TensorProto.FLOAT, [1])
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT32, ["batch_size", "feature_size", "sequence_length"]
    )

    graph_inputs = [
        input_features,
        max_length,
        min_length,
        num_beams,
        num_return_sequences,
        length_penalty,
        repetition_penalty,
        attention_mask,
    ]

    # graph outputs
    sequences = helper.make_tensor_value_info(
        "sequences", TensorProto.INT32, ["batch_size", "num_return_sequences", "max_length"]
    )
    graph_outputs = [sequences]

    # Initializers/opsets
    # Delete shared data between decoder/encoder and move to larger graph initializers
    initializers = get_shared_initializers(encoder_model, decoder_model)
    node.attribute.extend(
        [
            helper.make_attribute("decoder", decoder_model.graph),
            helper.make_attribute("encoder", encoder_model.graph),
        ]
    )
    opset_import = [helper.make_opsetid(domain="com.microsoft", version=1), helper.make_opsetid(domain="", version=17)]

    beam_graph = helper.make_graph([node], "beam-search-test", graph_inputs, graph_outputs, initializers)
    beam_model = helper.make_model(beam_graph, producer_name="pytorch", opset_imports=opset_import)

    onnx.save(
        beam_model,
        args.beam_model_output_dir,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        convert_attribute=True,
        location=f"{os.path.basename(args.beam_model_output_dir)}.data",
    )
    onnx.checker.check_model(args.beam_model_output_dir, full_check=True)
