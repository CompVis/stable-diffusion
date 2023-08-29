# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import logging
import os
from typing import List, Union

import coloredlogs
from constants import AttentionInputIDs, AttentionOutputIDs, Operators
from onnx import helper, load_model
from onnx_model import NodeProto, OnnxModel
from shape_infer_helper import SymbolicShapeInferenceHelper

logger = logging.getLogger(__name__)


class PackingMode:
    def __init__(
        self,
        model: OnnxModel,
    ):
        self.model: OnnxModel = model
        self.nodes_to_remove: List = []
        self.nodes_to_add: List = []
        self.prune_graph: bool = False
        self.node_name_to_graph_name: dict = {}
        self.this_graph_name: str = self.model.model.graph.name
        self.attention_nodes = self.model.get_nodes_by_op_type(Operators.ATTENTION)

    def _try_getting_attention_mask(self) -> Union[str, None]:
        first_attention_node = self._try_getting_first_attention()
        # check if attention has mask
        if not first_attention_node or len(first_attention_node.input) <= AttentionInputIDs.MASK_INDEX:
            return None

        attention_mask = first_attention_node.input[AttentionInputIDs.MASK_INDEX]

        # check if all attention nodes have same mask
        for node in self.attention_nodes:
            if (
                len(node.input) <= AttentionInputIDs.MASK_INDEX
                or node.input[AttentionInputIDs.MASK_INDEX] != attention_mask
            ):
                return None

        return attention_mask

    def _try_getting_first_attention(self) -> Union[NodeProto, None]:
        if len(self.attention_nodes) <= 0:
            return None

        return self.attention_nodes[0]

    def _try_getting_last_layernorm(self) -> Union[NodeProto, None]:
        last_layernorm_node = None
        for node in self.model.nodes():
            if node.op_type == Operators.LAYERNORM or node.op_type == Operators.SKIPLAYERNORM:
                last_layernorm_node = node
        return last_layernorm_node

    def _are_attentions_supportted(self) -> bool:
        for node in self.attention_nodes:
            if OnnxModel.get_node_attribute(node, "past_present_share_buffer") is not None:
                return False
            if OnnxModel.get_node_attribute(node, "do_rotary") is not None:
                return False
            unidirection_attr = OnnxModel.get_node_attribute(node, "unidirectional")
            if unidirection_attr is not None and unidirection_attr != 0:
                return False
            if len(node.input) > AttentionInputIDs.PAST and not node.input[AttentionInputIDs.PAST]:
                return False
            if (
                len(node.input) > AttentionInputIDs.PAST_SEQUENCE_LENGTH
                and not node.input[AttentionInputIDs.PAST_SEQUENCE_LENGTH]
            ):
                return False
        return True

    def _insert_removepadding_node(self, inputs: List[str], outputs: List[str]) -> None:
        new_node = helper.make_node(
            Operators.REMOVEPADDING,
            inputs=inputs,
            outputs=outputs,
            name=self.model.create_node_name(Operators.REMOVEPADDING),
        )

        new_node.domain = "com.microsoft"
        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

    def _insert_restorepadding_node(self, inputs: List[str], outputs: List[str]) -> None:
        new_node = helper.make_node(
            Operators.RESTOREPADDING,
            inputs=inputs,
            outputs=outputs,
            name=self.model.create_node_name(Operators.RESTOREPADDING),
        )

        new_node.domain = "com.microsoft"
        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

    def _replace_attention_with_packing_attention(self, token_offset: str, cumulative_sequence_length: str) -> None:
        for attention in self.attention_nodes:
            packed_attention = helper.make_node(
                Operators.PACKEDATTENTION,
                inputs=[
                    attention.input[AttentionInputIDs.INPUT],
                    attention.input[AttentionInputIDs.WEIGHTS],
                    attention.input[AttentionInputIDs.BIAS],
                    token_offset,
                    cumulative_sequence_length,
                    attention.input[AttentionInputIDs.RELATIVE_POSITION_BIAS]
                    if len(attention.input) > AttentionInputIDs.RELATIVE_POSITION_BIAS
                    else "",
                ],
                outputs=[attention.output[AttentionOutputIDs.OUTPUT]],
                name=self.model.create_node_name(Operators.PACKEDATTENTION),
            )

            attributes = []
            for attr in attention.attribute:
                if attr.name in ["num_heads", "qkv_hidden_sizes", "scale"]:
                    attributes.append(attr)

            packed_attention.attribute.extend(attributes)
            packed_attention.domain = "com.microsoft"
            self.nodes_to_add.append(packed_attention)
            self.nodes_to_remove.append(attention)
            self.node_name_to_graph_name[packed_attention.name] = self.this_graph_name

    def convert(self, use_symbolic_shape_infer: bool = True) -> None:
        logger.debug("start converting to packing model...")
        if not self._are_attentions_supportted():
            return

        attention_mask = self._try_getting_attention_mask()
        if not attention_mask:
            return

        first_attention_node = self._try_getting_first_attention()
        last_layernorm_node = self._try_getting_last_layernorm()
        if not last_layernorm_node:
            return

        # insert RemovePadding
        first_attention_input = first_attention_node.input[AttentionInputIDs.INPUT]
        input_to_remove_padding = first_attention_input
        output_without_padding = first_attention_input + "_no_padding"
        token_offset = first_attention_input + "_token_offset"
        cumulated_seq_len = first_attention_input + "_cumulated_seq_len"
        max_seq_len = first_attention_input + "_max_seq_len"
        self._insert_removepadding_node(
            [input_to_remove_padding, attention_mask],
            [output_without_padding, token_offset, cumulated_seq_len, max_seq_len],
        )
        self.model.replace_input_of_all_nodes(input_to_remove_padding, output_without_padding)
        logger.debug("inserted RemovePadding before Attention")

        # insert RestorePadding
        restorepadding_input = last_layernorm_node.output[0] + "_restore_input"
        self._insert_restorepadding_node([restorepadding_input, token_offset], [last_layernorm_node.output[0]])
        self.model.replace_output_of_all_nodes(last_layernorm_node.output[0], restorepadding_input)
        logger.debug(f"inserted RestorePadding after last {last_layernorm_node.op_type} layer")

        # insert PackingAttention
        self._replace_attention_with_packing_attention(token_offset, cumulated_seq_len)
        logger.debug("replaced Attention with PackedAttention")

        self.model.remove_nodes(self.nodes_to_remove)
        self.model.add_nodes(self.nodes_to_add, self.node_name_to_graph_name)

        if self.prune_graph:
            self.model.prune_graph()
        elif self.nodes_to_remove or self.nodes_to_add:
            self.model.update_graph()
        self.model.clean_shape_infer()
        if use_symbolic_shape_infer:
            # Use symbolic shape inference since custom operators (like Gelu, SkipLayerNormalization etc)
            # are not recognized by onnx shape inference.
            shape_infer_helper = SymbolicShapeInferenceHelper(self.model.model, verbose=0)
            inferred_model = shape_infer_helper.infer_shapes(self.model.model, auto_merge=True, guess_output_rank=False)
            if inferred_model:
                self.model.model = inferred_model


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert to packing mode tool for ONNX Runtime. It converts BERT like model to use packing mode."
    )
    parser.add_argument("--input", required=True, type=str, help="input onnx model path")

    parser.add_argument("--output", required=True, type=str, help="optimized onnx model path")

    parser.add_argument("--verbose", required=False, action="store_true", help="show debug information.")
    parser.set_defaults(verbose=False)

    parser.add_argument(
        "--use_external_data_format",
        required=False,
        action="store_true",
        help="use external data format to store large model (>2GB)",
    )
    parser.set_defaults(use_external_data_format=False)

    args = parser.parse_args()

    return args


def _setup_logger(verbose):
    if verbose:
        coloredlogs.install(
            level="DEBUG",
            fmt="[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s",
        )
    else:
        coloredlogs.install(fmt="%(funcName)20s: %(message)s")


def main():
    args = _parse_arguments()

    _setup_logger(args.verbose)

    logger.debug("arguments:{args}")

    if os.path.realpath(args.input) == os.path.realpath(args.output):
        logger.warning("Specified the same input and output path. Note that this may overwrite the original model")

    model = load_model(args.input)
    packing_mode = PackingMode(OnnxModel(model))
    packing_mode.convert()
    packing_mode.model.save_model_to_file(args.output, use_external_data_format=args.use_external_data_format)


if __name__ == "__main__":
    main()
