# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

from fusion_attention import AttentionMask, FusionAttention
from onnx import TensorProto, helper
from onnx_model import OnnxModel

logger = logging.getLogger(__name__)


class FusionBartAttention(FusionAttention):
    """
    Fuse Bart Attention subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
        attention_mask: AttentionMask,
    ):
        super().__init__(model, hidden_size, num_heads, attention_mask)

    def check_runtime_shape_path(
        self,
        reshape_qkv_2,
        reshape_qkv_1,
        reshape_q_2,
        reshape_k_2,
        reshape_v_2,
        root_input,
    ):
        concat_qkv_2_path = self.model.match_parent_path(reshape_qkv_2, ["Concat"], [1])
        if concat_qkv_2_path is None:
            return False
        concat_qkv_2 = concat_qkv_2_path[0]

        reshape_qkv_2_path_1 = self.model.match_parent_path(concat_qkv_2, ["Unsqueeze", "Gather", "Shape"], [0, 0, 0])
        reshape_qkv_2_path_2 = self.model.match_parent_path(concat_qkv_2, ["Unsqueeze", "Gather", "Shape"], [1, 0, 0])
        if reshape_qkv_2_path_1 is None or reshape_qkv_2_path_2 is None:
            return False

        _, gather_1, shape_1 = reshape_qkv_2_path_1
        _, gather_2, shape_2 = reshape_qkv_2_path_2

        if shape_1.input[0] != root_input or shape_2.input[0] != root_input:
            return False

        reshape_qkv_1_path_1 = self.model.match_parent_path(reshape_qkv_1, ["Concat", "Unsqueeze", "Gather"], [1, 0, 0])
        reshape_qkv_1_path_2 = self.model.match_parent_path(reshape_qkv_1, ["Concat", "Unsqueeze", "Gather"], [1, 2, 0])
        if reshape_qkv_1_path_1 is None or reshape_qkv_1_path_2 is None:
            return False
        if reshape_qkv_1_path_1[-1].name != gather_1.name or reshape_qkv_1_path_2[-1].name != gather_2.name:
            return False

        reshape_q_2_path = self.model.match_parent_path(reshape_q_2, ["Concat", "Unsqueeze", "Mul"], [1, 0, 0])
        reshape_k_2_path = self.model.match_parent_path(reshape_k_2, ["Concat", "Unsqueeze", "Mul"], [1, 0, 0])
        reshape_v_2_path = self.model.match_parent_path(reshape_v_2, ["Concat", "Unsqueeze", "Mul"], [1, 0, 0])
        if reshape_q_2_path is None or reshape_k_2_path is None or reshape_v_2_path is None:
            return False

        mul_q = reshape_q_2_path[-1]
        mul_k = reshape_k_2_path[-1]
        mul_v = reshape_v_2_path[-1]

        gather_1_out = gather_1.output[0]
        if mul_q.input[0] != gather_1_out or mul_k.input[0] != gather_1_out or mul_v.input[0] != gather_1_out:
            return False

        return True

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
        qkv_nodes = self.model.match_parent_path(
            normalize_node,
            ["Add", "MatMul", "Reshape", "Transpose", "Reshape", "MatMul"],
            [1, 1, 0, 0, 0, 0],
        )
        if qkv_nodes is not None:
            (
                add_out,
                matmul_out,
                reshape_qkv_2,
                transpose_qkv,
                reshape_qkv_1,
                matmul_qkv,
            ) = qkv_nodes
        else:
            return

        other_inputs = []
        for input in normalize_node.input:
            if input not in output_name_to_node:
                continue
            if input == qkv_nodes[0].output[0]:
                continue
            other_inputs.append(input)
        if len(other_inputs) != 1:
            return
        root_input = other_inputs[0]

        # Sometimes the input name to the attention MatMul nodes does not match the input name to the end
        # SkipLayerNormalization node (name saved in root_input). We find the true input name to the MatMul
        # nodes by getting the initial SkipLayerNormalization node and checking how many MatMul nodes are
        # children nodes for each of its output names.
        """
                                        root_input
                    +---------------------------------------------------+
                    |                                                   |
                    |                                                   |
        SkipLayerNormalization --> Attention --> MatMul --> SkipLayerNormalization
        """
        skip_layernorm = output_name_to_node[root_input]
        # For some attention blocks, the end SkipLayerNormalization node may point to an Add node whose
        # child is the LayerNormalization node.
        if skip_layernorm.op_type == "Add":
            skip_layernorm = self.model.get_children(skip_layernorm)[0]
        for output in skip_layernorm.output:
            if not output:
                continue
            children = input_name_to_nodes[output]
            children_types = [child.op_type for child in children]
            if children_types.count("MatMul") >= 1:
                root_input = output
                break

        graph_input_names = set([node.name for node in self.model.graph().input])
        graph_output_names = set([node.name for node in self.model.graph().output])

        v_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Reshape", "Transpose", "Reshape", "Add", "MatMul"],
            [1, 0, 0, 0, None],
        )
        v_nodes_with_past_self_attn = self.model.match_parent_path(
            # Decoder attention with past value concatenated before MatMul
            matmul_qkv,
            ["Reshape", "Concat", "Transpose", "Reshape", "Add", "MatMul"],
            [1, 0, 1, 0, 0, None],
        )
        v_nodes_with_past_cross_attn = self.model.match_parent_path(
            # Decoder attention with past value directly used in MatMul
            matmul_qkv,
            ["Reshape"],
            [1],
        )
        past_v, present_v = "", ""
        reshape_v_2, add_v = None, None
        if v_nodes is not None:
            (reshape_v_2, transpose_v, reshape_v_1, add_v, matmul_v) = v_nodes
            # For initial pass through encoder-decoder_with_past to get starting past values (beam search)
            present_v = transpose_v.output[0]
        elif v_nodes_with_past_self_attn is not None:
            (reshape_v_2, concat_v, transpose_v, reshape_v_1, add_v, matmul_v) = v_nodes_with_past_self_attn
            v_nodes = v_nodes_with_past_self_attn
            past_v = concat_v.input[0]
            present_v = concat_v.output[0]
        elif (
            v_nodes_with_past_cross_attn is not None and v_nodes_with_past_cross_attn[-1].input[0] in graph_input_names
        ):
            v_nodes = v_nodes_with_past_cross_attn
            past_v = v_nodes[-1].input[0]
            present_v = v_nodes[-1].output[0]
            if present_v not in graph_output_names:
                identity_node_v = list(
                    filter(lambda node: node.op_type == "Identity", self.model.input_name_to_nodes()[past_v])
                )
                present_v = identity_node_v[0].output[0] if len(identity_node_v) == 1 else ""
        else:
            logger.debug("fuse_attention: failed to match v path")
            return
        past_v = past_v if past_v in graph_input_names else ""
        present_v = present_v if present_v in graph_output_names else ""

        qk_nodes_1 = self.model.match_parent_path(matmul_qkv, ["Softmax", "MatMul"], [0, 0])
        qk_nodes_2 = self.model.match_parent_path(
            matmul_qkv, ["Softmax", "Reshape", "Add", "Reshape", "MatMul"], [0, 0, 0, 0, 0]
        )
        if qk_nodes_1 is not None:
            _, matmul_qk = qk_nodes_1
            qk_nodes = qk_nodes_1
        elif qk_nodes_2 is not None:
            _, _, add_qk, _, matmul_qk = qk_nodes_2
            qk_nodes = qk_nodes_2
        else:
            return

        q_nodes = self.model.match_parent_path(
            matmul_qk,
            ["Reshape", "Transpose", "Reshape", "Mul", "Add", "MatMul"],
            [0, 0, 0, 0, 0, 1],
        )
        if q_nodes is not None:
            reshape_q_2, transpose_q, reshape_q_1, mul_q, add_q, matmul_q = q_nodes
        else:
            return

        k_nodes_with_bias = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Reshape", "Transpose", "Reshape", "Add", "MatMul"],
            [1, 0, 0, 0, 0, 1],
        )
        k_nodes_no_bias = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Reshape", "Transpose", "Reshape", "MatMul"],
            [1, 0, 0, 0, 0],
        )
        k_nodes_no_bias_with_past_self_attn = self.model.match_parent_path(
            # Decoder attention with past key concatenated before MatMul
            matmul_qk,
            ["Transpose", "Reshape", "Concat", "Transpose", "Reshape", "MatMul"],
            [1, 0, 0, 1, 0, 0],
        )
        k_nodes_no_bias_with_past_cross_attn = self.model.match_parent_path(
            # Decoder attention with past key directly used in MatMul
            matmul_qk,
            ["Transpose", "Reshape"],
            [1, 0],
        )
        past_k, present_k = "", ""
        reshape_k_2, reshape_k_1, matmul_k = None, None, None
        if k_nodes_with_bias is not None:
            _, reshape_k_2, transpose_k_1, reshape_k_1, add_k, matmul_k = k_nodes_with_bias
            k_nodes = k_nodes_with_bias
        elif k_nodes_no_bias is not None:
            _, reshape_k_2, transpose_k_1, reshape_k_1, matmul_k = k_nodes_no_bias
            k_nodes = k_nodes_no_bias
            # For initial pass through encoder-decoder_with_past to get starting past values (beam search)
            present_k = transpose_k_1.output[0]
        elif k_nodes_no_bias_with_past_self_attn is not None:
            _, reshape_k_2, concat_k, _, reshape_k_1, matmul_k = k_nodes_no_bias_with_past_self_attn
            k_nodes = k_nodes_no_bias_with_past_self_attn
            past_k = concat_k.input[0]
            present_k = concat_k.output[0]
        elif (
            k_nodes_no_bias_with_past_cross_attn is not None
            and k_nodes_no_bias_with_past_cross_attn[-1].input[0] in graph_input_names
        ):
            k_nodes = k_nodes_no_bias_with_past_cross_attn
            past_k = k_nodes[-1].input[0]
            present_k = k_nodes[-1].output[0]
            if present_k not in graph_output_names:
                identity_node_k = list(
                    filter(lambda node: node.op_type == "Identity", self.model.input_name_to_nodes()[past_k])
                )
                present_k = identity_node_k[0].output[0] if len(identity_node_k) == 1 else ""
        else:
            return
        past_k = past_k if past_k in graph_input_names else ""
        present_k = present_k if present_k in graph_output_names else ""

        if k_nodes in (k_nodes_no_bias, k_nodes_no_bias_with_past_self_attn):
            # Create empty Add node for attention graph
            bias_dim = self.model.get_initializer(add_v.input[0]).dims[0]
            empty_bias_name = "empty_bias"
            empty_tensor = self.model.get_initializer(empty_bias_name)
            if empty_tensor is None:
                empty_tensor = helper.make_tensor(empty_bias_name, TensorProto.FLOAT, [bias_dim], [0.0] * bias_dim)
                self.model.add_initializer(empty_tensor, self.this_graph_name)

            add_name = self.model.create_node_name("Add")
            add_k = helper.make_node("Add", [empty_bias_name, matmul_k.output[0]], [reshape_k_1.name], add_name)

        if not past_k and not self.check_runtime_shape_path(
            reshape_qkv_2,
            reshape_qkv_1,
            reshape_q_2,
            reshape_k_2,
            reshape_v_2,
            root_input,
        ):
            return

        three_root_inputs = past_k and past_v and matmul_k is None and "matmul_v" not in locals()
        one_root_input = (
            not three_root_inputs
            and matmul_k.input[0] == root_input
            and matmul_q.input[0] == root_input
            and matmul_v.input[0] == root_input
        )
        two_root_inputs = (
            not three_root_inputs
            and matmul_q.input[0] == root_input
            and matmul_k.input[0] == matmul_v.input[0]
            and matmul_k.input[0] != matmul_q.input[0]
        )

        # There are 5 types of attention:
        # 1) Encoder attention with one_root_input=True and qk_nodes=qk_nodes_1
        # 2) Decoder attention with one_root_input=True and qk_nodes=qk_nodes_2
        # 3) Decoder attention with past with one_root_input=True and qk_nodes=qk_nodes_1 and past_k=past_decoder_key and past_v=past_decoder_value
        # 4) Decoder cross attention with two_root_inputs=True and qk_nodes=qk_nodes_1
        # 5) Decoder cross attention with past with three_root_inputs=True and qk_nodes=qk_nodes_1
        encoder_attention = one_root_input and qk_nodes == qk_nodes_1
        decoder_attention = one_root_input and qk_nodes == qk_nodes_2
        decoder_attention_with_past = encoder_attention and past_k and past_v
        decoder_cross_attention = two_root_inputs and qk_nodes == qk_nodes_1
        decoder_cross_attention_with_past = three_root_inputs and qk_nodes == qk_nodes_1

        # For decoder_attention, the attention mask needs to be included in the attention node
        mask_index = None
        if decoder_attention:
            mask_nodes_bart = self.model.match_parent_path(
                add_qk,
                ["Where"],
                [1],
            )
            mask_nodes_whisper = self.model.match_parent_path(
                add_qk,
                ["Expand", "Unsqueeze", "Unsqueeze", "Where"],
                [1, 0, 0, 0],
            )
            if mask_nodes_whisper is not None:
                mask_index = mask_nodes_whisper[0].output[-1]
            elif mask_nodes_bart is not None:
                mask_index = mask_nodes_bart[0].output[-1]

        if (
            encoder_attention
            or decoder_attention
            or decoder_attention_with_past
            or decoder_cross_attention
            or decoder_cross_attention_with_past
        ):
            attention_last_node = reshape_qkv_2
            num_heads, hidden_size = self.get_num_heads_and_hidden_size(reshape_q_1)

            if num_heads <= 0 or hidden_size <= 0 or (hidden_size % num_heads) != 0:
                logger.debug("fuse_attention: failed to detect num_heads or hidden_size")
                return

            new_node = None
            if decoder_attention_with_past or decoder_cross_attention or decoder_cross_attention_with_past:
                # Note: Decoder attention with past key and past value is fused as multihead attention
                # rather than attention because multihead attention supports separate past key and past
                # value whereas attention supports concatenated past key and past value.
                new_node = (
                    self.create_multihead_attention_node(
                        matmul_q,
                        matmul_k if decoder_cross_attention or decoder_attention_with_past else past_k,
                        matmul_v if decoder_cross_attention or decoder_attention_with_past else past_v,
                        add_q,
                        add_k if decoder_cross_attention or decoder_attention_with_past else None,
                        add_v if decoder_cross_attention or decoder_attention_with_past else None,
                        num_heads,
                        hidden_size,
                        attention_last_node.output[0],
                        past_k=past_k if decoder_attention_with_past else "",
                        past_v=past_v if decoder_attention_with_past else "",
                        present_k=present_k,
                        present_v=present_v,
                        packed_qkv=decoder_attention_with_past,
                    )
                    if self.use_multi_head_attention
                    else None
                )
            else:
                # Temporarily set multihead attention flag to false
                use_multi_head_attention_ground_truth = self.use_multi_head_attention
                self.use_multi_head_attention = False
                new_node = self.create_attention_node(
                    None,
                    matmul_q,
                    matmul_k,
                    matmul_v,
                    add_q,
                    add_k,
                    add_v,
                    num_heads,
                    hidden_size,
                    root_input,
                    attention_last_node.output[0],
                    add_qk_str=mask_index if decoder_attention else None,
                    past_k=past_k,
                    past_v=past_v,
                    present_k=present_k,
                    present_v=present_v,
                )
                self.use_multi_head_attention = use_multi_head_attention_ground_truth
            if new_node is None:
                return

            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name

            self.nodes_to_remove.extend([attention_last_node, transpose_qkv, matmul_qkv])
            self.nodes_to_remove.extend(qk_nodes)

            # When using multihead attention, keep MatMul nodes in original graph
            if decoder_attention_with_past or decoder_cross_attention or decoder_cross_attention_with_past:
                if q_nodes[-1].op_type == "MatMul":
                    q_nodes.pop()
                if k_nodes[-1].op_type == "MatMul":
                    k_nodes.pop()
                if v_nodes[-1].op_type == "MatMul":
                    v_nodes.pop()

            self.nodes_to_remove.extend(q_nodes)
            self.nodes_to_remove.extend(k_nodes)
            self.nodes_to_remove.extend(v_nodes)

            # Use prune graph to remove mask nodes since they are shared by all attention nodes.
            self.prune_graph = True
