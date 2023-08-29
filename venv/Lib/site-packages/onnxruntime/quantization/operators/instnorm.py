# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from .qdq_base_operator import QDQOperatorBase


class QDQInstanceNormalization(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "InstanceNormalization"

        # Input
        self.quantizer.quantize_activation_tensor(node.input[0])
        if not self.disable_qdq_for_node_output:
            self.quantizer.quantize_activation_tensor(node.output[0])

        # Scale
        if self.quantizer.is_per_channel():
            self.quantizer.quantize_weight_tensor_per_channel(node.input[1], axis=1)
        else:
            self.quantizer.quantize_weight_tensor(node.input[1])

        # Bias
        self.quantizer.quantize_bias_tensor(node.input[2], node.input[0], node.input[1])
