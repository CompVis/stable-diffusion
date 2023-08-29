# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


class Operators:
    ATTENTION = "Attention"
    LAYERNORM = "LayerNormalization"
    PACKEDATTENTION = "PackedAttention"
    REMOVEPADDING = "RemovePadding"
    RESTOREPADDING = "RestorePadding"
    SKIPLAYERNORM = "SkipLayerNormalization"


class AttentionInputIDs:
    INPUT = 0
    WEIGHTS = 1
    BIAS = 2
    MASK_INDEX = 3
    PAST = 4
    RELATIVE_POSITION_BIAS = 5
    PAST_SEQUENCE_LENGTH = 6


class AttentionOutputIDs:
    OUTPUT = 0
    PRESENT = 1
