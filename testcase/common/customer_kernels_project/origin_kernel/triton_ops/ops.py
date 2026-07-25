import torch.nn as nn
from sequence_tensor.seq_tensor import SequenceTensor
from .kernels import *


class FlashAttention(nn.Module):
    """warp function as nn.Module for fp8 quant"""
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, q, k, v, mask_fn):
        if hasattr(self, 'enable_fp8_serving') and self.enable_fp8_serving:
            query_amax = self.query_amax
            key_amax = self.key_amax
            value_amax = self.value_amax
            output_amax = self.output_amax
            enable_fp8_serving = True
        else:
            query_amax = None
            key_amax = None
            value_amax = None
            output_amax = None
            enable_fp8_serving = False

        result = FlashAttentionFunc.apply(
            q._data,
            k._data,
            v._data,
            q._attn_arg,
            k._attn_arg,
            q._offset,
            k._offset,
            q._max_length,
            k._max_length,
            self.scale,
            mask_fn,
            False,
            query_amax,
            key_amax,
            value_amax,
            output_amax,
            enable_fp8_serving,
        )
        return SequenceTensor(result, q._offset, q._position, q._attn_arg, q._max_length)


class Rope(nn.Module):
    """warp function as nn.Module for fp8 quant"""
    def __init__(self, base=10000.0, reverse=False):
        super().__init__()
        self.base = base
        self.reverse = reverse

    def forward(self, seq):
        xseq = RopeFunc.apply(
            seq._data, seq._position, seq._offset, seq._max_length, self.base, self.reverse
        )
        return SequenceTensor(
            xseq, seq._offset, seq._position, seq._attn_arg, seq._max_length
        )


def flash_attention(q, k, v, scale, mask_fn):
    result = FlashAttentionFunc.apply(
        q._data,
        k._data,
        v._data,
        q._attn_arg,
        k._attn_arg,
        q._offset,
        k._offset,
        q._max_length,
        k._max_length,
        scale,
        mask_fn,
        False,
    )
    return SequenceTensor(result, q._offset, q._position, q._attn_arg, q._max_length)


def rope(seq, base=10000.0, reverse=False):
    xseq = RopeFunc.apply(
        seq._data, seq._position, seq._offset, seq._max_length, base, reverse
    )
    return SequenceTensor(
        xseq, seq._offset, seq._position, seq._attn_arg, seq._max_length
    )


def fused_matmul(x, w, b):
    y = FusedMatmul.apply(x._data, w, b)
    return SequenceTensor(y, x._offset, x._position, x._attn_arg, x._max_length)


def fused_swiglu(x, w_g, w_fc, b_g, b_fc, is_training=True, is_recompute=False):
    if is_training:
        w_g = w_g.to(x.dtype)
        b_g = b_g.to(x.dtype)
        w_fc = w_fc.to(x.dtype)
        b_fc = b_fc.to(x.dtype)
    y = FusedSwiglu.apply(x._data, w_g, w_fc, b_g, b_fc, is_training, is_recompute)
    return SequenceTensor(y, x._offset, x._position, x._attn_arg, x._max_length)


def softcap(x, softcap=50.0):
    y = Softcap.apply(x._data, softcap)
    return SequenceTensor(y, x._offset, x._position, x._attn_arg, x._max_length)
