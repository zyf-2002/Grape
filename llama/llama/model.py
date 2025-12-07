# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from torch import nn

all_w_list = []

SCALE = 1 << 16
SCALE2 = 1 << 32
layer_num = 0
num_norm = 0
import torch

def save_int(t: torch.Tensor, path):
    if path[-4:] != '.bin':
        raise ValueError('Path must end with .bin')

    # 如果是形状 (1, tokens, dim) 的 tensor，去掉第 0 维
    if t.dim() >= 3 and t.shape[0] == 1:
        t_ = t.squeeze(0)  # (1, tokens, dim) -> (tokens, dim)
    else:
        t_ = t

    # 转成 int32
    arr = t_.cpu().detach().numpy().astype(np.int32)

    # 以二进制追加方式写入文件
    with open(path, 'ab') as f:
        arr.tofile(f)
        f.flush()
    
def load_int(path, device = 0):
    if path[-4:] != '.bin':
        raise ValueError('Path must end with .bin')
    return torch.from_numpy(np.fromfile(path, dtype=np.int32)).to(device)


def cosine_similarity_torch(x, y):
    x = x.flatten().to(torch.float32)
    y = y.flatten().to(torch.float32)
    return torch.dot(x, y) / (torch.norm(x) * torch.norm(y) + 1e-8)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        temp = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).to(torch.float16)
        
        return temp
        
        #return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        
        tempx = (x.to(torch.float64) / SCALE).to(torch.float16)
        rms = self._norm(tempx.float()).type_as(tempx)

        rms_int = torch.round(rms.to(torch.float64) * SCALE).to(torch.int32)
        
        # print("rms max abs:", rms.abs().max().item())
        # print("rms_weight max abs:", self.weight.abs().max().item())
        
        
        #量化
        
        global num_norm
        w = torch.round(self.weight.to(torch.float64) * SCALE).to(torch.int32)
        #print(w.shape)
        
        if layer_num > 28:
            if num_norm % 2 == 0:
                save_int(w, f'../data/W/NormFirst_layer.bin')
                all_w_list.append(w)
            else:
                save_int(w, f'../data/W/NormSecond_layer.bin')
        
        
        num_norm += 1

        
       # 乘法+反量化
        output = ((rms_int.to(torch.int64)) * (w.to(torch.int64)) + SCALE//2) // SCALE
        output = output.to(torch.int32)


        
        
        #yuan =  rms * self.weight
        
        # similarity = cosine_similarity_torch(yuan, output)
        # print(f"余弦相似度: {similarity:.6f}")
        # diff = (yuan - output).to(torch.float32)   # 转成 float32
        # print("Max absolute difference after rounding:", diff.abs().max().item())
        
        
        return output

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.dim = args.dim

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        
        

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        
        # self.cache_k_float = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # ).cuda()
        # self.cache_v_float = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        
        WK_int = torch.round(((self.wk.weight).to(torch.float64)) * SCALE).to(torch.int32)
        WQ_int = torch.round(((self.wq.weight).to(torch.float64)) * SCALE).to(torch.int32)
        WV_int = torch.round(((self.wv.weight).to(torch.float64)) * SCALE).to(torch.int32)
        WO_int = torch.round(((self.wo.weight).to(torch.float64)) * SCALE).to(torch.int32)
        
        if layer_num > 28:
            save_int(WK_int, f'../data/W/Attention_K_layer.bin')
            save_int(WQ_int, f'../data/W/Attention_Q_layer.bin')
            save_int(WV_int, f'../data/W/Attention_V_layer.bin')  #保存权重到文件
            save_int(WO_int, f'../data/W/Attention_O_layer.bin')
       
        x_int = x

        # 提升到 int64 防止溢出
        XQ_int = torch.matmul(x_int.to(torch.float64), WQ_int.t().to(torch.float64)).to(torch.int64)
        XK_int = torch.matmul(x_int.to(torch.float64), WK_int.t().to(torch.float64)).to(torch.int64)
        XV_int = torch.matmul(x_int.to(torch.float64), WV_int.t().to(torch.float64)).to(torch.int64)
        
            
        # 反量化
        xq = (XQ_int + SCALE//2) // SCALE
        xk = (XK_int + SCALE//2) // SCALE
        xv = (XV_int + SCALE//2) // SCALE
       
        
        #xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
       
        

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq = (xq.to(torch.float64) / SCALE).to(torch.float32)
        xk = (xk.to(torch.float64) / SCALE).to(torch.float32)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        xq = (xq.to(torch.float64) * SCALE).to(torch.int32)
        xk = (xk.to(torch.float64) * SCALE).to(torch.int32)
        
        
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xv)
        

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
        

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        
        scores = torch.matmul(xq.to(torch.float64), keys.transpose(2, 3).to(torch.float64)).to(torch.int64)
        scores = (scores + SCALE//2) // SCALE
        scores = (scores.to(torch.float64) / SCALE).to(torch.float32)
        scores = scores / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(scores)
        # scores =scores * ~mask1
        
        # if mask is not None:
        #     scores = scores + mask
        #     shift = scores
        #     shift = (shift.to(torch.float64) / SCALE2).to(torch.float32)
            
        #     shift = (shift.to(torch.float64) / math.sqrt(self.head_dim)).to(torch.float32)
        #     shift = math.sqrt(self.head_dim) * torch.log((torch.exp(shift)).sum(axis = -1, keepdim = True))
        #     shift = (shift.to(torch.float64) * SCALE2).to(torch.int64)
        #     scores -= shift
        #     scores = (scores.to(torch.float64) / (SCALE2 * math.sqrt(self.head_dim))).to(torch.float32)
            
        #     scores = (torch.exp(scores).float()) 
        # else:
        #     scores = (scores.to(torch.float64) / SCALE2).to(torch.float32)
        #     scores = scores / math.sqrt(self.head_dim)
        #     scores = F.softmax(scores.float(), dim=-1).type_as(xq)

            
        scores = (scores.to(torch.float64) * SCALE).to(torch.int32)
        
        #print("rms max abs:", scores.to(torch.float64).abs().max().item())
        
        
        
        output = torch.matmul(scores.to(torch.float64), values.to(torch.float64)).to(torch.int64)  # (bs, n_local_heads, seqlen, head_dim)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        
        output = (output + SCALE//2) // SCALE
        
        
        
        wout = torch.matmul(output.to(torch.float64), WO_int.t().to(torch.float64)).to(torch.int64)
        wout = (wout + SCALE//2) // SCALE
        
        
        
        
        
        # #--------------------------------------------------------------------------------------------------------------------------
        # bsz, seqlen, _ = x.shape
        # xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        # xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # self.cache_k_float[:bsz, start_pos : start_pos + seqlen] = xk
        # self.cache_v_float[:bsz, start_pos : start_pos + seqlen] = xv

        # keys = self.cache_k_float[:bsz, : start_pos + seqlen]
        # values = self.cache_v_float[:bsz, : start_pos + seqlen]

        # # repeat k/v heads if n_kv_heads < n_heads
        # keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        # values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        # xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        # keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        # values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        # scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # if mask is not None:
        #     scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        # scores = F.softmax(scores.float(), dim=-1).type_as(xv)
        # output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        # output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
       
        # nwout = self.wo(output)
        
        # similarity = cosine_similarity_torch(nwout, wout)
        # print(f"余弦相似度: {similarity:.6f}")
        # diff = (nwout - wout).to(torch.float32)   # 转成 float32
        # print("Max absolute difference after rounding:", diff.abs().max().item())
        #-------------------------------------------------------------------------------------------------------------------------------------
        return wout


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        x_int = x
        x_int.to(torch.int32)
        w1_int = torch.round(((self.w1.weight).to(torch.float64)) * SCALE).to(torch.int32)
        w2_int = torch.round(((self.w2.weight).to(torch.float64)) * SCALE).to(torch.int32)
        w3_int = torch.round(((self.w3.weight).to(torch.float64)) * SCALE).to(torch.int32)
        
        if layer_num > 28:
            # max_x  = torch.abs(x_int).max().item()
            # max_w1 = torch.abs(w1_int).max().item()
            # max_w2 = torch.abs(w2_int).max().item()
            # max_w3 = torch.abs(w3_int).max().item()

            # print(f"max |X|   = {max_x}")
            # print(f"max |W1|  = {max_w1}")
            # print(f"max |W2|  = {max_w2}")
            # print(f"max |W3|  = {max_w3}")

            save_int(w1_int, f'../data/W/gate_layer.bin')
            save_int(w2_int, f'../data/W/down_layer.bin')
            save_int(w3_int.t(), f'../data/W/up_layer.bin')
            save_int(x_int, f'../data/X.bin')
        
        
        
        
        
        temp_x1 = torch.matmul(x_int.to(torch.float64), w1_int.t().to(torch.float64)).to(torch.int64)
        
        temp_x3 = torch.matmul(x_int.to(torch.float64), w3_int.t().to(torch.float64)).to(torch.int64)
        

        
        q_temp_x3 = ((temp_x3 + SCALE//2) // SCALE).to(torch.int32)
        rem_temp_x3 = (temp_x3 - q_temp_x3 * SCALE).to(torch.int32)
        
        
        
        # print(q_temp_x3.shape)
        if layer_num > 28:
            save_int(q_temp_x3, f'../data/X_up.bin')
            save_int(rem_temp_x3, f'../data/X_up_rem.bin')
            # y_quot = load_int('../data/X_up.bin')
            # y_quot = y_quot.reshape(1, 1024, 11008) 
            # print(y_quot.shape)
            # if torch.equal(y_quot, q_temp_x3):
            #     print("✔️ 还原成功，完全一致！")
                
        # if layer_num > 28:
        #     y_quot = load_int('../data/X_up.bin')
        #     y_quot = y_quot.reshape(1024, 11008) 
        #     y_rem = load_int('../data/X_up_rem.bin')
        #     y_rem = y_rem.reshape(1024, 11008)
        #     xx = load_int('../data/X.bin')
        #     xx = xx.reshape(1024, 4096)
        #     ww3 = load_int('../data/W/up_layer.bin')
        #     ww3 = ww3.reshape(4096, 11008)
        #     temp_test = torch.matmul(xx.to(torch.float64), ww3.t().to(torch.float64)).to(torch.int64)
        #     reco_temp_x3 = (q_temp_x3.to(torch.int64)) * SCALE + rem_temp_x3
        
            # if torch.equal(reco_temp_x3, temp_test):
            #     print("✔️ 还原成功，完全一致！")
        
        
        temp_x1 = (temp_x1 + SCALE//2) // SCALE

        temp_x1 = (temp_x1.to(torch.float64) / SCALE).to(torch.float16)
        #print("rms max abs:", temp_x1.to(torch.float16).abs().max().item())
        temp_x1 = F.silu(temp_x1)
        temp_x1 = torch.round((temp_x1.to(torch.float64)) * SCALE).to(torch.int32)
        temp_x13 = ((temp_x1.to(torch.int64)) * (q_temp_x3.to(torch.int64)) + SCALE//2) // SCALE
        temp_x13 = temp_x13.to(torch.int32)
        #print(temp_x13.shape)
        temp_x123 = torch.matmul(temp_x13.to(torch.float64), w2_int.t().to(torch.float64)).to(torch.int64)
        
        temp_x123 = (temp_x123 + SCALE//2) // SCALE
        #out = (temp_x123.to(torch.float64) / SCALE).to(torch.float16)
        out = temp_x123.to(torch.int32)
        
        #print("rms max abs:", out.to(torch.float64).abs().max().item())
        
        #temp = self.w2(F.silu(self.w1(x)) * self.w3(x))
    
        # similarity = cosine_similarity_torch(temp, out)
        # print(f"余弦相似度: {similarity:.6f}")
        # diff = (temp - out).to(torch.float32)   # 转成 float32
        # print("Max absolute difference after rounding:", diff.abs().max().item())
        return out     #self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        global layer_num
        layer_num += 1
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        

        h = h.to(torch.int32)
        
        out = h + self.feed_forward(self.ffn_norm(h))
        
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        files_to_clear = [
            "../data/W/NormFirst_layer.bin",
            "../data/W/NormSecond_layer.bin",
            "../data/W/gate_layer.bin",
            "../data/W/down_layer.bin",
            "../data/W/up_layer.bin",
            "../data/W/Attention_V_layer.bin",
            "../data/W/Attention_Q_layer.bin",
            "../data/W/Attention_K_layer.bin",
            "../data/W/Attention_O_layer.bin",
        ]
        # 遍历每个文件，用 wb 模式打开就会清空文件内容
        for path in files_to_clear:
            # 如果文件不存在也没关系，会创建空文件
            with open(path, 'wb') as f:
                pass


        _bsz, seqlen = tokens.shape
        
        h = self.tok_embeddings(tokens)
        print("---------------------------------------------------------------------------")
        #print(h.shape)
        h_temp = h
        h_int = torch.round((h.to(torch.float32)) * SCALE).to(torch.int32)
        
       
        h = h_int
        
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        for i, layer in enumerate(self.layers):
            print(f"------------------------------------------------------正在执行第 {i} 层-----------------")  # 这里 i 从 0 开始
            h = layer(h, start_pos, freqs_cis, mask)
            
        # all_w = torch.stack(all_w_list, dim=0)
        # print(all_w.shape)
        # data = load_int("../data/W/NormFirst_layer.bin").view_as(all_w)
        # print(torch.equal(all_w, data)) 
            
        h = self.norm(h)
        # h = (h.to(torch.float64) / SCALE).to(torch.float16)
        # output = self.output(h).float()
        output_weight = torch.round(((self.output.weight).to(torch.float64)) * SCALE).to(torch.int32)
        output = torch.matmul(h.to(torch.float64), output_weight.t().to(torch.float64)).to(torch.int64)
        output = (output + SCALE//2) // SCALE
        output = (output.to(torch.float64) / SCALE).to(torch.float16)
        #print(output.shape)
        return output
