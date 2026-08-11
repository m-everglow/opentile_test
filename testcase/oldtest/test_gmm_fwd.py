# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import triton
import triton.language as tl
import torch
import torch_npu
import numpy as np
import itertools
import pytest


CORE_NUM = 28
DEV = "npu"


def group_gemm_split_m_high(A, B, split_sizes):
    """
    高精度真值
    """
    # 1. 按照 split_sizes 将 A 在第 0 维切分成 g 个小矩阵
    #    A_parts[i] 的形状为 (mi, K)
    # A = A.to(torch.float32)
    # B = B.to(torch.float32)
    A_parts = torch.split(A, split_sizes, dim=0)
    
    # 2. 使用 zip 将切分后的 A 与 B 对应起来，分别做矩阵乘法
    #    a: (mi, K), b: (K, N) -> res: (mi, N)
    #    这一步利用了 Python 列表推导式，逻辑非常清晰
    results = [torch.matmul(a, b.transpose(0,1)) for a, b in zip(A_parts, B)]
    
    # 3. 将计算结果在第 0 维拼接回 (M, N)
    return torch.cat(results, dim=0)

def group_gemm_split_m_low(A, B, split_sizes):
    """
    小算子拼接模拟
    """
    # 1. 按照 split_sizes 将 A 在第 0 维切分成 g 个小矩阵
    #    A_parts[i] 的形状为 (mi, K)
    A_parts = torch.split(A, split_sizes, dim=0)
    
    # 2. 使用 zip 将切分后的 A 与 B 对应起来，分别做矩阵乘法
    #    a: (mi, K), b: (K, N) -> res: (mi, N)
    #    这一步利用了 Python 列表推导式，逻辑非常清晰
    results = [torch.matmul(a, b.transpose(0,1)) for a, b in zip(A_parts, B)]
    
    # 3. 将计算结果在第 0 维拼接回 (M, N)
    return torch.cat(results, dim=0)

BLOCK_M_CANDIDATES = [32, 64, 128, 256]
BLOCK_N_CANDIDATES = [128, 256, 768]
BLOCK_K_CANDIDATES = [128, 256, 512]

autotune_configs = [
    triton.Config({
        'BLOCK_SIZE_M': m, 
        'BLOCK_SIZE_N': n, 
        'BLOCK_SIZE_K': k
    }) for m, n, k in itertools.product(
        BLOCK_M_CANDIDATES, 
        BLOCK_N_CANDIDATES, 
        BLOCK_K_CANDIDATES
    )
]

# @triton.autotune(configs=autotune_configs, key=['avg_group_size', 'fp_ratio','G', 'N', 'K'])
@triton.jit
def group_gemm_split_m_kernel(
    a_ptr, 
    b_ptr, 
    c_ptr,
    # Look-Up Tables (LUTs) for Split-M Scheduling
    lut_g_size,
    avg_group_size, fp_ratio,
    # Matrix dimensions
    G, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bg, stride_bn, stride_bk,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Group GEMM Kernel (Split-M).
    """
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)

    # 计算全局总BlockM数
    num_blocks_m_global = 0
    for g in range(G):
        g_size = tl.load(lut_g_size + g)
        num_blocks_m_global += tl.cdiv(g_size, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_blocks = num_blocks_m_global * num_blocks_n

    # 增量计算参数
    g_last = 0
    m_start_base = 0
    num_blocks_m_accum = 0

    for block_idx in range(pid, total_blocks, num_core):

        pid_mb_global = block_idx // num_blocks_n  # 当前Block的全局BlockM ID
        pid_n = block_idx % num_blocks_n           # 当前Block的BlockN ID

        g_now = 0               # 当前Group ID
        g_size_now = 0          # 当前Group的M大小
        
        # 计算当前所属的group，及其对应的size、所属block、偏移位置
        found = 1
        for g in range(g_last, G):
            if found:
                g_size = tl.load(lut_g_size + g)
                g_blocks = tl.cdiv(g_size, BLOCK_SIZE_M)
                m_start_base += g_size
                if (num_blocks_m_accum + g_blocks) > pid_mb_global:
                    g_now = g
                    g_last = g
                    g_size_now = g_size
                    found = 0
                    num_blocks_m_accum -= g_blocks
                    m_start_base -= g_size
                num_blocks_m_accum += g_blocks

        # group内blockm偏移
        pid_mb_in_g = pid_mb_global - num_blocks_m_accum
        # M维度起始偏移
        m_start = m_start_base + pid_mb_in_g * BLOCK_SIZE_M
        # 计算当前block大小
        remaining_m = g_size_now - pid_mb_in_g * BLOCK_SIZE_M
        m_size = min(BLOCK_SIZE_M, remaining_m)
        
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_m = tl.arange(0, BLOCK_SIZE_M)

        a_ptrs_base = a_ptr + (m_start + offs_m[:, None]) * stride_am + offs_k[None, :] * stride_ak

        b_ptrs_base = b_ptr + (g_now * stride_bg) + \
                    (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)

        msk_m = offs_m < m_size
        msk_n = offs_n < N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

            a_ptrs = a_ptrs_base + k * BLOCK_SIZE_K * stride_ak
            b_ptrs = b_ptrs_base + k * BLOCK_SIZE_K * stride_bk
            
            msk_k = offs_k < (K - k * BLOCK_SIZE_K)

            # Load A
            a = tl.load(
                a_ptrs,
                mask=msk_m[:, None] & msk_k[None, :],
                other=0.0
            )

            # Load B
            b = tl.load(
                b_ptrs,
                mask=msk_n[:, None] & msk_k[None, :],
                other=0.0
            )

            accumulator = tl.dot(a, tl.trans(b), accumulator)

        c_ptrs = c_ptr + (m_start + offs_m[:, None]) * stride_cm + offs_n[None, :] * stride_cn    

        tl.store(c_ptrs, accumulator, mask=msk_m[:, None] & msk_n[None, :])


def group_gemm_split_m_triton(a, b, m_splits):
    """
    a: (Sum_M, K)
    b: (G, K, N)
    m_splits: List[int], 每个组的 M 大小，sum(m_splits) == Sum_M
    """
    device = a.device
    # 维度检查
    assert a.shape[0] == sum(m_splits)
    assert b.shape[0] == len(m_splits)
    G, N, K = b.shape

    avg_group_size = sum(m_splits) // G
    avg_group_size = avg_group_size if avg_group_size == triton.next_power_of_2(avg_group_size) \
                                    else triton.next_power_of_2(avg_group_size) // 2

    lut_g_size = []
    full_blocks = 0
    partial_blocks = 0
    for g_idx, m_size in enumerate(m_splits):
        lut_g_size.append(m_size)
        full_blocks += m_size // avg_group_size
        partial_blocks += 1 if m_size % avg_group_size != 0 else 0

    lut_g_size = torch.tensor(lut_g_size, dtype=torch.int32, device=device)

    fp_ratio = full_blocks / (full_blocks + partial_blocks)

    if fp_ratio >= 0.95:
        fp_ratio = 10
    elif fp_ratio <= 0.05:
        fp_ratio = 0
    else:
        fp_ratio = round(fp_ratio / 0.1) * 1  # 1步长
    print('avg_group_size ',avg_group_size,' fp_ratio ',fp_ratio)

    c = torch.empty((a.shape[0], N), device=device, dtype=torch.float32)
    
    # Grid: Cube核数
    grid = (CORE_NUM // 2,)

    # 指定最佳tile适配msprof性能测试
    # BLOCK_SIZE_M = triton.next_power_of_2(sum(m_splits) // G)
    # BLOCK_SIZE_M = min(128, BLOCK_SIZE_M)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    print("BLOCK_SIZE_M: ", BLOCK_SIZE_M)
    
    group_gemm_split_m_kernel[grid](
        a, b, c,
        lut_g_size,
        avg_group_size, fp_ratio,
        G, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        # multibuffer=True,
    )
    
    return c


def group_gemm_split_m_aclnn(A, B, split_sizes):
    x = [A,]
    weight = [B.transpose(1,2).contiguous(),]
    group_list = torch.tensor(np.cumsum(split_sizes), dtype=torch.int64, device=A.device)
    split_item = 3
    npu_out = torch_npu.npu_grouped_matmul(x, weight, group_list=group_list, split_item=split_item, group_type=0)
    return torch.cat(npu_out, dim=0)


def gen_normal_int_list(g, mu=50, var=20):
    """
    生成服从N(μ, σ²)的整数列表
    :param g: 列表长度（正整数）
    :param mu: 均值（默认50）
    :param var: 方差（默认20）
    :return: 整数列表
    """

    if not isinstance(g, int) or g <= 0:
        raise ValueError("列表长度g必须为正整数")

    sigma = np.sqrt(var)
    normal_float = np.random.normal(loc=mu, scale=sigma, size=g)
    normal_int = np.round(normal_float).astype(int)
    
    return normal_int.tolist()


@pytest.mark.parametrize("g, mu, var, N, K", [
        # 等长
        # (1, 50, 0, 64, 64)
        (80, 64, 0, 768, 1024),
        (160, 64, 0, 768, 1024),
        (240, 64, 0, 768, 1024),
        (80, 512, 0, 768, 1024),
        (160, 512, 0, 768, 1024),
        (240, 512, 0, 768, 1024),
        # # 不等长
        # (80, 64, 20, 768, 1024),
        # (160, 64, 20, 768, 1024),
        # (240, 64, 20, 768, 1024),
        # (80, 512, 100, 768, 1024),
        # (160, 512, 100, 768, 1024),
        # (240, 512, 100, 768, 1024),
    ])
#@pytest.mark.parametrize("dtype", [torch.float32, torch.float32], ids=["fp32", "fp32"])
@pytest.mark.parametrize("dtype", [torch.float16], ids=["fp16"])
def test_group_gemm(g, mu, var, N, K, dtype):
    # Unit Test
    torch.npu.set_device(0)
    torch.manual_seed(0)


    m_splits = gen_normal_int_list(g, mu, var)
    M = sum(m_splits)
    # print(" g:", g, " mu:", mu, " var:", var, " M:", M, " N:", N, " K:", K)

    # 构造数据
    A = torch.randn(M, K, device="cpu", dtype=dtype)
    B = torch.randn(g, N, K, device="cpu", dtype=dtype)    
    A = A.to(DEV)
    B = B.to(DEV)

    true_output = group_gemm_split_m_high(A.cpu(), B.cpu(), m_splits)
    npu_combine_output = group_gemm_split_m_low(A.npu(), B.npu(), m_splits)
    triton_output = group_gemm_split_m_triton(A, B, m_splits)
    torch_aclnn_output = group_gemm_split_m_aclnn(A, B, m_splits).to(torch.float32)

    print(f"torch_aclnn_output:{torch_aclnn_output}")
    print(f"triton_output:{triton_output}")
    print(f"true_output:{true_output}")
    assert torch.allclose(triton_output, torch_aclnn_output, 1e-3, 1e-3)
    # print("compare with asc:", torch.allclose(triton_output, torch_aclnn_output, 1e-2, 1e-2))
    # assert compare_cv(true_output.npu(), npu_combine_output.npu(), triton_output.npu())
