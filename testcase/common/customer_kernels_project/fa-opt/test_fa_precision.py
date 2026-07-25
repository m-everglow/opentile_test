import torch
import sys

from flash_attention_gpu import FlashAttentionFunc as FlashAttentionFuncGPU
from flash_attention_maskin_0512 import FlashAttentionFunc as FlashAttentionFuncMaskin
from flash_attention_dev import FlashAttentionFunc

def print_mismatched_positions(tensor1, tensor2, atol=1e-3, rtol=1e-2, max_print=20):
    """
    打印两个tensor不相等的位置及其值
    :param tensor1: 第一个tensor
    :param tensor2: 第二个tensor
    :param atol: 绝对误差容差
    :param rtol: 相对误差容差
    :param max_print: 最大打印不匹配位置数量
    """
    # 计算绝对误差
    abs_diff = torch.abs(tensor1 - tensor2)
    
    # 计算相对误差（避免除以0）
    denom = torch.abs(tensor2) + atol
    rel_diff = abs_diff / denom
    
    # 找到不匹配的位置
    mismatch_mask = (abs_diff > atol) | (rel_diff > rtol)
    
    # 获取不匹配位置的索引
    mismatch_indices = torch.where(mismatch_mask)
    
    num_mismatches = len(mismatch_indices[0])
    print(f"\n=== 不匹配位置统计 ===")
    print(f"总元素数: {tensor1.numel()}")
    print(f"不匹配元素数: {num_mismatches}")
    print(f"不匹配比例: {num_mismatches / tensor1.numel() * 100:.2f}%")
    
    if num_mismatches > 0:
        print(f"\n=== 前{min(max_print, num_mismatches)}个不匹配位置 ===")
        for i in range(min(max_print, num_mismatches)):
            idx = tuple(idx[i].item() for idx in mismatch_indices)
            val1 = tensor1[idx].item()
            val2 = tensor2[idx].item()
            diff = abs(val1 - val2)
            print(f"位置 {idx}: triton={val1:.6f}, native={val2:.6f}, 绝对误差={diff:.6f}")
    
    return num_mismatches


def torch_native_attention(q, k, v, scale, mask_fn, q_attn_arg=None, k_attn_arg=None, cu_seqlens_q=None, cu_seqlens_k=None):
    """
    PyTorch原生实现的Attention，与Triton kernel完全一致
    支持多batch且每个batch序列长度不等
    :param q: [total_q_seq, q_head, qk_dim]
    :param k: [total_kv_seq, kv_head, qk_dim]
    :param v: [total_kv_seq, kv_head, v_dim]
    :param scale: 缩放因子
    :param mask_fn: 掩码函数类型
    :param q_attn_arg: query的attn_arg
    :param k_attn_arg: key的attn_arg
    :param cu_seqlens_q: query的序列长度累积和 [batch+1]
    :param cu_seqlens_k: key的序列长度累积和 [batch+1]
    :return: output [total_q_seq, q_head, v_dim]
    """
    q_len, q_head, qk_dim = q.shape
    k_len, kv_head, v_dim = v.shape
    
    # 如果q_head != kv_head，需要处理grouped query attention
    if q_head != kv_head:
        head_group = q_head // kv_head
        k = k.unsqueeze(1).repeat(1, head_group, 1, 1).reshape(k_len, q_head, qk_dim)
        v = v.unsqueeze(1).repeat(1, head_group, 1, 1).reshape(k_len, q_head, v_dim)
    
    # 创建索引偏移
    q_offset = torch.arange(q_len, device=q.device)
    k_offset = torch.arange(k_len, device=k.device)
    
    # 计算注意力分数
    k_transposed = k.transpose(0, 1).transpose(1, 2)  # [q_head, qk_dim, k_len]
    q_reshaped = q.transpose(0, 1)  # [q_head, q_len, qk_dim]
    scores = torch.matmul(q_reshaped, k_transposed) * scale  # [q_head, q_len, k_len]
    
    # 应用batch内的序列边界掩码（防止跨batch attention）
    if cu_seqlens_q is not None and cu_seqlens_k is not None:
        batch_size = len(cu_seqlens_q) - 1
        batch_mask = torch.zeros(q_len, k_len, device=q.device, dtype=torch.bool)
        
        for b in range(batch_size):
            q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b+1].item()
            k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b+1].item()
            batch_mask[q_start:q_end, k_start:k_end] = True
        
        # 初始时将所有位置设为无效，然后根据batch_mask设置有效位置
        scores = scores.masked_fill(~batch_mask.unsqueeze(0), float('-inf'))
    
    # 应用掩码（完全按照Triton kernel的mask_fn实现）
    if mask_fn == 1 or mask_fn == 2:
        # 根据Triton kernel中的mask_fn实现
        tril_causal = q_offset[:, None] >= k_offset[None, :]  # q >= k
        triu_causal = q_offset[:, None] <= k_offset[None, :]  # q <= k
        
        if mask_fn == 1:
            # TYPE 1: (triu_causal & ((q_attn_arg == k_attn_arg) | (k_attn_arg == 0))) | diagonal
            causal_mask = triu_causal
        else:
            # TYPE 2: (tril_causal & ((q_attn_arg == k_attn_arg) | (k_attn_arg == 0))) | diagonal
            causal_mask = tril_causal
        
        # 应用attn_arg条件
        if q_attn_arg is not None and k_attn_arg is not None:
            arg_mask = (q_attn_arg[:, None] == k_attn_arg[None, :]) | (k_attn_arg[None, :] == 0)
            mask = causal_mask & arg_mask
        else:
            mask = causal_mask
        
        # 对角线总是保留
        diag_mask = q_offset[:, None] == k_offset[None, :]
        final_mask = mask | diag_mask
        
        # 对分数应用掩码
        scores = scores.masked_fill(~final_mask.unsqueeze(0), float('-inf'))
    
    # 计算softmax
    attn_weights = torch.softmax(scores, dim=-1)  # [q_head, q_len, k_len]
    
    # 乘以v
    v_reshaped = v.transpose(0, 1)  # [q_head, k_len, v_dim]
    output = torch.matmul(attn_weights, v_reshaped)  # [q_head, q_len, v_dim]
    
    # 转置回 [q_len, q_head, v_dim]
    output = output.transpose(0, 1)
    
    return output

def compare(triton_output, native_output, q):
    # 计算误差
    max_abs_error = torch.max(torch.abs(triton_output - native_output)).item()
    mean_abs_error = torch.mean(torch.abs(triton_output - native_output)).item()
    max_rel_error = torch.max(torch.abs(triton_output - native_output) / (torch.abs(native_output) + 1e-6)).item()
    
    print(f"Max Absolute Error: {max_abs_error:.6e}")
    print(f"Mean Absolute Error: {mean_abs_error:.6e}")
    print(f"Max Relative Error: {max_rel_error:.6e}")
    
    # 检查是否通过
    atol = 1e-2
    rtol = 1e-2
    passed = torch.allclose(triton_output, native_output, atol=atol, rtol=rtol)
    print(f"Test {q.shape=}, {'PASSED' if passed else 'FAILED'} (atol={atol}, rtol={rtol})")
    if not passed:
        # 打印不匹配位置
        # print_mismatched_positions(triton_output, native_output, atol=atol, rtol=rtol)
        
        passed = torch.allclose(triton_output[:64,...], native_output[:64,...], atol=atol, rtol=rtol)
        if not passed:
            # 额外打印[:64]和[64:]的统计
            print(f"\n=== [:64] 区域 ===")
            print_mismatched_positions(triton_output[:64,...], native_output[:64,...], atol=atol, rtol=rtol)
        passed = torch.allclose(triton_output[64:,...], native_output[64:,...], atol=atol, rtol=rtol)
        if not passed:
            print(f"\n=== [64:] 区域 ===")
            print_mismatched_positions(triton_output[64:,...], native_output[64:,...], atol=atol, rtol=rtol)


def test_flash_attention():
    """测试FlashAttention的精度对比（支持多batch且每个batch序列长度不等）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'npu')
    print(f"Testing on device: {device}")
    
    # 测试配置 - 支持多batch且每个batch序列长度不等
    # (q_seq_lens, k_seq_lens, q_head, kv_head, qk_dim, v_dim, mask_fn, sparse_opt)
    # q_seq_lens和k_seq_lens是列表，每个元素代表一个batch的序列长度
    test_cases = [
        # 单batch测试（基准）
        #([1100], [1100], 1, 1, 64, 64, 1, False),
        # 多batch等长测试
        #([512, 512], [512, 512], 8, 8, 64, 64, 1, False),
        # 多batch不等长测试 - 递增序列
        # ([128, 256, 512], [128, 256, 512], 16, 16, 64, 64, 1, False), # GQA 
        # 多batch不等长测试 - 递减序列
        #([512, 256, 128], [512, 256, 128], 8, 8, 64, 64, 1, False),
        # 多batch不等长测试 - 随机长度
        ([384, 640, 192, 448], [384, 640, 192, 448], 8, 8, 64, 64, 1, False),
        #([384, 640, 192, 448], [400, 800, 218, 500], 8, 8, 64, 64, 1, False),
    ]
    
    for i, (q_seq_lens, k_seq_lens, q_head, kv_head, qk_dim, v_dim, mask_fn, sparse_opt) in enumerate(test_cases):
        batch_size = len(q_seq_lens)
        total_q_seq = sum(q_seq_lens)
        total_k_seq = sum(k_seq_lens)
        max_q_seq = max(q_seq_lens)
        max_k_seq = max(k_seq_lens)
        
        print(f"\n=== Test Case {i+1} ===")
        print(f"Config: batch={batch_size}, q_seq_lens={q_seq_lens}, k_seq_lens={k_seq_lens}")
        print(f"        q_head={q_head}, kv_head={kv_head}, qk_dim={qk_dim}, v_dim={v_dim}, mask_fn={mask_fn}")
        print(f"        total_q_seq={total_q_seq}, total_k_seq={total_k_seq}, max_q_seq={max_q_seq}, max_k_seq={max_k_seq}")
        
        # 生成随机输入
        torch.manual_seed(42)
        q = torch.randn(total_q_seq, q_head, qk_dim, device=device, dtype=torch.float16)
        k = torch.randn(total_k_seq, kv_head, qk_dim, device=device, dtype=torch.float16)
        v = torch.randn(total_k_seq, kv_head, v_dim, device=device, dtype=torch.float16)
        
        # 生成cu_seqlens（累积和）
        cu_seqlens_q = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
        cu_seqlens_k = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
        for b in range(batch_size):
            cu_seqlens_q[b+1] = cu_seqlens_q[b] + q_seq_lens[b]
            cu_seqlens_k[b+1] = cu_seqlens_k[b] + k_seq_lens[b]
        
        print(f"        cu_seqlens_q={cu_seqlens_q.cpu().tolist()}")
        print(f"        cu_seqlens_k={cu_seqlens_k.cpu().tolist()}")
        
        # 生成attn_arg（用于稀疏注意力）
        q_attn_arg = torch.zeros(total_q_seq, device=device, dtype=torch.int32)
        k_attn_arg = torch.zeros(total_k_seq, device=device, dtype=torch.int32)
        
        # 计算scale
        scale = 1.0 / (qk_dim ** 0.5)
        
        # Triton实现
        triton_output = FlashAttentionFuncGPU.apply(
            q, k, v,
            q_attn_arg, k_attn_arg,
            cu_seqlens_q, cu_seqlens_k,
            max_q_seq, max_k_seq,
            scale, mask_fn, sparse_opt
        )
        
        # PyTorch原生实现（支持不等长序列）
        native_output = torch_native_attention(q, k, v, scale, mask_fn, q_attn_arg, k_attn_arg, cu_seqlens_q, cu_seqlens_k)
        
        compare(triton_output, native_output, q)
        print("-----------------GPU triton done----------------")

        triton_output2 = FlashAttentionFuncMaskin.apply(
            q, k, v,
            q_attn_arg, k_attn_arg,
            cu_seqlens_q, cu_seqlens_k,
            max_q_seq, max_k_seq,
            scale, mask_fn, sparse_opt
        )
        compare(triton_output2, native_output, q)
        print("-----------------Maskin triton done----------------")

        triton_output3 = FlashAttentionFunc.apply(
            q, k, v,
            q_attn_arg, k_attn_arg,
            cu_seqlens_q, cu_seqlens_k,
            max_q_seq, max_k_seq,
            scale, mask_fn, sparse_opt
        )
        # compare(triton_output, triton_output3, q)
        compare(triton_output3, native_output, q)
        print("-----------------MaskOut triton done----------------")


if __name__ == "__main__":
    test_flash_attention()
