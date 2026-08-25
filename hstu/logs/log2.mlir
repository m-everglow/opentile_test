============================= test session starts ==============================
platform linux -- Python 3.11.0, pytest-8.4.1, pluggy-1.6.0
rootdir: /data/m00953828/hstu
plugins: cov-6.2.1, metadata-3.1.1, html-4.1.1, hypothesis-6.70.1, anyio-4.14.2, xdist-3.8.0, repeat-0.9.4
collected 5100 items / 5000 deselected / 100 selected

hstu/test/test_hstu_fwd_jagged.py .F.......FFF.....F.F....F..F.......... [ 38%]
......F..................................F...FF........FF.F..F           [100%]

=================================== FAILURES ===================================
_________ test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-2-100] _________

batch_size = 2048, num_heads_q = 2, num_heads_k = 2, seq_len_q = 64
seq_len_k = 32, attention_dim = 256, linear_dim = 256, dtype_str = 'fp8'

    @pytest.mark.parametrize(
        "batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim",
        [
            (2, 8, 4, 256, 64, 64, 64), # GQA
            (2, 8, 4, 64, 64, 64, 64),
            (16, 4, 4, 32, 499, 64, 32), # (v_d != qk_d)
            (5, 8, 4, 128, 256, 96, 128), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 64), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 80), # (v_d != qk_d)
            (2, 4, 1, 1001, 901, 128, 48), # (v_d != qk_d)
            (16, 4, 4, 501, 1000, 64, 64), # delta_q
            (8, 8, 8, 8000, 8000, 128, 128),
            (8, 8, 8, 8000, 8000, 256, 256),
            (2048, 4, 4, 52, 1000, 64, 64), # delta_q
            (2048, 2, 2, 32, 32, 256, 256),
            (2048, 2, 2, 64, 32, 256, 256),
            (96, 2, 2, 512, 3072, 256, 256), # delta_q
            (96, 2, 2, 512, 3072, 512, 512), # delta_q
            (1, 2, 2, 32, 32, 256, 256),
            (2, 8, 4, 32, 32, 32, 32),
        ],
    )
    @pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp8"])
    def test_triton_matches_golden(
        batch_size: int,
        num_heads_q: int,
        num_heads_k: int,
        seq_len_q: int,
        seq_len_k: int,
        attention_dim: int,
        linear_dim: int,
        dtype_str: str,
    ):
        # ===========prepare input===========
        set_seed(42)
        errors = []
        device = torch.device("npu")
        alpha = 1.0 / (attention_dim ** 0.5)
        type_mapper = {"fp16":torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
        dtype = type_mapper.get(dtype_str)
        q, k, v, seq_offset_q, seq_offset_k = get_or_generate_data(
            batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k,
            attention_dim, linear_dim, device, dtype
        )
    
    
        golden_output = golden_op_exec_high(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
    
        sim_output = golden_op_exec_low(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
        triton_output = triton_hstu_attention_fwd(
            q=q,
            k=k,
            v=v,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
            seq_offsets_q=seq_offset_q,
            seq_offsets_k=seq_offset_k,
            num_context=None,
            num_target=None,
            alpha=alpha,
            silu_scale=1.0 / seq_len_q,
        ).view([-1, num_heads_q, linear_dim]).to(device)
    
        if golden_output is None:
            return
    
>       assert compare_cv(golden_output.npu(), sim_output.npu(), triton_output.npu())
E       AssertionError: assert tensor(False, device='npu:0')
E        +  where tensor(False, device='npu:0') = compare_cv(tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0'), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16))
E        +    where tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0') = npu()
E        +      where npu = tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0').npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu

hstu/test/test_hstu_fwd_jagged.py:326: AssertionError
----------------------------- Captured stdout call -----------------------------

[INFO] Checking data cache: ./data_cache/bs2048_hq2_hk2_sq64_sk32_ad256_ld256_dttorch.float8_e4m3fn.pt
[INFO] Cache hit. Loading data...
[INFO] Data loaded successfully.
[INFO] Input shapes - Q: torch.Size([131072, 2, 256]), K: torch.Size([65536, 2, 256]), V: torch.Size([65536, 2, 256])
[INFO] Device: npu, Dtype: torch.float8_e4m3fn
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
BLOCK_M: 32  BLOCK_N: 384
err_threshold:0.00048828125
mare_npu:0.9999999403953552 mare_gpu:0.018160074949264526
mere_npu:0.00018856054521165788 mere_gpu:0.00030364940175786614
rmse_npu:0.003827436827123165 rmse_gpu:0.0007414165884256363
MARE:55.06584930419922 MERE:0.38617199659347534 RMSE:5.16232967376709
new golden cv result:False
----------------------------- Captured stderr call -----------------------------
[W825 08:06:08.740960096 ToKernelNpu.cpp:164] Warning: Device do not support double dtype now, dtype cast replace with float. (function operator())
________ test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-10-100] _________

batch_size = 2048, num_heads_q = 2, num_heads_k = 2, seq_len_q = 64
seq_len_k = 32, attention_dim = 256, linear_dim = 256, dtype_str = 'fp8'

    @pytest.mark.parametrize(
        "batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim",
        [
            (2, 8, 4, 256, 64, 64, 64), # GQA
            (2, 8, 4, 64, 64, 64, 64),
            (16, 4, 4, 32, 499, 64, 32), # (v_d != qk_d)
            (5, 8, 4, 128, 256, 96, 128), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 64), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 80), # (v_d != qk_d)
            (2, 4, 1, 1001, 901, 128, 48), # (v_d != qk_d)
            (16, 4, 4, 501, 1000, 64, 64), # delta_q
            (8, 8, 8, 8000, 8000, 128, 128),
            (8, 8, 8, 8000, 8000, 256, 256),
            (2048, 4, 4, 52, 1000, 64, 64), # delta_q
            (2048, 2, 2, 32, 32, 256, 256),
            (2048, 2, 2, 64, 32, 256, 256),
            (96, 2, 2, 512, 3072, 256, 256), # delta_q
            (96, 2, 2, 512, 3072, 512, 512), # delta_q
            (1, 2, 2, 32, 32, 256, 256),
            (2, 8, 4, 32, 32, 32, 32),
        ],
    )
    @pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp8"])
    def test_triton_matches_golden(
        batch_size: int,
        num_heads_q: int,
        num_heads_k: int,
        seq_len_q: int,
        seq_len_k: int,
        attention_dim: int,
        linear_dim: int,
        dtype_str: str,
    ):
        # ===========prepare input===========
        set_seed(42)
        errors = []
        device = torch.device("npu")
        alpha = 1.0 / (attention_dim ** 0.5)
        type_mapper = {"fp16":torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
        dtype = type_mapper.get(dtype_str)
        q, k, v, seq_offset_q, seq_offset_k = get_or_generate_data(
            batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k,
            attention_dim, linear_dim, device, dtype
        )
    
    
        golden_output = golden_op_exec_high(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
    
        sim_output = golden_op_exec_low(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
        triton_output = triton_hstu_attention_fwd(
            q=q,
            k=k,
            v=v,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
            seq_offsets_q=seq_offset_q,
            seq_offsets_k=seq_offset_k,
            num_context=None,
            num_target=None,
            alpha=alpha,
            silu_scale=1.0 / seq_len_q,
        ).view([-1, num_heads_q, linear_dim]).to(device)
    
        if golden_output is None:
            return
    
>       assert compare_cv(golden_output.npu(), sim_output.npu(), triton_output.npu())
E       AssertionError: assert tensor(False, device='npu:0')
E        +  where tensor(False, device='npu:0') = compare_cv(tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0'), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16))
E        +    where tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0') = npu()
E        +      where npu = tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0').npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu

hstu/test/test_hstu_fwd_jagged.py:326: AssertionError
----------------------------- Captured stdout call -----------------------------

[INFO] Checking data cache: ./data_cache/bs2048_hq2_hk2_sq64_sk32_ad256_ld256_dttorch.float8_e4m3fn.pt
[INFO] Cache hit. Loading data...
[INFO] Data loaded successfully.
[INFO] Input shapes - Q: torch.Size([131072, 2, 256]), K: torch.Size([65536, 2, 256]), V: torch.Size([65536, 2, 256])
[INFO] Device: npu, Dtype: torch.float8_e4m3fn
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
BLOCK_M: 32  BLOCK_N: 384
err_threshold:0.00048828125
mare_npu:0.9999998807907104 mare_gpu:0.018160074949264526
mere_npu:0.00018856048700399697 mere_gpu:0.00030364940175786614
rmse_npu:0.003923055250197649 rmse_gpu:0.0007414165884256363
MARE:55.06584548950195 MERE:0.3861718773841858 RMSE:5.29129695892334
new golden cv result:False
________ test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-11-100] _________

batch_size = 2048, num_heads_q = 2, num_heads_k = 2, seq_len_q = 64
seq_len_k = 32, attention_dim = 256, linear_dim = 256, dtype_str = 'fp8'

    @pytest.mark.parametrize(
        "batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim",
        [
            (2, 8, 4, 256, 64, 64, 64), # GQA
            (2, 8, 4, 64, 64, 64, 64),
            (16, 4, 4, 32, 499, 64, 32), # (v_d != qk_d)
            (5, 8, 4, 128, 256, 96, 128), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 64), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 80), # (v_d != qk_d)
            (2, 4, 1, 1001, 901, 128, 48), # (v_d != qk_d)
            (16, 4, 4, 501, 1000, 64, 64), # delta_q
            (8, 8, 8, 8000, 8000, 128, 128),
            (8, 8, 8, 8000, 8000, 256, 256),
            (2048, 4, 4, 52, 1000, 64, 64), # delta_q
            (2048, 2, 2, 32, 32, 256, 256),
            (2048, 2, 2, 64, 32, 256, 256),
            (96, 2, 2, 512, 3072, 256, 256), # delta_q
            (96, 2, 2, 512, 3072, 512, 512), # delta_q
            (1, 2, 2, 32, 32, 256, 256),
            (2, 8, 4, 32, 32, 32, 32),
        ],
    )
    @pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp8"])
    def test_triton_matches_golden(
        batch_size: int,
        num_heads_q: int,
        num_heads_k: int,
        seq_len_q: int,
        seq_len_k: int,
        attention_dim: int,
        linear_dim: int,
        dtype_str: str,
    ):
        # ===========prepare input===========
        set_seed(42)
        errors = []
        device = torch.device("npu")
        alpha = 1.0 / (attention_dim ** 0.5)
        type_mapper = {"fp16":torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
        dtype = type_mapper.get(dtype_str)
        q, k, v, seq_offset_q, seq_offset_k = get_or_generate_data(
            batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k,
            attention_dim, linear_dim, device, dtype
        )
    
    
        golden_output = golden_op_exec_high(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
    
        sim_output = golden_op_exec_low(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
        triton_output = triton_hstu_attention_fwd(
            q=q,
            k=k,
            v=v,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
            seq_offsets_q=seq_offset_q,
            seq_offsets_k=seq_offset_k,
            num_context=None,
            num_target=None,
            alpha=alpha,
            silu_scale=1.0 / seq_len_q,
        ).view([-1, num_heads_q, linear_dim]).to(device)
    
        if golden_output is None:
            return
    
>       assert compare_cv(golden_output.npu(), sim_output.npu(), triton_output.npu())
E       AssertionError: assert tensor(False, device='npu:0')
E        +  where tensor(False, device='npu:0') = compare_cv(tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0'), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16))
E        +    where tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0') = npu()
E        +      where npu = tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0').npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu

hstu/test/test_hstu_fwd_jagged.py:326: AssertionError
----------------------------- Captured stdout call -----------------------------

[INFO] Checking data cache: ./data_cache/bs2048_hq2_hk2_sq64_sk32_ad256_ld256_dttorch.float8_e4m3fn.pt
[INFO] Cache hit. Loading data...
[INFO] Data loaded successfully.
[INFO] Input shapes - Q: torch.Size([131072, 2, 256]), K: torch.Size([65536, 2, 256]), V: torch.Size([65536, 2, 256])
[INFO] Device: npu, Dtype: torch.float8_e4m3fn
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
BLOCK_M: 32  BLOCK_N: 384
err_threshold:0.00048828125
mare_npu:0.9999999403953552 mare_gpu:0.018160074949264526
mere_npu:0.00020381654030643404 mere_gpu:0.00030364940175786614
rmse_npu:0.005483211483806372 rmse_gpu:0.0007414165884256363
MARE:55.06584930419922 MERE:0.4174162745475769 RMSE:7.395587921142578
new golden cv result:False
________ test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-12-100] _________

batch_size = 2048, num_heads_q = 2, num_heads_k = 2, seq_len_q = 64
seq_len_k = 32, attention_dim = 256, linear_dim = 256, dtype_str = 'fp8'

    @pytest.mark.parametrize(
        "batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim",
        [
            (2, 8, 4, 256, 64, 64, 64), # GQA
            (2, 8, 4, 64, 64, 64, 64),
            (16, 4, 4, 32, 499, 64, 32), # (v_d != qk_d)
            (5, 8, 4, 128, 256, 96, 128), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 64), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 80), # (v_d != qk_d)
            (2, 4, 1, 1001, 901, 128, 48), # (v_d != qk_d)
            (16, 4, 4, 501, 1000, 64, 64), # delta_q
            (8, 8, 8, 8000, 8000, 128, 128),
            (8, 8, 8, 8000, 8000, 256, 256),
            (2048, 4, 4, 52, 1000, 64, 64), # delta_q
            (2048, 2, 2, 32, 32, 256, 256),
            (2048, 2, 2, 64, 32, 256, 256),
            (96, 2, 2, 512, 3072, 256, 256), # delta_q
            (96, 2, 2, 512, 3072, 512, 512), # delta_q
            (1, 2, 2, 32, 32, 256, 256),
            (2, 8, 4, 32, 32, 32, 32),
        ],
    )
    @pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp8"])
    def test_triton_matches_golden(
        batch_size: int,
        num_heads_q: int,
        num_heads_k: int,
        seq_len_q: int,
        seq_len_k: int,
        attention_dim: int,
        linear_dim: int,
        dtype_str: str,
    ):
        # ===========prepare input===========
        set_seed(42)
        errors = []
        device = torch.device("npu")
        alpha = 1.0 / (attention_dim ** 0.5)
        type_mapper = {"fp16":torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
        dtype = type_mapper.get(dtype_str)
        q, k, v, seq_offset_q, seq_offset_k = get_or_generate_data(
            batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k,
            attention_dim, linear_dim, device, dtype
        )
    
    
        golden_output = golden_op_exec_high(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
    
        sim_output = golden_op_exec_low(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
        triton_output = triton_hstu_attention_fwd(
            q=q,
            k=k,
            v=v,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
            seq_offsets_q=seq_offset_q,
            seq_offsets_k=seq_offset_k,
            num_context=None,
            num_target=None,
            alpha=alpha,
            silu_scale=1.0 / seq_len_q,
        ).view([-1, num_heads_q, linear_dim]).to(device)
    
        if golden_output is None:
            return
    
>       assert compare_cv(golden_output.npu(), sim_output.npu(), triton_output.npu())
E       AssertionError: assert tensor(False, device='npu:0')
E        +  where tensor(False, device='npu:0') = compare_cv(tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0'), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16))
E        +    where tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0') = npu()
E        +      where npu = tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0').npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu

hstu/test/test_hstu_fwd_jagged.py:326: AssertionError
----------------------------- Captured stdout call -----------------------------

[INFO] Checking data cache: ./data_cache/bs2048_hq2_hk2_sq64_sk32_ad256_ld256_dttorch.float8_e4m3fn.pt
[INFO] Cache hit. Loading data...
[INFO] Data loaded successfully.
[INFO] Input shapes - Q: torch.Size([131072, 2, 256]), K: torch.Size([65536, 2, 256]), V: torch.Size([65536, 2, 256])
[INFO] Device: npu, Dtype: torch.float8_e4m3fn
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
BLOCK_M: 32  BLOCK_N: 384
err_threshold:0.00048828125
mare_npu:0.9999999403953552 mare_gpu:0.018160074949264526
mere_npu:0.0002305146335856989 mere_gpu:0.00030364940175786614
rmse_npu:0.007424802519381046 rmse_gpu:0.0007414165884256363
MARE:55.06584930419922 MERE:0.47209396958351135 RMSE:10.0143461227417
new golden cv result:False
________ test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-18-100] _________

batch_size = 2048, num_heads_q = 2, num_heads_k = 2, seq_len_q = 64
seq_len_k = 32, attention_dim = 256, linear_dim = 256, dtype_str = 'fp8'

    @pytest.mark.parametrize(
        "batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim",
        [
            (2, 8, 4, 256, 64, 64, 64), # GQA
            (2, 8, 4, 64, 64, 64, 64),
            (16, 4, 4, 32, 499, 64, 32), # (v_d != qk_d)
            (5, 8, 4, 128, 256, 96, 128), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 64), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 80), # (v_d != qk_d)
            (2, 4, 1, 1001, 901, 128, 48), # (v_d != qk_d)
            (16, 4, 4, 501, 1000, 64, 64), # delta_q
            (8, 8, 8, 8000, 8000, 128, 128),
            (8, 8, 8, 8000, 8000, 256, 256),
            (2048, 4, 4, 52, 1000, 64, 64), # delta_q
            (2048, 2, 2, 32, 32, 256, 256),
            (2048, 2, 2, 64, 32, 256, 256),
            (96, 2, 2, 512, 3072, 256, 256), # delta_q
            (96, 2, 2, 512, 3072, 512, 512), # delta_q
            (1, 2, 2, 32, 32, 256, 256),
            (2, 8, 4, 32, 32, 32, 32),
        ],
    )
    @pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp8"])
    def test_triton_matches_golden(
        batch_size: int,
        num_heads_q: int,
        num_heads_k: int,
        seq_len_q: int,
        seq_len_k: int,
        attention_dim: int,
        linear_dim: int,
        dtype_str: str,
    ):
        # ===========prepare input===========
        set_seed(42)
        errors = []
        device = torch.device("npu")
        alpha = 1.0 / (attention_dim ** 0.5)
        type_mapper = {"fp16":torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
        dtype = type_mapper.get(dtype_str)
        q, k, v, seq_offset_q, seq_offset_k = get_or_generate_data(
            batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k,
            attention_dim, linear_dim, device, dtype
        )
    
    
        golden_output = golden_op_exec_high(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
    
        sim_output = golden_op_exec_low(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
        triton_output = triton_hstu_attention_fwd(
            q=q,
            k=k,
            v=v,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
            seq_offsets_q=seq_offset_q,
            seq_offsets_k=seq_offset_k,
            num_context=None,
            num_target=None,
            alpha=alpha,
            silu_scale=1.0 / seq_len_q,
        ).view([-1, num_heads_q, linear_dim]).to(device)
    
        if golden_output is None:
            return
    
>       assert compare_cv(golden_output.npu(), sim_output.npu(), triton_output.npu())
E       AssertionError: assert tensor(False, device='npu:0')
E        +  where tensor(False, device='npu:0') = compare_cv(tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0'), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16))
E        +    where tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0') = npu()
E        +      where npu = tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0').npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu

hstu/test/test_hstu_fwd_jagged.py:326: AssertionError
----------------------------- Captured stdout call -----------------------------

[INFO] Checking data cache: ./data_cache/bs2048_hq2_hk2_sq64_sk32_ad256_ld256_dttorch.float8_e4m3fn.pt
[INFO] Cache hit. Loading data...
[INFO] Data loaded successfully.
[INFO] Input shapes - Q: torch.Size([131072, 2, 256]), K: torch.Size([65536, 2, 256]), V: torch.Size([65536, 2, 256])
[INFO] Device: npu, Dtype: torch.float8_e4m3fn
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
BLOCK_M: 32  BLOCK_N: 384
err_threshold:0.00048828125
mare_npu:0.9999998807907104 mare_gpu:0.018160074949264526
mere_npu:0.00018856053065974265 mere_gpu:0.00030364940175786614
rmse_npu:0.003849690081551671 rmse_gpu:0.0007414165884256363
MARE:55.06584548950195 MERE:0.38617196679115295 RMSE:5.1923441886901855
new golden cv result:False
________ test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-20-100] _________

batch_size = 2048, num_heads_q = 2, num_heads_k = 2, seq_len_q = 64
seq_len_k = 32, attention_dim = 256, linear_dim = 256, dtype_str = 'fp8'

    @pytest.mark.parametrize(
        "batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim",
        [
            (2, 8, 4, 256, 64, 64, 64), # GQA
            (2, 8, 4, 64, 64, 64, 64),
            (16, 4, 4, 32, 499, 64, 32), # (v_d != qk_d)
            (5, 8, 4, 128, 256, 96, 128), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 64), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 80), # (v_d != qk_d)
            (2, 4, 1, 1001, 901, 128, 48), # (v_d != qk_d)
            (16, 4, 4, 501, 1000, 64, 64), # delta_q
            (8, 8, 8, 8000, 8000, 128, 128),
            (8, 8, 8, 8000, 8000, 256, 256),
            (2048, 4, 4, 52, 1000, 64, 64), # delta_q
            (2048, 2, 2, 32, 32, 256, 256),
            (2048, 2, 2, 64, 32, 256, 256),
            (96, 2, 2, 512, 3072, 256, 256), # delta_q
            (96, 2, 2, 512, 3072, 512, 512), # delta_q
            (1, 2, 2, 32, 32, 256, 256),
            (2, 8, 4, 32, 32, 32, 32),
        ],
    )
    @pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp8"])
    def test_triton_matches_golden(
        batch_size: int,
        num_heads_q: int,
        num_heads_k: int,
        seq_len_q: int,
        seq_len_k: int,
        attention_dim: int,
        linear_dim: int,
        dtype_str: str,
    ):
        # ===========prepare input===========
        set_seed(42)
        errors = []
        device = torch.device("npu")
        alpha = 1.0 / (attention_dim ** 0.5)
        type_mapper = {"fp16":torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
        dtype = type_mapper.get(dtype_str)
        q, k, v, seq_offset_q, seq_offset_k = get_or_generate_data(
            batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k,
            attention_dim, linear_dim, device, dtype
        )
    
    
        golden_output = golden_op_exec_high(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
    
        sim_output = golden_op_exec_low(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
        triton_output = triton_hstu_attention_fwd(
            q=q,
            k=k,
            v=v,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
            seq_offsets_q=seq_offset_q,
            seq_offsets_k=seq_offset_k,
            num_context=None,
            num_target=None,
            alpha=alpha,
            silu_scale=1.0 / seq_len_q,
        ).view([-1, num_heads_q, linear_dim]).to(device)
    
        if golden_output is None:
            return
    
>       assert compare_cv(golden_output.npu(), sim_output.npu(), triton_output.npu())
E       AssertionError: assert tensor(False, device='npu:0')
E        +  where tensor(False, device='npu:0') = compare_cv(tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0'), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16))
E        +    where tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0') = npu()
E        +      where npu = tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0').npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu

hstu/test/test_hstu_fwd_jagged.py:326: AssertionError
----------------------------- Captured stdout call -----------------------------

[INFO] Checking data cache: ./data_cache/bs2048_hq2_hk2_sq64_sk32_ad256_ld256_dttorch.float8_e4m3fn.pt
[INFO] Cache hit. Loading data...
[INFO] Data loaded successfully.
[INFO] Input shapes - Q: torch.Size([131072, 2, 256]), K: torch.Size([65536, 2, 256]), V: torch.Size([65536, 2, 256])
[INFO] Device: npu, Dtype: torch.float8_e4m3fn
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
BLOCK_M: 32  BLOCK_N: 384
err_threshold:0.00048828125
mare_npu:0.9999998807907104 mare_gpu:0.018160074949264526
mere_npu:0.00017521140398457646 mere_gpu:0.00030364940175786614
rmse_npu:0.001353772939182818 rmse_gpu:0.0007414165884256363
MARE:55.06584548950195 MERE:0.3588329553604126 RMSE:1.825927495956421
new golden cv result:False
________ test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-25-100] _________

batch_size = 2048, num_heads_q = 2, num_heads_k = 2, seq_len_q = 64
seq_len_k = 32, attention_dim = 256, linear_dim = 256, dtype_str = 'fp8'

    @pytest.mark.parametrize(
        "batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim",
        [
            (2, 8, 4, 256, 64, 64, 64), # GQA
            (2, 8, 4, 64, 64, 64, 64),
            (16, 4, 4, 32, 499, 64, 32), # (v_d != qk_d)
            (5, 8, 4, 128, 256, 96, 128), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 64), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 80), # (v_d != qk_d)
            (2, 4, 1, 1001, 901, 128, 48), # (v_d != qk_d)
            (16, 4, 4, 501, 1000, 64, 64), # delta_q
            (8, 8, 8, 8000, 8000, 128, 128),
            (8, 8, 8, 8000, 8000, 256, 256),
            (2048, 4, 4, 52, 1000, 64, 64), # delta_q
            (2048, 2, 2, 32, 32, 256, 256),
            (2048, 2, 2, 64, 32, 256, 256),
            (96, 2, 2, 512, 3072, 256, 256), # delta_q
            (96, 2, 2, 512, 3072, 512, 512), # delta_q
            (1, 2, 2, 32, 32, 256, 256),
            (2, 8, 4, 32, 32, 32, 32),
        ],
    )
    @pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp8"])
    def test_triton_matches_golden(
        batch_size: int,
        num_heads_q: int,
        num_heads_k: int,
        seq_len_q: int,
        seq_len_k: int,
        attention_dim: int,
        linear_dim: int,
        dtype_str: str,
    ):
        # ===========prepare input===========
        set_seed(42)
        errors = []
        device = torch.device("npu")
        alpha = 1.0 / (attention_dim ** 0.5)
        type_mapper = {"fp16":torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
        dtype = type_mapper.get(dtype_str)
        q, k, v, seq_offset_q, seq_offset_k = get_or_generate_data(
            batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k,
            attention_dim, linear_dim, device, dtype
        )
    
    
        golden_output = golden_op_exec_high(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
    
        sim_output = golden_op_exec_low(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
        triton_output = triton_hstu_attention_fwd(
            q=q,
            k=k,
            v=v,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
            seq_offsets_q=seq_offset_q,
            seq_offsets_k=seq_offset_k,
            num_context=None,
            num_target=None,
            alpha=alpha,
            silu_scale=1.0 / seq_len_q,
        ).view([-1, num_heads_q, linear_dim]).to(device)
    
        if golden_output is None:
            return
    
>       assert compare_cv(golden_output.npu(), sim_output.npu(), triton_output.npu())
E       AssertionError: assert tensor(False, device='npu:0')
E        +  where tensor(False, device='npu:0') = compare_cv(tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0'), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16))
E        +    where tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0') = npu()
E        +      where npu = tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0').npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu

hstu/test/test_hstu_fwd_jagged.py:326: AssertionError
----------------------------- Captured stdout call -----------------------------

[INFO] Checking data cache: ./data_cache/bs2048_hq2_hk2_sq64_sk32_ad256_ld256_dttorch.float8_e4m3fn.pt
[INFO] Cache hit. Loading data...
[INFO] Data loaded successfully.
[INFO] Input shapes - Q: torch.Size([131072, 2, 256]), K: torch.Size([65536, 2, 256]), V: torch.Size([65536, 2, 256])
[INFO] Device: npu, Dtype: torch.float8_e4m3fn
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
BLOCK_M: 32  BLOCK_N: 384
err_threshold:0.00048828125
mare_npu:0.9999998807907104 mare_gpu:0.018160074949264526
mere_npu:0.00018856063252314925 mere_gpu:0.00030364940175786614
rmse_npu:0.0037880234885960817 rmse_gpu:0.0007414165884256363
MARE:55.06584548950195 MERE:0.38617217540740967 RMSE:5.109169960021973
new golden cv result:False
________ test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-28-100] _________

batch_size = 2048, num_heads_q = 2, num_heads_k = 2, seq_len_q = 64
seq_len_k = 32, attention_dim = 256, linear_dim = 256, dtype_str = 'fp8'

    @pytest.mark.parametrize(
        "batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim",
        [
            (2, 8, 4, 256, 64, 64, 64), # GQA
            (2, 8, 4, 64, 64, 64, 64),
            (16, 4, 4, 32, 499, 64, 32), # (v_d != qk_d)
            (5, 8, 4, 128, 256, 96, 128), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 64), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 80), # (v_d != qk_d)
            (2, 4, 1, 1001, 901, 128, 48), # (v_d != qk_d)
            (16, 4, 4, 501, 1000, 64, 64), # delta_q
            (8, 8, 8, 8000, 8000, 128, 128),
            (8, 8, 8, 8000, 8000, 256, 256),
            (2048, 4, 4, 52, 1000, 64, 64), # delta_q
            (2048, 2, 2, 32, 32, 256, 256),
            (2048, 2, 2, 64, 32, 256, 256),
            (96, 2, 2, 512, 3072, 256, 256), # delta_q
            (96, 2, 2, 512, 3072, 512, 512), # delta_q
            (1, 2, 2, 32, 32, 256, 256),
            (2, 8, 4, 32, 32, 32, 32),
        ],
    )
    @pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp8"])
    def test_triton_matches_golden(
        batch_size: int,
        num_heads_q: int,
        num_heads_k: int,
        seq_len_q: int,
        seq_len_k: int,
        attention_dim: int,
        linear_dim: int,
        dtype_str: str,
    ):
        # ===========prepare input===========
        set_seed(42)
        errors = []
        device = torch.device("npu")
        alpha = 1.0 / (attention_dim ** 0.5)
        type_mapper = {"fp16":torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
        dtype = type_mapper.get(dtype_str)
        q, k, v, seq_offset_q, seq_offset_k = get_or_generate_data(
            batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k,
            attention_dim, linear_dim, device, dtype
        )
    
    
        golden_output = golden_op_exec_high(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
    
        sim_output = golden_op_exec_low(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
        triton_output = triton_hstu_attention_fwd(
            q=q,
            k=k,
            v=v,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
            seq_offsets_q=seq_offset_q,
            seq_offsets_k=seq_offset_k,
            num_context=None,
            num_target=None,
            alpha=alpha,
            silu_scale=1.0 / seq_len_q,
        ).view([-1, num_heads_q, linear_dim]).to(device)
    
        if golden_output is None:
            return
    
>       assert compare_cv(golden_output.npu(), sim_output.npu(), triton_output.npu())
E       AssertionError: assert tensor(False, device='npu:0')
E        +  where tensor(False, device='npu:0') = compare_cv(tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0'), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16))
E        +    where tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0') = npu()
E        +      where npu = tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0').npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu

hstu/test/test_hstu_fwd_jagged.py:326: AssertionError
----------------------------- Captured stdout call -----------------------------

[INFO] Checking data cache: ./data_cache/bs2048_hq2_hk2_sq64_sk32_ad256_ld256_dttorch.float8_e4m3fn.pt
[INFO] Cache hit. Loading data...
[INFO] Data loaded successfully.
[INFO] Input shapes - Q: torch.Size([131072, 2, 256]), K: torch.Size([65536, 2, 256]), V: torch.Size([65536, 2, 256])
[INFO] Device: npu, Dtype: torch.float8_e4m3fn
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
BLOCK_M: 32  BLOCK_N: 384
err_threshold:0.00048828125
mare_npu:0.9999998807907104 mare_gpu:0.018160074949264526
mere_npu:0.0001885606034193188 mere_gpu:0.00030364940175786614
rmse_npu:0.0037682612892240286 rmse_gpu:0.0007414165884256363
MARE:55.06584548950195 MERE:0.3861721158027649 RMSE:5.082515716552734
new golden cv result:False
________ test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-45-100] _________

batch_size = 2048, num_heads_q = 2, num_heads_k = 2, seq_len_q = 64
seq_len_k = 32, attention_dim = 256, linear_dim = 256, dtype_str = 'fp8'

    @pytest.mark.parametrize(
        "batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim",
        [
            (2, 8, 4, 256, 64, 64, 64), # GQA
            (2, 8, 4, 64, 64, 64, 64),
            (16, 4, 4, 32, 499, 64, 32), # (v_d != qk_d)
            (5, 8, 4, 128, 256, 96, 128), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 64), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 80), # (v_d != qk_d)
            (2, 4, 1, 1001, 901, 128, 48), # (v_d != qk_d)
            (16, 4, 4, 501, 1000, 64, 64), # delta_q
            (8, 8, 8, 8000, 8000, 128, 128),
            (8, 8, 8, 8000, 8000, 256, 256),
            (2048, 4, 4, 52, 1000, 64, 64), # delta_q
            (2048, 2, 2, 32, 32, 256, 256),
            (2048, 2, 2, 64, 32, 256, 256),
            (96, 2, 2, 512, 3072, 256, 256), # delta_q
            (96, 2, 2, 512, 3072, 512, 512), # delta_q
            (1, 2, 2, 32, 32, 256, 256),
            (2, 8, 4, 32, 32, 32, 32),
        ],
    )
    @pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp8"])
    def test_triton_matches_golden(
        batch_size: int,
        num_heads_q: int,
        num_heads_k: int,
        seq_len_q: int,
        seq_len_k: int,
        attention_dim: int,
        linear_dim: int,
        dtype_str: str,
    ):
        # ===========prepare input===========
        set_seed(42)
        errors = []
        device = torch.device("npu")
        alpha = 1.0 / (attention_dim ** 0.5)
        type_mapper = {"fp16":torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
        dtype = type_mapper.get(dtype_str)
        q, k, v, seq_offset_q, seq_offset_k = get_or_generate_data(
            batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k,
            attention_dim, linear_dim, device, dtype
        )
    
    
        golden_output = golden_op_exec_high(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
    
        sim_output = golden_op_exec_low(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
        triton_output = triton_hstu_attention_fwd(
            q=q,
            k=k,
            v=v,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
            seq_offsets_q=seq_offset_q,
            seq_offsets_k=seq_offset_k,
            num_context=None,
            num_target=None,
            alpha=alpha,
            silu_scale=1.0 / seq_len_q,
        ).view([-1, num_heads_q, linear_dim]).to(device)
    
        if golden_output is None:
            return
    
>       assert compare_cv(golden_output.npu(), sim_output.npu(), triton_output.npu())
E       AssertionError: assert tensor(False, device='npu:0')
E        +  where tensor(False, device='npu:0') = compare_cv(tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0'), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16))
E        +    where tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0') = npu()
E        +      where npu = tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0').npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu

hstu/test/test_hstu_fwd_jagged.py:326: AssertionError
----------------------------- Captured stdout call -----------------------------

[INFO] Checking data cache: ./data_cache/bs2048_hq2_hk2_sq64_sk32_ad256_ld256_dttorch.float8_e4m3fn.pt
[INFO] Cache hit. Loading data...
[INFO] Data loaded successfully.
[INFO] Input shapes - Q: torch.Size([131072, 2, 256]), K: torch.Size([65536, 2, 256]), V: torch.Size([65536, 2, 256])
[INFO] Device: npu, Dtype: torch.float8_e4m3fn
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
BLOCK_M: 32  BLOCK_N: 384
err_threshold:0.00048828125
mare_npu:0.9999998807907104 mare_gpu:0.018160074949264526
mere_npu:0.00018856067617889494 mere_gpu:0.00030364940175786614
rmse_npu:0.0036798131186515093 rmse_gpu:0.0007414165884256363
MARE:55.06584548950195 MERE:0.38617226481437683 RMSE:4.963219165802002
new golden cv result:False
________ test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-80-100] _________

batch_size = 2048, num_heads_q = 2, num_heads_k = 2, seq_len_q = 64
seq_len_k = 32, attention_dim = 256, linear_dim = 256, dtype_str = 'fp8'

    @pytest.mark.parametrize(
        "batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim",
        [
            (2, 8, 4, 256, 64, 64, 64), # GQA
            (2, 8, 4, 64, 64, 64, 64),
            (16, 4, 4, 32, 499, 64, 32), # (v_d != qk_d)
            (5, 8, 4, 128, 256, 96, 128), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 64), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 80), # (v_d != qk_d)
            (2, 4, 1, 1001, 901, 128, 48), # (v_d != qk_d)
            (16, 4, 4, 501, 1000, 64, 64), # delta_q
            (8, 8, 8, 8000, 8000, 128, 128),
            (8, 8, 8, 8000, 8000, 256, 256),
            (2048, 4, 4, 52, 1000, 64, 64), # delta_q
            (2048, 2, 2, 32, 32, 256, 256),
            (2048, 2, 2, 64, 32, 256, 256),
            (96, 2, 2, 512, 3072, 256, 256), # delta_q
            (96, 2, 2, 512, 3072, 512, 512), # delta_q
            (1, 2, 2, 32, 32, 256, 256),
            (2, 8, 4, 32, 32, 32, 32),
        ],
    )
    @pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp8"])
    def test_triton_matches_golden(
        batch_size: int,
        num_heads_q: int,
        num_heads_k: int,
        seq_len_q: int,
        seq_len_k: int,
        attention_dim: int,
        linear_dim: int,
        dtype_str: str,
    ):
        # ===========prepare input===========
        set_seed(42)
        errors = []
        device = torch.device("npu")
        alpha = 1.0 / (attention_dim ** 0.5)
        type_mapper = {"fp16":torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
        dtype = type_mapper.get(dtype_str)
        q, k, v, seq_offset_q, seq_offset_k = get_or_generate_data(
            batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k,
            attention_dim, linear_dim, device, dtype
        )
    
    
        golden_output = golden_op_exec_high(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
    
        sim_output = golden_op_exec_low(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
        triton_output = triton_hstu_attention_fwd(
            q=q,
            k=k,
            v=v,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
            seq_offsets_q=seq_offset_q,
            seq_offsets_k=seq_offset_k,
            num_context=None,
            num_target=None,
            alpha=alpha,
            silu_scale=1.0 / seq_len_q,
        ).view([-1, num_heads_q, linear_dim]).to(device)
    
        if golden_output is None:
            return
    
>       assert compare_cv(golden_output.npu(), sim_output.npu(), triton_output.npu())
E       AssertionError: assert tensor(False, device='npu:0')
E        +  where tensor(False, device='npu:0') = compare_cv(tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0'), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16))
E        +    where tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0') = npu()
E        +      where npu = tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0').npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu

hstu/test/test_hstu_fwd_jagged.py:326: AssertionError
----------------------------- Captured stdout call -----------------------------

[INFO] Checking data cache: ./data_cache/bs2048_hq2_hk2_sq64_sk32_ad256_ld256_dttorch.float8_e4m3fn.pt
[INFO] Cache hit. Loading data...
[INFO] Data loaded successfully.
[INFO] Input shapes - Q: torch.Size([131072, 2, 256]), K: torch.Size([65536, 2, 256]), V: torch.Size([65536, 2, 256])
[INFO] Device: npu, Dtype: torch.float8_e4m3fn
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
BLOCK_M: 32  BLOCK_N: 384
err_threshold:0.00048828125
mare_npu:0.9999998807907104 mare_gpu:0.018160074949264526
mere_npu:0.00018665350216906518 mere_gpu:0.00030364940175786614
rmse_npu:0.003592065069824457 rmse_gpu:0.0007414165884256363
MARE:55.06584548950195 MERE:0.3822663724422455 RMSE:4.84486722946167
new golden cv result:False
________ test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-84-100] _________

batch_size = 2048, num_heads_q = 2, num_heads_k = 2, seq_len_q = 64
seq_len_k = 32, attention_dim = 256, linear_dim = 256, dtype_str = 'fp8'

    @pytest.mark.parametrize(
        "batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim",
        [
            (2, 8, 4, 256, 64, 64, 64), # GQA
            (2, 8, 4, 64, 64, 64, 64),
            (16, 4, 4, 32, 499, 64, 32), # (v_d != qk_d)
            (5, 8, 4, 128, 256, 96, 128), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 64), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 80), # (v_d != qk_d)
            (2, 4, 1, 1001, 901, 128, 48), # (v_d != qk_d)
            (16, 4, 4, 501, 1000, 64, 64), # delta_q
            (8, 8, 8, 8000, 8000, 128, 128),
            (8, 8, 8, 8000, 8000, 256, 256),
            (2048, 4, 4, 52, 1000, 64, 64), # delta_q
            (2048, 2, 2, 32, 32, 256, 256),
            (2048, 2, 2, 64, 32, 256, 256),
            (96, 2, 2, 512, 3072, 256, 256), # delta_q
            (96, 2, 2, 512, 3072, 512, 512), # delta_q
            (1, 2, 2, 32, 32, 256, 256),
            (2, 8, 4, 32, 32, 32, 32),
        ],
    )
    @pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp8"])
    def test_triton_matches_golden(
        batch_size: int,
        num_heads_q: int,
        num_heads_k: int,
        seq_len_q: int,
        seq_len_k: int,
        attention_dim: int,
        linear_dim: int,
        dtype_str: str,
    ):
        # ===========prepare input===========
        set_seed(42)
        errors = []
        device = torch.device("npu")
        alpha = 1.0 / (attention_dim ** 0.5)
        type_mapper = {"fp16":torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
        dtype = type_mapper.get(dtype_str)
        q, k, v, seq_offset_q, seq_offset_k = get_or_generate_data(
            batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k,
            attention_dim, linear_dim, device, dtype
        )
    
    
        golden_output = golden_op_exec_high(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
    
        sim_output = golden_op_exec_low(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
        triton_output = triton_hstu_attention_fwd(
            q=q,
            k=k,
            v=v,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
            seq_offsets_q=seq_offset_q,
            seq_offsets_k=seq_offset_k,
            num_context=None,
            num_target=None,
            alpha=alpha,
            silu_scale=1.0 / seq_len_q,
        ).view([-1, num_heads_q, linear_dim]).to(device)
    
        if golden_output is None:
            return
    
>       assert compare_cv(golden_output.npu(), sim_output.npu(), triton_output.npu())
E       AssertionError: assert tensor(False, device='npu:0')
E        +  where tensor(False, device='npu:0') = compare_cv(tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0'), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16))
E        +    where tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0') = npu()
E        +      where npu = tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0').npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu

hstu/test/test_hstu_fwd_jagged.py:326: AssertionError
----------------------------- Captured stdout call -----------------------------

[INFO] Checking data cache: ./data_cache/bs2048_hq2_hk2_sq64_sk32_ad256_ld256_dttorch.float8_e4m3fn.pt
[INFO] Cache hit. Loading data...
[INFO] Data loaded successfully.
[INFO] Input shapes - Q: torch.Size([131072, 2, 256]), K: torch.Size([65536, 2, 256]), V: torch.Size([65536, 2, 256])
[INFO] Device: npu, Dtype: torch.float8_e4m3fn
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
BLOCK_M: 32  BLOCK_N: 384
err_threshold:0.00048828125
mare_npu:0.9999998807907104 mare_gpu:0.018160074949264526
mere_npu:0.00020381668582558632 mere_gpu:0.00030364940175786614
rmse_npu:0.005453512072563171 rmse_gpu:0.0007414165884256363
MARE:55.06584548950195 MERE:0.4174165725708008 RMSE:7.355530261993408
new golden cv result:False
________ test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-85-100] _________

batch_size = 2048, num_heads_q = 2, num_heads_k = 2, seq_len_q = 64
seq_len_k = 32, attention_dim = 256, linear_dim = 256, dtype_str = 'fp8'

    @pytest.mark.parametrize(
        "batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim",
        [
            (2, 8, 4, 256, 64, 64, 64), # GQA
            (2, 8, 4, 64, 64, 64, 64),
            (16, 4, 4, 32, 499, 64, 32), # (v_d != qk_d)
            (5, 8, 4, 128, 256, 96, 128), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 64), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 80), # (v_d != qk_d)
            (2, 4, 1, 1001, 901, 128, 48), # (v_d != qk_d)
            (16, 4, 4, 501, 1000, 64, 64), # delta_q
            (8, 8, 8, 8000, 8000, 128, 128),
            (8, 8, 8, 8000, 8000, 256, 256),
            (2048, 4, 4, 52, 1000, 64, 64), # delta_q
            (2048, 2, 2, 32, 32, 256, 256),
            (2048, 2, 2, 64, 32, 256, 256),
            (96, 2, 2, 512, 3072, 256, 256), # delta_q
            (96, 2, 2, 512, 3072, 512, 512), # delta_q
            (1, 2, 2, 32, 32, 256, 256),
            (2, 8, 4, 32, 32, 32, 32),
        ],
    )
    @pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp8"])
    def test_triton_matches_golden(
        batch_size: int,
        num_heads_q: int,
        num_heads_k: int,
        seq_len_q: int,
        seq_len_k: int,
        attention_dim: int,
        linear_dim: int,
        dtype_str: str,
    ):
        # ===========prepare input===========
        set_seed(42)
        errors = []
        device = torch.device("npu")
        alpha = 1.0 / (attention_dim ** 0.5)
        type_mapper = {"fp16":torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
        dtype = type_mapper.get(dtype_str)
        q, k, v, seq_offset_q, seq_offset_k = get_or_generate_data(
            batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k,
            attention_dim, linear_dim, device, dtype
        )
    
    
        golden_output = golden_op_exec_high(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
    
        sim_output = golden_op_exec_low(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
        triton_output = triton_hstu_attention_fwd(
            q=q,
            k=k,
            v=v,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
            seq_offsets_q=seq_offset_q,
            seq_offsets_k=seq_offset_k,
            num_context=None,
            num_target=None,
            alpha=alpha,
            silu_scale=1.0 / seq_len_q,
        ).view([-1, num_heads_q, linear_dim]).to(device)
    
        if golden_output is None:
            return
    
>       assert compare_cv(golden_output.npu(), sim_output.npu(), triton_output.npu())
E       AssertionError: assert tensor(False, device='npu:0')
E        +  where tensor(False, device='npu:0') = compare_cv(tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0'), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16))
E        +    where tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0') = npu()
E        +      where npu = tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0').npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu

hstu/test/test_hstu_fwd_jagged.py:326: AssertionError
----------------------------- Captured stdout call -----------------------------

[INFO] Checking data cache: ./data_cache/bs2048_hq2_hk2_sq64_sk32_ad256_ld256_dttorch.float8_e4m3fn.pt
[INFO] Cache hit. Loading data...
[INFO] Data loaded successfully.
[INFO] Input shapes - Q: torch.Size([131072, 2, 256]), K: torch.Size([65536, 2, 256]), V: torch.Size([65536, 2, 256])
[INFO] Device: npu, Dtype: torch.float8_e4m3fn
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
BLOCK_M: 32  BLOCK_N: 384
err_threshold:0.00048828125
mare_npu:0.9999998807907104 mare_gpu:0.018160074949264526
mere_npu:0.000209537596674636 mere_gpu:0.00030364940175786614
rmse_npu:0.00597195653244853 rmse_gpu:0.0007414165884256363
MARE:55.06584548950195 MERE:0.42913299798965454 RMSE:8.054792404174805
new golden cv result:False
________ test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-94-100] _________

batch_size = 2048, num_heads_q = 2, num_heads_k = 2, seq_len_q = 64
seq_len_k = 32, attention_dim = 256, linear_dim = 256, dtype_str = 'fp8'

    @pytest.mark.parametrize(
        "batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim",
        [
            (2, 8, 4, 256, 64, 64, 64), # GQA
            (2, 8, 4, 64, 64, 64, 64),
            (16, 4, 4, 32, 499, 64, 32), # (v_d != qk_d)
            (5, 8, 4, 128, 256, 96, 128), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 64), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 80), # (v_d != qk_d)
            (2, 4, 1, 1001, 901, 128, 48), # (v_d != qk_d)
            (16, 4, 4, 501, 1000, 64, 64), # delta_q
            (8, 8, 8, 8000, 8000, 128, 128),
            (8, 8, 8, 8000, 8000, 256, 256),
            (2048, 4, 4, 52, 1000, 64, 64), # delta_q
            (2048, 2, 2, 32, 32, 256, 256),
            (2048, 2, 2, 64, 32, 256, 256),
            (96, 2, 2, 512, 3072, 256, 256), # delta_q
            (96, 2, 2, 512, 3072, 512, 512), # delta_q
            (1, 2, 2, 32, 32, 256, 256),
            (2, 8, 4, 32, 32, 32, 32),
        ],
    )
    @pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp8"])
    def test_triton_matches_golden(
        batch_size: int,
        num_heads_q: int,
        num_heads_k: int,
        seq_len_q: int,
        seq_len_k: int,
        attention_dim: int,
        linear_dim: int,
        dtype_str: str,
    ):
        # ===========prepare input===========
        set_seed(42)
        errors = []
        device = torch.device("npu")
        alpha = 1.0 / (attention_dim ** 0.5)
        type_mapper = {"fp16":torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
        dtype = type_mapper.get(dtype_str)
        q, k, v, seq_offset_q, seq_offset_k = get_or_generate_data(
            batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k,
            attention_dim, linear_dim, device, dtype
        )
    
    
        golden_output = golden_op_exec_high(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
    
        sim_output = golden_op_exec_low(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
        triton_output = triton_hstu_attention_fwd(
            q=q,
            k=k,
            v=v,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
            seq_offsets_q=seq_offset_q,
            seq_offsets_k=seq_offset_k,
            num_context=None,
            num_target=None,
            alpha=alpha,
            silu_scale=1.0 / seq_len_q,
        ).view([-1, num_heads_q, linear_dim]).to(device)
    
        if golden_output is None:
            return
    
>       assert compare_cv(golden_output.npu(), sim_output.npu(), triton_output.npu())
E       AssertionError: assert tensor(False, device='npu:0')
E        +  where tensor(False, device='npu:0') = compare_cv(tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0'), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16))
E        +    where tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0') = npu()
E        +      where npu = tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0').npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu

hstu/test/test_hstu_fwd_jagged.py:326: AssertionError
----------------------------- Captured stdout call -----------------------------

[INFO] Checking data cache: ./data_cache/bs2048_hq2_hk2_sq64_sk32_ad256_ld256_dttorch.float8_e4m3fn.pt
[INFO] Cache hit. Loading data...
[INFO] Data loaded successfully.
[INFO] Input shapes - Q: torch.Size([131072, 2, 256]), K: torch.Size([65536, 2, 256]), V: torch.Size([65536, 2, 256])
[INFO] Device: npu, Dtype: torch.float8_e4m3fn
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
BLOCK_M: 32  BLOCK_N: 384
err_threshold:0.00048828125
mare_npu:0.9999998807907104 mare_gpu:0.018160074949264526
mere_npu:0.0002038166276179254 mere_gpu:0.00030364940175786614
rmse_npu:0.005480462685227394 rmse_gpu:0.0007414165884256363
MARE:55.06584548950195 MERE:0.41741645336151123 RMSE:7.391880512237549
new golden cv result:False
________ test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-95-100] _________

batch_size = 2048, num_heads_q = 2, num_heads_k = 2, seq_len_q = 64
seq_len_k = 32, attention_dim = 256, linear_dim = 256, dtype_str = 'fp8'

    @pytest.mark.parametrize(
        "batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim",
        [
            (2, 8, 4, 256, 64, 64, 64), # GQA
            (2, 8, 4, 64, 64, 64, 64),
            (16, 4, 4, 32, 499, 64, 32), # (v_d != qk_d)
            (5, 8, 4, 128, 256, 96, 128), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 64), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 80), # (v_d != qk_d)
            (2, 4, 1, 1001, 901, 128, 48), # (v_d != qk_d)
            (16, 4, 4, 501, 1000, 64, 64), # delta_q
            (8, 8, 8, 8000, 8000, 128, 128),
            (8, 8, 8, 8000, 8000, 256, 256),
            (2048, 4, 4, 52, 1000, 64, 64), # delta_q
            (2048, 2, 2, 32, 32, 256, 256),
            (2048, 2, 2, 64, 32, 256, 256),
            (96, 2, 2, 512, 3072, 256, 256), # delta_q
            (96, 2, 2, 512, 3072, 512, 512), # delta_q
            (1, 2, 2, 32, 32, 256, 256),
            (2, 8, 4, 32, 32, 32, 32),
        ],
    )
    @pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp8"])
    def test_triton_matches_golden(
        batch_size: int,
        num_heads_q: int,
        num_heads_k: int,
        seq_len_q: int,
        seq_len_k: int,
        attention_dim: int,
        linear_dim: int,
        dtype_str: str,
    ):
        # ===========prepare input===========
        set_seed(42)
        errors = []
        device = torch.device("npu")
        alpha = 1.0 / (attention_dim ** 0.5)
        type_mapper = {"fp16":torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
        dtype = type_mapper.get(dtype_str)
        q, k, v, seq_offset_q, seq_offset_k = get_or_generate_data(
            batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k,
            attention_dim, linear_dim, device, dtype
        )
    
    
        golden_output = golden_op_exec_high(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
    
        sim_output = golden_op_exec_low(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
        triton_output = triton_hstu_attention_fwd(
            q=q,
            k=k,
            v=v,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
            seq_offsets_q=seq_offset_q,
            seq_offsets_k=seq_offset_k,
            num_context=None,
            num_target=None,
            alpha=alpha,
            silu_scale=1.0 / seq_len_q,
        ).view([-1, num_heads_q, linear_dim]).to(device)
    
        if golden_output is None:
            return
    
>       assert compare_cv(golden_output.npu(), sim_output.npu(), triton_output.npu())
E       AssertionError: assert tensor(False, device='npu:0')
E        +  where tensor(False, device='npu:0') = compare_cv(tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0'), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16))
E        +    where tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0') = npu()
E        +      where npu = tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0').npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu

hstu/test/test_hstu_fwd_jagged.py:326: AssertionError
----------------------------- Captured stdout call -----------------------------

[INFO] Checking data cache: ./data_cache/bs2048_hq2_hk2_sq64_sk32_ad256_ld256_dttorch.float8_e4m3fn.pt
[INFO] Cache hit. Loading data...
[INFO] Data loaded successfully.
[INFO] Input shapes - Q: torch.Size([131072, 2, 256]), K: torch.Size([65536, 2, 256]), V: torch.Size([65536, 2, 256])
[INFO] Device: npu, Dtype: torch.float8_e4m3fn
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
BLOCK_M: 32  BLOCK_N: 384
err_threshold:0.00048828125
mare_npu:0.9999998807907104 mare_gpu:0.018160074949264526
mere_npu:0.00018856054521165788 mere_gpu:0.00030364940175786614
rmse_npu:0.003760232124477625 rmse_gpu:0.0007414165884256363
MARE:55.06584548950195 MERE:0.38617199659347534 RMSE:5.071685791015625
new golden cv result:False
________ test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-97-100] _________

batch_size = 2048, num_heads_q = 2, num_heads_k = 2, seq_len_q = 64
seq_len_k = 32, attention_dim = 256, linear_dim = 256, dtype_str = 'fp8'

    @pytest.mark.parametrize(
        "batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim",
        [
            (2, 8, 4, 256, 64, 64, 64), # GQA
            (2, 8, 4, 64, 64, 64, 64),
            (16, 4, 4, 32, 499, 64, 32), # (v_d != qk_d)
            (5, 8, 4, 128, 256, 96, 128), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 64), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 80), # (v_d != qk_d)
            (2, 4, 1, 1001, 901, 128, 48), # (v_d != qk_d)
            (16, 4, 4, 501, 1000, 64, 64), # delta_q
            (8, 8, 8, 8000, 8000, 128, 128),
            (8, 8, 8, 8000, 8000, 256, 256),
            (2048, 4, 4, 52, 1000, 64, 64), # delta_q
            (2048, 2, 2, 32, 32, 256, 256),
            (2048, 2, 2, 64, 32, 256, 256),
            (96, 2, 2, 512, 3072, 256, 256), # delta_q
            (96, 2, 2, 512, 3072, 512, 512), # delta_q
            (1, 2, 2, 32, 32, 256, 256),
            (2, 8, 4, 32, 32, 32, 32),
        ],
    )
    @pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp8"])
    def test_triton_matches_golden(
        batch_size: int,
        num_heads_q: int,
        num_heads_k: int,
        seq_len_q: int,
        seq_len_k: int,
        attention_dim: int,
        linear_dim: int,
        dtype_str: str,
    ):
        # ===========prepare input===========
        set_seed(42)
        errors = []
        device = torch.device("npu")
        alpha = 1.0 / (attention_dim ** 0.5)
        type_mapper = {"fp16":torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
        dtype = type_mapper.get(dtype_str)
        q, k, v, seq_offset_q, seq_offset_k = get_or_generate_data(
            batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k,
            attention_dim, linear_dim, device, dtype
        )
    
    
        golden_output = golden_op_exec_high(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
    
        sim_output = golden_op_exec_low(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
        triton_output = triton_hstu_attention_fwd(
            q=q,
            k=k,
            v=v,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
            seq_offsets_q=seq_offset_q,
            seq_offsets_k=seq_offset_k,
            num_context=None,
            num_target=None,
            alpha=alpha,
            silu_scale=1.0 / seq_len_q,
        ).view([-1, num_heads_q, linear_dim]).to(device)
    
        if golden_output is None:
            return
    
>       assert compare_cv(golden_output.npu(), sim_output.npu(), triton_output.npu())
E       AssertionError: assert tensor(False, device='npu:0')
E        +  where tensor(False, device='npu:0') = compare_cv(tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0'), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16))
E        +    where tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0') = npu()
E        +      where npu = tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0').npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu

hstu/test/test_hstu_fwd_jagged.py:326: AssertionError
----------------------------- Captured stdout call -----------------------------

[INFO] Checking data cache: ./data_cache/bs2048_hq2_hk2_sq64_sk32_ad256_ld256_dttorch.float8_e4m3fn.pt
[INFO] Cache hit. Loading data...
[INFO] Data loaded successfully.
[INFO] Input shapes - Q: torch.Size([131072, 2, 256]), K: torch.Size([65536, 2, 256]), V: torch.Size([65536, 2, 256])
[INFO] Device: npu, Dtype: torch.float8_e4m3fn
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
BLOCK_M: 32  BLOCK_N: 384
err_threshold:0.00048828125
mare_npu:0.9999999403953552 mare_gpu:0.018160074949264526
mere_npu:0.0005165674374438822 mere_gpu:0.00030364940175786614
rmse_npu:0.018335571512579918 rmse_gpu:0.0007414165884256363
MARE:55.06584930419922 MERE:1.0579301118850708 RMSE:24.730457305908203
new golden cv result:False
________ test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-100-100] ________

batch_size = 2048, num_heads_q = 2, num_heads_k = 2, seq_len_q = 64
seq_len_k = 32, attention_dim = 256, linear_dim = 256, dtype_str = 'fp8'

    @pytest.mark.parametrize(
        "batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim",
        [
            (2, 8, 4, 256, 64, 64, 64), # GQA
            (2, 8, 4, 64, 64, 64, 64),
            (16, 4, 4, 32, 499, 64, 32), # (v_d != qk_d)
            (5, 8, 4, 128, 256, 96, 128), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 64), # (v_d != qk_d)
            (4, 4, 4, 512, 1024, 72, 80), # (v_d != qk_d)
            (2, 4, 1, 1001, 901, 128, 48), # (v_d != qk_d)
            (16, 4, 4, 501, 1000, 64, 64), # delta_q
            (8, 8, 8, 8000, 8000, 128, 128),
            (8, 8, 8, 8000, 8000, 256, 256),
            (2048, 4, 4, 52, 1000, 64, 64), # delta_q
            (2048, 2, 2, 32, 32, 256, 256),
            (2048, 2, 2, 64, 32, 256, 256),
            (96, 2, 2, 512, 3072, 256, 256), # delta_q
            (96, 2, 2, 512, 3072, 512, 512), # delta_q
            (1, 2, 2, 32, 32, 256, 256),
            (2, 8, 4, 32, 32, 32, 32),
        ],
    )
    @pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp8"])
    def test_triton_matches_golden(
        batch_size: int,
        num_heads_q: int,
        num_heads_k: int,
        seq_len_q: int,
        seq_len_k: int,
        attention_dim: int,
        linear_dim: int,
        dtype_str: str,
    ):
        # ===========prepare input===========
        set_seed(42)
        errors = []
        device = torch.device("npu")
        alpha = 1.0 / (attention_dim ** 0.5)
        type_mapper = {"fp16":torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
        dtype = type_mapper.get(dtype_str)
        q, k, v, seq_offset_q, seq_offset_k = get_or_generate_data(
            batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k,
            attention_dim, linear_dim, device, dtype
        )
    
    
        golden_output = golden_op_exec_high(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
    
        sim_output = golden_op_exec_low(
            q=q,
            k=k,
            v=v,
            silu_scale=1.0 / seq_len_q,
            alpha=alpha,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_k,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
        ).view([-1, num_heads_q, linear_dim]).to("npu")
    
        triton_output = triton_hstu_attention_fwd(
            q=q,
            k=k,
            v=v,
            max_seq_len_q=seq_len_q,
            max_seq_len_k=seq_len_k,
            seq_offsets_q=seq_offset_q,
            seq_offsets_k=seq_offset_k,
            num_context=None,
            num_target=None,
            alpha=alpha,
            silu_scale=1.0 / seq_len_q,
        ).view([-1, num_heads_q, linear_dim]).to(device)
    
        if golden_output is None:
            return
    
>       assert compare_cv(golden_output.npu(), sim_output.npu(), triton_output.npu())
E       AssertionError: assert tensor(False, device='npu:0')
E        +  where tensor(False, device='npu:0') = compare_cv(tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0'), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16), tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16))
E        +    where tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0') = npu()
E        +      where npu = tensor([[[0.8102, 1.0370, 0.8618,  ..., 0.8344, 0.8453, 0.9014],\n         [0.9800, 0.9654, 0.8799,  ..., 0.8610, 0.882......, 0.9650, 0.9000, 0.9486],\n         [1.0049, 1.0256, 0.9102,  ..., 1.0818, 0.9655, 1.0209]]],\n       device='npu:0').npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu
E        +    and   tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16) = npu()
E        +      where npu = tensor([[[0.8101, 1.0371, 0.8618,  ..., 0.8345, 0.8452, 0.9014],\n         [0.9800, 0.9653, 0.8799,  ..., 0.8608, 0.882...0.9487],\n         [1.0049, 1.0254, 0.9102,  ..., 1.0820, 0.9653, 1.0205]]],\n       device='npu:0', dtype=torch.float16).npu

hstu/test/test_hstu_fwd_jagged.py:326: AssertionError
----------------------------- Captured stdout call -----------------------------

[INFO] Checking data cache: ./data_cache/bs2048_hq2_hk2_sq64_sk32_ad256_ld256_dttorch.float8_e4m3fn.pt
[INFO] Cache hit. Loading data...
[INFO] Data loaded successfully.
[INFO] Input shapes - Q: torch.Size([131072, 2, 256]), K: torch.Size([65536, 2, 256]), V: torch.Size([65536, 2, 256])
[INFO] Device: npu, Dtype: torch.float8_e4m3fn
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
zeros done
seq_lens_k done
q_dens jagged_to_dense done
k_dens jagged_to_dense done
v_dens jagged_to_dense done
permute done
matmul done
silu done
v_dens done
BLOCK_M: 32  BLOCK_N: 384
err_threshold:0.00048828125
mare_npu:0.9999998807907104 mare_gpu:0.018160074949264526
mere_npu:0.00018856048700399697 mere_gpu:0.00030364940175786614
rmse_npu:0.003907563630491495 rmse_gpu:0.0007414165884256363
MARE:55.06584548950195 MERE:0.3861718773841858 RMSE:5.270402431488037
new golden cv result:False
=============================== warnings summary ===============================
hstu/test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-1-100]
  /usr/local/python3.11.0/lib/python3.11/site-packages/torch/utils/backend_registration.py:148: UserWarning: Cannot create tensor with interal format while allow_internel_format=False, tensor will be created with base format. (Triggered internally at torch_npu/csrc/aten/common/TensorFactories.cpp:339.)
    return self.to(device=torch.device(f'{custom_backend_name}:{device_idx}'), non_blocking=non_blocking, **kwargs)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED hstu/test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-2-100]
FAILED hstu/test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-10-100]
FAILED hstu/test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-11-100]
FAILED hstu/test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-12-100]
FAILED hstu/test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-18-100]
FAILED hstu/test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-20-100]
FAILED hstu/test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-25-100]
FAILED hstu/test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-28-100]
FAILED hstu/test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-45-100]
FAILED hstu/test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-80-100]
FAILED hstu/test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-84-100]
FAILED hstu/test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-85-100]
FAILED hstu/test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-94-100]
FAILED hstu/test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-95-100]
FAILED hstu/test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-97-100]
FAILED hstu/test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp8-2048-2-2-64-32-256-256-100-100]
==== 16 failed, 84 passed, 5000 deselected, 1 warning in 179.88s (0:02:59) =====
