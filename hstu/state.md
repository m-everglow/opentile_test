# Forward:
## hstu_bwd_xkp分支
【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp16-2048-2-2-64-32-256-256]
【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp16-2-8-4-256-64-64-64]
【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp16-2-8-4-64-64-64-64]
【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_fwd_jagged.py::test_triton_matches_golden[fp16-2-8-4-32-32-32-32]
# Backward:
## master分支：
【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-2048-2-2-32-32-256-256-False]
## hstu_bwd_xkp分支
"batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim"
【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-2048-2-2-32-32-256-256]
【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-2048-2-2-64-32-256-256]
【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-2-8-8-32-32-32-32]
【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-2-8-4-32-32-32-32]
【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-2-8-4-256-64-64-64]
【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-2-2-2-32-32-32-32]
【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-1-1-1-16-16-16-16]


【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-1-1-1-24-16-16-16]
【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-1-1-1-16-24-16-16]
【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-1-1-1-16-30-16-16]
【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-1-1-1-16-176-16-16]
【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-1-1-1-32-30-16-16]
【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-1-1-1-30-32-16-16]


【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-1-2-2-24-16-16-16]

【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-2-2-2-24-16-16-16]
【psss】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-16-4-4-24-16-16-16]
【psss】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-16-4-4-32-16-64-32]
【psss】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-16-4-4-32-31-64-32]
【psss】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-16-4-4-32-63-64-32]
【psss】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-16-4-4-32-127-64-32]
【psss】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-16-4-4-32-127-64-32]
【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-16-4-4-32-160-64-32]
【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-16-4-4-32-176-64-32]


【pass】TRITON_INTERPRET=1 pytest ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-16-4-4-32-512-64-32]

Progress:
0119: head_num_q != head_num_k case passed

HSTU后向NPU验证情况：
fp16精度PASS: 13/13
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-2-8-4-256-64-64-64-False] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-16-4-4-32-499-64-32-False] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-5-8-4-128-256-96-128-False] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-4-4-4-512-1024-72-64-False] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-4-4-4-512-1024-72-80-False] pass


pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-2-4-1-1001-901-128-48-False] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-16-4-4-501-1000-64-64-False] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-96-2-2-512-3072-256-256-False] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-96-2-2-512-3072-512-512-False] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-2048-4-4-52-1000-64-64-False] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-8-8-8-8000-8000-128-128-True] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-8-8-8-8000-8000-256-256-True] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[fp16-2048-2-2-32-32-256-256-True] pass

bf16精度PASS: 13/13
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[bf16-2-8-4-256-64-64-64-False] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[bf16-16-4-4-32-499-64-32-False] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[bf16-5-8-4-128-256-96-128-False] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[bf16-4-4-4-512-1024-72-64-False] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[bf16-4-4-4-512-1024-72-80-False] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[bf16-2-4-1-1001-901-128-48-False] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[bf16-16-4-4-501-1000-64-64-False] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[bf16-96-2-2-512-3072-256-256-False] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[bf16-96-2-2-512-3072-512-512-False] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[bf16-2048-4-4-52-1000-64-64-False] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[bf16-8-8-8-8000-8000-128-128-True] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[bf16-8-8-8-8000-8000-256-256-True] pass
pytest -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd[bf16-2048-2-2-32-32-256-256-True] pass

pytest -n8 -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd

# CI看护的PASS用例
cd triton-ops/native/hstu
用例1：（QK等长，快速验证）PASS 12
pytest -sv hstu_demo_fwd_bwd_ok_ci.py::test_op_ci
用例2：（前向验收shape）PASS 34
pytest -n8 -sv ./test/test_hstu_fwd_jagged.py::test_triton_matches_golden
用例3：（后向验收shape）PASS 26
pytest -n8 -sv ./test/test_hstu_bwd_jagged.py::test_triton_matches_asc_bwd