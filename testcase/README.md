# HighPriority50Operators testcase inventory

本目录按 `/Users/lingli/Downloads/record.md` 的顺序整理 50 个逻辑用例。

- 编号目录名以 `record.md` 中的测试入口为准；只把空格等不适合目录名的字符规范为下划线。
- 每个编号目录下直接并列放原始测试入口文件和对应 kernel 实现文件，没有额外的 `source/` 层级，也没有改写 kernel、shape、dtype、golden 或测试逻辑。
- 测试入口和 kernel 本来就在同一个文件时只保留一份；确实需要多个本地文件时（例如 FP8 FA、softcap 或 diffusion attention），这些文件直接并列放在该编号目录下。
- 多个用例共享的完整工程依赖放在 `common/`，避免在每个目录重复复制。
- `MANIFEST.tsv` 是 50 项的入口、kernel、来源和状态总表。
- `CASE_FILES.tsv` 明确列出每项同目录保存的测试文件、kernel 文件及覆盖关系。
- `COVERAGE_GAPS.md` 记录不能被原始测试严格覆盖的项目，禁止用无关测试冒充。
- `NEEDS_CONFIRMATION.md` 记录原来源冲突及用户最终确认的选择。
- `SHA256SUMS` 锁定所有编号目录和 `common/` 内的原始文件副本。

## 公共依赖

- `common/mojo_opset_project`：本地 Mojo OpSet Python 包及测试公共文件。
- `common/flash_linear_attention_project`：完整 `fla/` Python 包和工程元数据。
- `common/megablocks_project`：Megablocks Python 包、源码和构建元数据。
- `common/vllm_ascend_fla`：vLLM Ascend FLA 相关包和直接依赖。
- `common/customer_kernels_project`：Customer_Kernels 固定提交的完整工作树（不含 `.git`）。

## 特别说明

`41_chunk_scaled_dot_kkt_fwd_no_gate` 与
`42_chunk_scaled_dot_kkt_fwd_gated` 是两个独立回归场景，但使用完全相同的
`fla/ops/common/chunk_scaled_dot_kkt.py`：

- no_gate：调用 `chunk_scaled_dot_kkt_fwd` 时 `g=None`；
- gated：调用同一入口时传入 `g`。

没有为二者创建或修改任何 Triton kernel。

本地 `mojo_opset-master` 的
`mojo_opset/backends/ttx/kernels/npu/flash_attention.py` 曾在第 321 行混入一段会破坏
Python 语法的会话 ID。`common/mojo_opset_project` 中只对这个公共依赖采用了下载目录中
同一 Mojo 快照的干净原件，SHA256 为
`2e643be5b9d08664c8bf370b663f86c45a78caca0b6c4ffc304fdf10758291f4`；
编号目录中的 `test_attention.py` 本身仍是本地测试文件的逐字节副本。
