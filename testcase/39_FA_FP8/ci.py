"""A5 end-to-end gate for the production FP8 FlashAttention forward kernel.

The numerical contract is not reimplemented in this file.  It calls the
byte-identical upstream test function copied next to this file as
``fp8_fwd_contract.py``:

  Downloads/test_fa_fwd_fp.py::test_op
  SHA256 7682230827455db59f8a06d7a1e64ff58c3ef953d1e0e1760be15246927b4c2d

That function imports and launches a compatibility-patched copy of the
production kernel source:

  Downloads/fa_forward_fp8.py::attention -> ::_attn_fwd_fp8
  compatibility SHA256 bca89d0df1322c3af90262403d6bac03d86dc114bb05c0d15527f4e560cc9e9c
  upstream SHA256 cdcce1fc9a7bcb875afaed4973a3c00a8623dd3e876280354592ca6afb5d7c70

The compatibility copy temporarily omits ``propagate_nan=True`` from the two
``tl.max(qk, 1, ...)`` reductions because the deployed Converter does not
accept that keyword.  Each omission has a
``TODO(CONVERTER_PROPAGATE_NAN)`` marker so the original contract can be
restored after the Converter upgrade.  The outer ``tl.maximum`` NaN behavior
is unchanged.  Native MLIR diagnostics and bounded reproducer metadata are
included when compilation fails.

Acceptance specialization:
  Z=128, H=8, N_CTX=1024, HEAD_DIM=64, noncausal
  base Q/K/V=float32 normal(mean=0, std=0.5), seed=20, generated on NPU
  quantized Q/K/V=float8_e4m3fn; Q block=64, K/V block=128
  production=fa_forward_fp8.attention
  golden=test_fa_fwd_fp.tforward_npu
  full-output acceptance=max(abs(actual-golden)) < 0.1

CI-only difference from the original active pytest matrix:
  BM=64/BN=128 is a derived full-shape specialization already validated on A5.
  The original D64 noncausal active row uses BM=128/BN=256, which is outside
  the currently declared A5 acceptance boundary because of UB planning limits.

The environment/device setup and compact failure handling follow the known-good
Paged Decode E2E structure.  They do not change input generation, quantization,
kernel arguments, golden computation, or comparison.
"""

import faulthandler
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


# Select the in-tree OpenTile backend before importing Triton kernel sources.
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_BACKEND", "opentile")
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
os.environ.setdefault("TRITON_DEBUG", "1")
os.environ.setdefault("MLIR_ENABLE_DIAGNOSTICS", "operations,stacktraces")
_DEFAULT_REPRODUCER_PREFIX = f"/tmp/fp8_fa_repro_{os.getpid()}"
os.environ.setdefault("TRITON_REPRODUCER_PATH", _DEFAULT_REPRODUCER_PREFIX)

_EXPECTED_CONTRACT_SHA256 = (
    "7682230827455db59f8a06d7a1e64ff58c3ef953d1e0e1760be15246927b4c2d"
)
_EXPECTED_KERNEL_SHA256 = (
    "bca89d0df1322c3af90262403d6bac03d86dc114bb05c0d15527f4e560cc9e9c"
)
_UPSTREAM_KERNEL_SHA256 = (
    "cdcce1fc9a7bcb875afaed4973a3c00a8623dd3e876280354592ca6afb5d7c70"
)
_HERE = Path(__file__).resolve().parent
_CONTRACT_PATH = _HERE / "fp8_fwd_contract.py"
_KERNEL_PATH = _HERE / "fa_forward_fp8.py"
_REPORT_PATH = Path(
    os.environ.get(
        "FP8_FA_REPORT_PATH",
        str(_HERE / "fp8_fa_e2e_report.log"),
    )
)
_STAGE_START = time.monotonic()
_LAST_STAGE = "module_import"
_LAST_NATIVE_STDERR_PATH = None
_CONTRACT_MODULE = None
_PREFLIGHT_DONE = False
_TRITON_ASCEND_ONLY_LAUNCH_OPTIONS = (
    "multibuffer",
    "enable_auto_bind_sub_block",
    "sync_solver",
    "limit_auto_multi_buffer_of_local_buffer",
    "set_workspace_multibuffer",
)


def _emit(message):
    """Print and durably save one compact line even under pytest capture."""
    print(message, flush=True)
    try:
        with _REPORT_PATH.open("a", encoding="utf-8") as report_file:
            report_file.write(message + "\n")
            report_file.flush()
            os.fsync(report_file.fileno())
    except OSError as log_error:
        print(
            f"[FP8_FA_LOG_WARNING] path={_REPORT_PATH} "
            f"type={type(log_error).__name__} message={log_error}",
            flush=True,
        )


try:
    _REPORT_PATH.write_text("", encoding="utf-8")
except OSError:
    # Pytest will still receive stdout/stderr if the testcase directory is
    # unexpectedly read-only.
    pass


def _stage(message):
    global _LAST_STAGE
    _LAST_STAGE = message
    elapsed = time.monotonic() - _STAGE_START
    _emit(f"[FP8_FA_STAGE] elapsed={elapsed:.3f}s {message}")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact(value, limit=1200):
    return str(value).replace("\n", " ").replace("\r", " ")[:limit]


def _git_head(path):
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "NA"
    return result.stdout.strip() if result.returncode == 0 else "NA"


def _find_git_root(path):
    current = Path(path).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


class _NativeStderrCapture:
    """Capture C++/MLIR fd2 diagnostics while preserving bounded reporting."""

    def __init__(self, label):
        self.path = _HERE / f"fp8_fa_native_{label}.log"
        self._saved_fd = None

    def __enter__(self):
        global _LAST_NATIVE_STDERR_PATH
        sys.stderr.flush()
        self._saved_fd = os.dup(2)
        capture_fd = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o644,
        )
        os.dup2(capture_fd, 2)
        os.close(capture_fd)
        _LAST_NATIVE_STDERR_PATH = self.path
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.stderr.flush()
        finally:
            if self._saved_fd is not None:
                os.dup2(self._saved_fd, 2)
                os.close(self._saved_fd)
                self._saved_fd = None
        return False


def _emit_native_stderr_summary(expand_tail):
    path = _LAST_NATIVE_STDERR_PATH
    if path is None or not path.exists():
        _emit("[FP8_FA_NATIVE_STDERR] path=NA bytes=0 sha256=NA")
        return
    size = path.stat().st_size
    digest = _sha256(path)
    _emit(
        f"[FP8_FA_NATIVE_STDERR] path={path} bytes={size} sha256={digest}"
    )
    if not expand_tail or size == 0:
        return
    with path.open("rb") as source_file:
        if size > 12288:
            source_file.seek(size - 12288)
        tail = source_file.read().decode("utf-8", errors="replace")
    _emit("[FP8_FA_NATIVE_STDERR_TAIL_BEGIN]")
    for line in tail.splitlines()[-120:]:
        _emit(f"[NATIVE] {_compact(line, 1000)}")
    _emit("[FP8_FA_NATIVE_STDERR_TAIL_END]")


def _reproducer_files():
    prefix = Path(os.environ["TRITON_REPRODUCER_PATH"])
    return sorted(
        prefix.parent.glob(prefix.name + "*.repro.mlir"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )


def _emit_reproducer_summary():
    files = _reproducer_files()
    _emit(
        "[FP8_FA_REPRODUCERS] "
        f"prefix={os.environ['TRITON_REPRODUCER_PATH']} count={len(files)}"
    )
    for path in files[-8:]:
        _emit(
            "[FP8_FA_REPRODUCER] "
            f"name={path.name} bytes={path.stat().st_size} sha256={_sha256(path)}"
        )
    if not files:
        return
    relevant = []
    counts = {"ftof": 0, "fp8e4": 0, "fp8e5": 0}
    try:
        with files[-1].open("r", encoding="utf-8", errors="replace") as source_file:
            for line_number, line in enumerate(source_file, 1):
                lowered = line.lower()
                for key in counts:
                    if key in lowered:
                        counts[key] += 1
                if (
                    len(relevant) < 12
                    and ("ftof" in lowered or "f8e4m3" in lowered or "f8e5m2" in lowered)
                ):
                    relevant.append((line_number, _compact(line, 900)))
    except OSError as error:
        _emit(
            f"[FP8_FA_REPRODUCER_READ_ERROR] "
            f"type={type(error).__name__} message={_compact(error)}"
        )
        return
    _emit(
        "[FP8_FA_REPRODUCER_COUNTS] "
        + " ".join(f"{key}={value}" for key, value in counts.items())
    )
    for line_number, line in relevant:
        _emit(f"[FP8_FA_REPRODUCER_MATCH] line={line_number} text={line}")


def _parse_logical_device():
    raw = os.environ.get("OPENTILE_TEST_DEVICE", "0")
    return int(raw.rsplit(":", 1)[-1])


_stage("pytest_import_begin")
import pytest
_stage("pytest_import_done")

_stage("torch_import_begin")
torch = pytest.importorskip("torch")
_stage(f"torch_import_done version={torch.__version__}")
_stage("torch_npu_import_begin")
torch_npu = pytest.importorskip("torch_npu")
_stage(f"torch_npu_import_done version={getattr(torch_npu, '__version__', 'NA')}")

_DEVICE_ID = _parse_logical_device()
_stage(
    "device_config "
    f"logical_device={_DEVICE_ID} "
    f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', 'unset')}"
)
_stage("npu_is_available_begin")
_NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
_stage(f"npu_is_available_done value={int(_NPU_AVAILABLE)}")
if not _NPU_AVAILABLE:
    pytest.skip(
        "torch_npu is available, but no NPU device is visible",
        allow_module_level=True,
    )
_stage(f"set_device_begin logical_device={_DEVICE_ID}")
torch.npu.set_device(_DEVICE_ID)
_stage(f"set_device_done logical_device={_DEVICE_ID}")
_stage("module_import_done")


def _frontend_preflight():
    """Report the exact frontend/backend used by the formal E2E process."""
    global _PREFLIGHT_DONE
    if _PREFLIGHT_DONE:
        _stage("frontend_preflight_cached_pass")
        return
    _stage("frontend_preflight_import_begin")
    import triton
    import triton.language.standard as tl_standard
    from triton.backends.opentile.targets.ascend import compiler as ascend_compiler
    from triton.runtime import driver

    _stage("frontend_preflight_import_done")
    compiler_path = Path(ascend_compiler.__file__).resolve()
    standard_path = Path(tl_standard.__file__).resolve()
    triton_path = Path(triton.__file__).resolve()
    options = ascend_compiler.OpenTileNPUOptions()
    supported_fp8 = tuple(options.supported_fp8_dtypes)
    max_callable = getattr(tl_standard.max, "fn", tl_standard.max)
    try:
        max_signature = str(inspect.signature(max_callable))
    except (TypeError, ValueError):
        max_signature = "NA"
    converter_root = _find_git_root(compiler_path)
    opentile_root = compiler_path.parents[3]
    opentile_ir_root = opentile_root / "OpenTileIR"
    target = driver.active.get_current_target()
    opentileas_path = shutil.which("opentileas")
    opentileas_root = _find_git_root(opentileas_path) if opentileas_path else None
    opentileas_binary = Path(opentileas_path) if opentileas_path else None

    _emit(
        "[FP8_FA_PROVENANCE] "
        f"python={sys.executable} python_version={sys.version.split()[0]} "
        f"triton_version={getattr(triton, '__version__', 'NA')}"
    )
    _emit(
        "[FP8_FA_PROVENANCE] "
        f"triton_file={triton_path} standard_file={standard_path} "
        f"ascend_compiler_file={compiler_path}"
    )
    _emit(
        "[FP8_FA_PROVENANCE] "
        f"converter_root={converter_root or 'NA'} "
        f"converter_head={_git_head(converter_root) if converter_root else 'NA'} "
        f"opentile_ir_root={opentile_ir_root if opentile_ir_root.exists() else 'NA'} "
        f"opentile_ir_head={_git_head(opentile_ir_root) if opentile_ir_root.exists() else 'NA'}"
    )
    _emit(
        "[FP8_FA_PROVENANCE] "
        f"target={_compact(target)} backend_class={type(driver.active).__module__}."
        f"{type(driver.active).__name__} opentileas_path={opentileas_path or 'NA'} "
        f"opentileas_root={opentileas_root or 'NA'} "
        f"opentileas_head={_git_head(opentileas_root) if opentileas_root else 'NA'} "
        f"opentileas_sha256={_sha256(opentileas_binary) if opentileas_binary else 'NA'}"
    )
    _emit(
        "[FP8_FA_CAPABILITY] "
        f"supported_fp8_dtypes={supported_fp8} tl_max_signature={_compact(max_signature)} "
        f"mlir_diagnostics={os.environ.get('MLIR_ENABLE_DIAGNOSTICS')} "
        f"reproducer_prefix={os.environ.get('TRITON_REPRODUCER_PATH')}"
    )

    expected_fp8 = ("fp8e4nv", "fp8e5")
    if supported_fp8 != expected_fp8:
        raise RuntimeError(
            "active OpenTile frontend FP8 capability does not match the "
            f"validated third_party/opentile fix: expected={expected_fp8}, "
            f"actual={supported_fp8}"
        )
    _PREFLIGHT_DONE = True
    _stage("frontend_preflight_pass")


def _load_original_contract():
    """Load only the checked local originals after NPU device selection."""
    global _CONTRACT_MODULE
    if _CONTRACT_MODULE is not None:
        return _CONTRACT_MODULE

    contract_sha = _sha256(_CONTRACT_PATH)
    kernel_sha = _sha256(_KERNEL_PATH)
    if contract_sha != _EXPECTED_CONTRACT_SHA256:
        raise RuntimeError(
            f"fp8_fwd_contract.py SHA256 mismatch: {contract_sha}"
        )
    if kernel_sha != _EXPECTED_KERNEL_SHA256:
        raise RuntimeError(f"fa_forward_fp8.py SHA256 mismatch: {kernel_sha}")

    # CI plumbing only: force the sibling production source to win over a
    # same-named module cached by another testcase in a shared pytest process.
    existing_kernel = sys.modules.get("fa_forward_fp8")
    existing_path = getattr(existing_kernel, "__file__", None)
    if existing_kernel is not None and (
        existing_path is None
        or Path(existing_path).resolve() != _KERNEL_PATH.resolve()
    ):
        del sys.modules["fa_forward_fp8"]

    if "fa_forward_fp8" not in sys.modules:
        kernel_spec = importlib.util.spec_from_file_location(
            "fa_forward_fp8", _KERNEL_PATH
        )
        if kernel_spec is None or kernel_spec.loader is None:
            raise RuntimeError("cannot load local fa_forward_fp8.py")
        kernel_module = importlib.util.module_from_spec(kernel_spec)
        sys.modules["fa_forward_fp8"] = kernel_module
        kernel_spec.loader.exec_module(kernel_module)

    contract_spec = importlib.util.spec_from_file_location(
        "_opentile_fp8_fwd_contract", _CONTRACT_PATH
    )
    if contract_spec is None or contract_spec.loader is None:
        raise RuntimeError("cannot load local fp8_fwd_contract.py")
    contract_module = importlib.util.module_from_spec(contract_spec)
    sys.modules[contract_spec.name] = contract_module
    contract_spec.loader.exec_module(contract_module)
    _CONTRACT_MODULE = contract_module
    return contract_module


class _OpenTileLaunchCompat:
    """Filter Triton-Ascend-only launch options before OpenTile JIT binding."""

    def __init__(self, kernel):
        self._kernel = kernel

    def __getattr__(self, name):
        return getattr(self._kernel, name)

    def __getitem__(self, grid):
        launcher = self._kernel[grid]

        def launch(*args, **kwargs):
            removed = {
                name: kwargs.pop(name)
                for name in _TRITON_ASCEND_ONLY_LAUNCH_OPTIONS
                if name in kwargs
            }
            _stage(
                "opentile_launch_options "
                f"removed={','.join(removed) if removed else 'none'} "
                f"kept_num_stages={kwargs.get('num_stages', 'default')}"
            )
            return launcher(*args, **kwargs)

        return launch


def _install_opentile_launch_compat():
    """Adapt launch metadata only; leave the production kernel source intact."""
    kernel_module = sys.modules.get("fa_forward_fp8")
    if kernel_module is None:
        raise RuntimeError("fa_forward_fp8 module is not loaded")
    kernel = getattr(kernel_module, "_attn_fwd_fp8", None)
    if kernel is None or not hasattr(kernel, "__getitem__"):
        raise RuntimeError("cannot locate production _attn_fwd_fp8 JIT kernel")
    if not isinstance(kernel, _OpenTileLaunchCompat):
        kernel_module._attn_fwd_fp8 = _OpenTileLaunchCompat(kernel)
    _stage(
        "opentile_launch_compat_installed "
        f"filtered_options={','.join(_TRITON_ASCEND_ONLY_LAUNCH_OPTIONS)}"
    )


def _install_diagnostic_hooks(contract):
    """Add stage boundaries without changing source tensors or algorithms."""
    original_block_quantize = contract.block_quantize
    original_attention = contract.attention
    original_golden = contract.tforward_npu
    quantize_call = 0

    def logged_block_quantize(tensor, *args, **kwargs):
        nonlocal quantize_call
        quantize_call += 1
        label = ("q", "k", "v")[quantize_call - 1] if quantize_call <= 3 else str(quantize_call)
        _stage(
            f"quantize_{label}_begin "
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
        )
        result = original_block_quantize(tensor, *args, **kwargs)
        _stage(f"quantize_{label}_synchronize_begin")
        torch.npu.synchronize()
        _stage(
            f"quantize_{label}_done "
            f"dscale_shape={tuple(result[0].shape)} "
            f"quantized_dtype={result[1].dtype}"
        )
        return result

    def logged_attention(*args, **kwargs):
        _stage("production_attention_begin kernel=_attn_fwd_fp8")
        result = original_attention(*args, **kwargs)
        _stage("production_attention_returned")
        _stage("production_attention_synchronize_begin")
        torch.npu.synchronize()
        _stage(
            "production_attention_done "
            f"shape={tuple(result.shape)} dtype={result.dtype}"
        )
        return result

    def logged_golden(*args, **kwargs):
        _stage("golden_tforward_npu_begin")
        result = original_golden(*args, **kwargs)
        _stage("golden_tforward_npu_returned")
        _stage("golden_tforward_npu_synchronize_begin")
        torch.npu.synchronize()
        _stage(
            "golden_tforward_npu_done "
            f"shape={tuple(result.shape)} dtype={result.dtype}"
        )
        return result

    # test_op resolves these names from its defining module's globals, so it
    # still executes the byte-identical original body and arguments.
    contract.block_quantize = logged_block_quantize
    contract.attention = logged_attention
    contract.tforward_npu = logged_golden


def _verify_compat_source_contract():
    """Keep the temporary reduction workaround explicit and reversible."""
    kernel_text = _KERNEL_PATH.read_text(encoding="utf-8")
    reduce_nan_count = kernel_text.count("tl.max(qk, 1, propagate_nan=True)")
    compat_todo_count = kernel_text.count(
        "TODO(CONVERTER_PROPAGATE_NAN): restore propagate_nan=True"
    )
    _emit(
        "[FP8_FA_SOURCE_CONTRACT] "
        f"kernel_sha256={_sha256(_KERNEL_PATH)} "
        f"reduce_propagate_nan_true_count={reduce_nan_count} "
        f"compat_todo_count={compat_todo_count} "
        "exact_upstream_source=0 compat_no_reduce_nan_propagation=1"
    )
    if reduce_nan_count != 0 or compat_todo_count != 2:
        raise RuntimeError(
            "compatibility source must omit both reduction "
            "propagate_nan keywords and retain exactly two restoration "
            f"TODOs; live_keywords={reduce_nan_count}, todos={compat_todo_count}"
        )


def test_fp8_flash_attention_forward_e4m3fn_opentile():
    """Run one complete, production-shape A5 forward acceptance case."""
    fault_log = None
    try:
        try:
            fault_log = _REPORT_PATH.open("a", encoding="utf-8")
        except OSError:
            fault_log = None
        faulthandler.dump_traceback_later(
            180,
            repeat=False,
            file=fault_log if fault_log is not None else sys.stderr,
        )
        _stage(
            "test_begin "
            f"ci_sha256={_sha256(Path(__file__))} "
            f"contract_sha256={_sha256(_CONTRACT_PATH)} "
            f"kernel_sha256={_sha256(_KERNEL_PATH)} "
            f"upstream_kernel_sha256={_UPSTREAM_KERNEL_SHA256} "
            "exact_source=0 compat_no_reduce_nan_propagation=1 "
            f"OPENTILE_TEST_DEVICE={os.environ.get('OPENTILE_TEST_DEVICE', '0')} "
            "shape=128x8x1024x64 base_dtype=f32 kernel_dtype=e4m3fn "
            "causal=0 BM=64 BN=128 seed=20"
        )
        _frontend_preflight()
        _verify_compat_source_contract()
        with _NativeStderrCapture("production"):
            _stage("production_source_import_begin")
            contract = _load_original_contract()
            _stage("production_source_import_done")
            _install_opentile_launch_compat()
            _install_diagnostic_hooks(contract)
            _stage("diagnostic_hooks_installed")

            # Do not replace this call with locally generated tensors or a
            # rewritten golden. The original function owns the exact random
            # input, block quantization, production attention call, NPU golden,
            # and max-abs check.
            _stage("original_test_op_begin")
            contract.test_op(
                Z=128,
                H=8,
                N_CTX=1024,
                HEAD_DIM=64,
                causal=False,
                dtype=torch.float32,
                BM=64,
                BN=128,
            )
            _stage("original_test_op_done")
            _stage("final_synchronize_begin")
            torch.npu.synchronize()
            _stage("final_synchronize_done")
        _emit_native_stderr_summary(expand_tail=False)
        _emit_reproducer_summary()
        _stage("case_pass")
    except BaseException as error:
        _emit(
            f"[FP8_FA_ERROR] after={_LAST_STAGE!r} "
            f"type={type(error).__name__} message={_compact(error, 3000)}"
        )
        _emit_native_stderr_summary(expand_tail=True)
        _emit_reproducer_summary()
        # Re-raise so pytest can render the traceback.  The same error has already
        # been fsync'ed to fp8_fa_e2e_report.log in case pytest output is captured.
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if fault_log is not None:
            fault_log.close()
