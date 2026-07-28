# FP8 FlashAttention A5 daily compatibility regression v1

Copy these files into the existing `testcase/39_FA_FP8/` directory and use
the normal E2E framework entry for `ci.py`. No separate driver script is
required.

Pytest collects exactly one daily case:
`test_fp8_flash_attention_forward_e4m3fn_opentile`.

The case:

- records active Python, Triton, Converter and `opentileas` provenance;
- requires `supported_fp8_dtypes == ("fp8e4nv", "fp8e5")`;
- verifies the two temporary reduction workarounds and restoration TODOs;
- runs the compatibility-patched production E4M3FN specialization;
- preserves the original input generation, quantization, NPU golden and
  full-output `max_abs < 0.1` acceptance contract;
- captures bounded native diagnostics and reproducer metadata on failure.

This is a temporary compatibility build.  The only kernel-source differences
are the two omitted reduction arguments:

```python
# TODO(CONVERTER_PROPAGATE_NAN): restore propagate_nan=True after the Converter upgrade.
tl.max(qk, 1)
```

The two outer `tl.maximum(..., propagate_nan=tl.PropagateNan.ALL)` calls are
unchanged. Restore the two marked arguments after a future Converter frontend
upgrade adds reduction `propagate_nan`; until then this is the daily
compatibility gate.

Required code for this compatibility retest:

- OpenTileConverter branch `codex/fp8-fa-propagate-nan`
  - commit `37937ca7a79224a2a3e94fd48a19e63daa302171`
  - OpenTileIR latest `main` gitlink
    `24e15233fd3b42d8d1e1236f5d1d0ce27cea3838`
  - all MR changes are under `third_party/opentile/**`; it intentionally does
    not add `propagate_nan` support to community-owned Triton Python code
- OpenTileAS PR !322 branch `codex/fa-generic-correctness`
  - rebased commit `d1121be745a6bcf84d3e5e3938a4e5bd4eeebfb7`
  - base `8c1a8c57ef7efd5655597065d5d547ce085ee3b4`

This daily compatibility run requires the rebuilt Converter from `37937ca7` for
the OpenTileIR FP8 cast fix, while intentionally accepting that its
`tl.max` lacks reduction `propagate_nan`.  It must also use the rebuilt
`opentileas` from OpenTileAS PR !322.  Merely switching source branches
without rebuilding or changing `PYTHONPATH`/`PATH` is insufficient.

Recorded A5 acceptance:

- result: `2 passed` in the diagnostic predecessor, including the identical
  full E2E case now retained here;
- OpenTileAS CI temporary merge: `47a39c5` from PR !322;
- source head reported before merge: `d1121be7`;
- no `propagate_nan` frontend support was used.

Failure artifacts remain in the testcase directory:

- `fp8_fa_e2e_report.log`: compact single report to return;
- `fp8_fa_native_production.log`: raw native stderr;
- `/tmp/fp8_fa_repro_<pid>.*.repro.mlir`: bounded MLIR reproducers. The
  report contains their names, sizes and SHA256 values; do not paste full IR
  unless requested.

The exact source contracts are:

- `fp8_fwd_contract.py` SHA256
  `7682230827455db59f8a06d7a1e64ff58c3ef953d1e0e1760be15246927b4c2d`
- `fa_forward_fp8.py` SHA256
  - compatibility copy:
    `bca89d0df1322c3af90262403d6bac03d86dc114bb05c0d15527f4e560cc9e9c`
  - upstream original:
    `cdcce1fc9a7bcb875afaed4973a3c00a8623dd3e876280354592ca6afb5d7c70`
