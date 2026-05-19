# slang-gemm

Hermetic Bazel/Zig/Slang GEMM workspace targeting classic cuBLAS-compatible
GEMM coverage, cuBLAS correctness references, and Slang/Vulkan shader
benchmarking.

## Current Scope

Implemented API and reference surface:

- BLAS wrappers: `sgemm`, `dgemm`, `cgemm`, `zgemm`, and `hgemm`.
- Checked 64-bit entrypoints for classic GEMM calls.
- Complex `3m` wrappers: `cgemm3m`, `zgemm3m`, and `cgemm3mEx`.
- Pointer-array batched and strided-batched GEMM descriptors.
- Generic Ex families: `gemmEx`, `gemmBatchedEx`, and
  `gemmStridedBatchedEx`.
- CUDA-shaped `DataType`, `ComputeType`, and classic GEMM `Algo` enums.
- Scalar reference conversion/arithmetic for CUDA 13.1 real and complex type
  IDs, including FP16, BF16, integer, FP8, FP6, and FP4 encodings.
- cuBLAS FFI coverage for standard GEMM, Ex, batched, and strided-batched
  correctness checks.

Implemented shader assets:

- FP32 tiled Slang/Vulkan shader path, currently wired into the benchmark
  runner for regular SGEMM with rectangular shapes and all real `N/T/C` op
  combinations.
- Slang shader family targets for FP16, FP32/TF32, FP64, complex FP32,
  complex FP64, integer-to-I32, and packed low-precision-to-FP32 correctness
  paths.
- Optimized NVIDIA coop2 tensor shader targets:
  `//shaders:gemm_nvcoop2_f16_spv` and
  `//shaders:gemm_nvcoop2_f16_small_spv_gen`, plus
  `//shaders:gemm_nvcoop2_bf16_spv_gen`, wired for compatible FP16/BF16
  `GemmEx` cases with FP32 accumulation/output. The FP16 path supports
  transpose-aware layouts and strided-batched dispatch through `SV_GroupID.z`;
  the BF16 generated path currently supports `N/N`.

Important current limitation: most shader families compile and are embedded into
the benchmark binary, but optimized tensor dispatch is currently wired only for
the FP16/BF16 `GemmEx` fast paths above, FP16/BF16 strided or packed batched
`GemmEx`, and `N/N` INT8 `GemmEx` regular/batched/strided paths. Other
cuBLAS-supported cases run generic correctness/performance shaders in normal
manifest mode. In strict performance mode they are reported as blockers instead
of being counted as cuBLAS-class implementations.

cuBLASLt is intentionally out of scope for this phase.

## Requirements

- Bazel
- CUDA toolkit and cuBLAS, defaulting to `/usr/local/cuda-13.1`
- Vulkan-capable NVIDIA GPU for shader benchmarks
- Slang compiler available through the Bazel shader rules

CUDA is modeled as a pinned local SDK dependency because NVIDIA GPU libraries
are coupled to the installed driver/toolkit.

## Build And Test

Run the full suite:

```bash
bazel test //...
```

Useful individual targets:

```bash
bazel test //tests:gemm_correctness
bazel test //tests:gemm_classic_extended
bazel test //tests:gemm_manifest
bazel test //tests:gemm_cublas_correctness
bazel test //tests:gemm_shader_correctness
```

Build all shader artifacts:

```bash
bazel build //shaders:all
```

Or build specific shader families:

```bash
bazel build //shaders:gemm_nvcoop2_f16_spv
bazel build //shaders:gemm_nvcoop2_f16_small_spv_gen
bazel build //shaders:gemm_nvcoop2_bf16_spv_gen
bazel build //shaders:gemm_family_f16_scalar_spv
bazel build //shaders:gemm_family_f16_bf16_tensor_spv
bazel build //shaders:gemm_family_f32_tf32_spv
bazel build //shaders:gemm_family_f64_spv
bazel build //shaders:gemm_family_complex_f32_spv
bazel build //shaders:gemm_family_complex_f64_spv
bazel build //shaders:gemm_family_int8_int32_spv
bazel build //shaders:gemm_family_low_precision_packed_spv
```

## Default Benchmark

The default benchmark measures the currently wired FP32 SGEMM shader path for
each requested square size:

- cuBLAS `Sgemm`
- CPU scalar reference `sgemm`
- Slang/Vulkan FP32 shader

Example:

```bash
bazel run //:gemm_perf -- \
  --suite=classic,ex,batched,strided \
  --device-substr=RTX \
  --sizes=64,128,256,512,1024,2048,4096 \
  --iters=20 \
  --warmup=5
```

The default output is JSONL-style rows:

```json
{"status":"ok","api":"regular","family":"classic_sgemm","shader_family":"gemm_family_f32_tf32","shader_implemented":true,"dispatch_available":true,"op_a":"N","op_b":"N","a_type":"CUDA_R_32F","b_type":"CUDA_R_32F","c_type":"CUDA_R_32F","compute_type":"CUBLAS_COMPUTE_32F","m":512,"n":512,"k":512,"cublas_ms":0.0,"cublas_tflops":0.0,"scalar_ms":0.0,"scalar_tflops":0.0,"shader_ms":0.0,"shader_tflops":0.0,"shader_cublas_ratio":0.0}
```

`shader_cublas_ratio` is `shader_ms / cublas_ms`; lower is faster relative to
cuBLAS.

## All GEMM Manifest Benchmark

Manifest mode enumerates the classic cuBLAS-compatible GEMM surface and writes
one JSONL row per case. It covers:

- APIs: regular, checked 64-bit, pointer-array batched, strided-batched,
  complex `3m`, `GemmEx`, `GemmBatchedEx`, and `GemmStridedBatchedEx`.
- Ops: real and complex `N/T/C` combinations.
- Types: CUDA 13.1 real and complex data type enums, plus classic cuBLAS
  compute enums.
- Shapes: square sizes from `--sizes` and rectangular `MxNxK` shapes from
  `--rect-sizes`.

Run the practical full manifest:

```bash
bazel run //:gemm_perf -- \
  --manifest=all \
  --suite=classic,ex,batched,strided,3m \
  --ops=all \
  --types=all \
  --sizes=128,256,512,1024,2048,4096 \
  --rect-sizes=256x512x128,512x256x1024,1024x4096x512,4096x1024x2048 \
  --iters=20 \
  --warmup=5 \
  --only-cublas-supported \
  --output=bench.jsonl
```

`--only-cublas-supported` suppresses `cuBLAS_unsupported` rows, so the output
contains only cases that classic cuBLAS can run. The manifest regression test
requires every statically supported cuBLAS row to have a Slang shader family
and Zig/Vulkan dispatch path; `shader_unavailable` is now reserved for true
implementation gaps.

Run the strict tile-compatible performance contract:

```bash
bazel run //:gemm_perf -- \
  --manifest=all \
  --only-cublas-supported \
  --tile-compatible-only \
  --fail-every-row \
  --suite=classic,ex,batched,strided,3m \
  --ops=all \
  --types=all \
  --sizes=256,512,1024,2048,4096 \
  --rect-sizes= \
  --iters=20 \
  --warmup=5 \
  --device-substr="RTX 5090" \
  --output=bench.jsonl
```

`--tile-compatible-only` skips shapes that cannot map cleanly to the selected
family tile. `--fail-every-row` is intentionally stricter than
`--fail-on-regression`: every timed row must have `shader_cublas_ratio < 1.0`,
and cuBLAS-supported rows that only have generic correctness shaders are emitted
as `shader_unavailable` blockers.

Use a small smoke manifest while iterating:

```bash
bazel run //:gemm_perf -- \
  --manifest=all \
  --sizes=64 \
  --rect-sizes=32x64x16 \
  --iters=1 \
  --warmup=1 \
  --only-cublas-supported \
  --output=/tmp/slang-gemm-smoke.jsonl
```

Manifest row statuses:

Every row also includes `shader_family`, `shader_implemented`, and
`dispatch_available`. `shader_implemented` means a Slang shader family exists
for the API/type combination. `dispatch_available` means the Zig/Vulkan runner
can execute that shader path today.

- `ok`: cuBLAS and a matching Slang/Vulkan dispatch path were both available.
- `cuBLAS_unsupported`: the CUDA enum combination is not a classic cuBLAS GEMM
  combination.
- `shader_unavailable`: cuBLAS supports the case, but no optimized shader
  dispatch path is wired for the active benchmark contract. In normal manifest
  mode this is an implementation gap; in `--fail-every-row` mode it also marks
  generic correctness shaders that are not valid cuBLAS-class performance paths.
- `validation_failed`: a shader dispatch path exists, but the runtime returned
  no timing for that row, for example due to a Vulkan device/feature/timestamp
  issue during that dispatch.
- `perf_regression`: a timed shader row was slower than cuBLAS. With
  `--fail-on-regression`, only hardware-accelerated family geomean failures
  fail the process. With `--fail-every-row`, any row with
  `shader_cublas_ratio >= 1.0` fails the process.

Hardware-accelerated family ratios are summarized at the end of manifest runs.
The current gated accelerated families are FP16 `GemmEx`, `GemmBatchedEx`, and
`GemmStridedBatchedEx` for real `N/T/C` ops with FP32 accumulation/output;
BF16 `GemmEx`, `GemmBatchedEx`, and `GemmStridedBatchedEx` for `N/N`; and
INT8/I32 `GemmEx`, `GemmBatchedEx`, and `GemmStridedBatchedEx` for `N/N`.
FP16 requires `m % 128 == 0`, `n % 256 == 0`, and `k % 16 == 0` for the main
coop2 tile, with a small regular `N/N` specialization at `64x128`; BF16
requires `m % 64 == 0`, `n % 128 == 0`, and `k % 16 == 0`; INT8 requires
`m % 128 == 0`, `n % 256 == 0`, and `k % 32 == 0`. Individual outliers are
still labeled `perf_regression`; `--fail-on-regression` fails only when the
accelerated-family geometric mean is slower than cuBLAS.

## Shader Vs cuBLAS Table

This table is the current source of truth for which classic cuBLAS GEMM rows
have a Slang/Vulkan shader path wired in the benchmark runner.

| cuBLAS API / Case | cuBLAS Reference | Slang Shader Path | Vulkan Dispatch | Current Status |
| --- | --- | --- | --- | --- |
| `cublasSgemm`, regular and checked 64-bit, real `N/T/C` | yes | `gemm_family_f32_tf32_spv` | wired | correctness/perf reporting; not tensor-core parity yet |
| `cublasDgemm`, regular and checked 64-bit | yes | `gemm_family_f64_spv` | wired | correctness/perf reporting |
| `cublasCgemm`, regular and checked 64-bit | yes | `gemm_family_complex_f32_spv` | wired | correctness/perf reporting |
| `cublasZgemm`, regular and checked 64-bit | yes | `gemm_family_complex_f64_spv` | wired | correctness/perf reporting |
| `cublasCgemm3m` / `cublasZgemm3m` | yes | `gemm_family_complex_f32_spv`, `gemm_family_complex_f64_spv` | wired | correctness/perf reporting; not 3m tensor decomposition yet |
| `cublasGemmEx`, FP16 inputs, FP32 output/compute, tile-compatible | yes | `gemm_nvcoop2_f16_spv`; small `N/N` uses `gemm_nvcoop2_f16_small_spv_gen` | wired optimized coop2 for all real `N/T/C`; small `N/N` specialization | RTX 5090: old large-tile shader `104.79/133.60/144.22` TFLOP/s at `512/1024/2048`; strict 256³ small `N/N` row now passes with shader about `6.9` TFLOP/s vs cuBLAS about `3.5` |
| `cublasGemmEx`, BF16 inputs, FP32 output/compute, `N/N`, tile-compatible | yes | `gemm_nvcoop2_bf16_spv_gen` | wired optimized coop2 | RTX 5090: shader `67.95/75.71/80.89` TFLOP/s at `512/1024/2048`; cuBLAS `42.00/129.70/170.75`; ratios `0.618/1.713/2.111` |
| `cublasGemmEx`, FP16/BF16 inputs, FP32 output/compute, non-optimized shapes or ops | yes | `gemm_family_low_precision_packed_spv` | wired fallback for regular, batched, and strided Ex | correctness/perf reporting |
| `cublasGemmStridedBatchedEx`, FP16 inputs, FP32 output/compute, tile-compatible | yes | `gemm_nvcoop2_f16_spv` using `SV_GroupID.z` | wired optimized coop2 for all real `N/T/C` | RTX 5090, 4 batches old `N/N`: shader `118.17/139.15/144.45` TFLOP/s at `512/1024/2048`; cuBLAS `76.78/154.87/188.55`; ratios `0.650/1.113/1.305` |
| `cublasGemmEx`, INT8 inputs, INT32 output/compute, `N/N`, tile-compatible | yes | `gemm_nvcoop2_i8_spv` | wired optimized coop2 for regular, batched, and strided Ex | strict timed path, but currently slower than cuBLAS on 256³ |
| `cublasGemmEx`, INT8 inputs, INT32 output/compute, other ops | yes | `gemm_family_int8_int32_spv` | generic fallback | correctness/perf reporting; strict mode reports unsupported optimized ops as blockers |
| `cublasGemmEx`, FP32 inputs/output with TF32 compute mode | yes | `gemm_family_f32_tf32_spv` | wired generic for regular, batched, and strided Ex | RTX 5090 SGEMM generic shader `2.04/1.08/1.03` TFLOP/s vs cuBLAS `21.47/47.59/66.52`; TF32 coop shader pending |
| Classic pointer-array batched GEMM (`*gemmBatched`) | yes | real/complex family shaders | wired through packed contiguous batch dispatch in the Vulkan runner | correctness/perf reporting |
| `cublasGemmBatchedEx` pointer-array Ex | yes | FP16/BF16/FP32/INT8 family shaders | wired through packed contiguous batch dispatch in the Vulkan runner | correctness/perf reporting |
| `cublasGemmStridedBatchedEx`, BF16 inputs, FP32 output/compute, `N/N`, tile-compatible | yes | `gemm_nvcoop2_bf16_spv_gen` using `SV_GroupID.z` | wired optimized coop2 | strict timed path; still subject to row-level perf gate |
| `cublasGemmStridedBatchedEx`, INT8 inputs, INT32 output/compute, `N/N`, tile-compatible | yes | `gemm_nvcoop2_i8_spv` using `SV_GroupID.z` | wired optimized coop2 | strict timed path, but currently slower than cuBLAS on 256³ |
| `cublasGemmStridedBatchedEx`, other supported rows | yes | family shaders exist for supported static rows | wired generic fallback | correctness/perf reporting; strict mode reports unsupported optimized paths as blockers |

## Performance Target

The project goal is now strict row-level cuBLAS-class performance for the
tile-compatible suite. Normal manifest mode still reports generic correctness
shader timings, but strict mode does not treat those shaders as optimized.

On an RTX 5090 with driver `590.48.01`, an older `512,1024,2048` manifest run
reported:

| Case | 512 shader/cuBLAS TFLOP/s | 1024 shader/cuBLAS TFLOP/s | 2048 shader/cuBLAS TFLOP/s | Ratio trend |
| --- | ---: | ---: | ---: | --- |
| FP16 `GemmEx N/N` | `104.79 / 42.44` | `133.60 / 129.99` | `144.22 / 170.77` | `0.405`, `0.973`, `1.184` |
| BF16 `GemmEx N/N` | `67.95 / 42.00` | `75.71 / 129.70` | `80.89 / 170.75` | `0.618`, `1.713`, `2.111` |
| FP16 `GemmStridedBatchedEx N/N`, 4 batches | `118.17 / 76.78` | `139.15 / 154.87` | `144.45 / 188.55` | `0.650`, `1.113`, `1.305` |
| FP32 SGEMM generic shader | `2.04 / 21.47` | `1.08 / 47.59` | `1.03 / 66.52` | TF32 coop path pending |
| INT8/I32 generic shader | `0.65 / 16.21` | `0.61 / 80.14` | `0.56 / 177.97` | superseded for `N/N` Ex rows by coop2, but coop2 still needs throughput tuning |

A strict `256³` smoke run after adding `--tile-compatible-only`,
`--fail-every-row`, BF16 batched coop2, INT8 batched coop2, FP16 transpose
layouts, and the small FP16 `N/N` kernel reported:

| Status | Count | Meaning |
| --- | ---: | --- |
| `ok` | 7 | Optimized shader row was faster than cuBLAS |
| `perf_regression` | 31 | Optimized shader row ran but was slower than cuBLAS |
| `shader_unavailable` | 412 | cuBLAS-supported tile-compatible row still lacks an optimized shader path |

The strict summary was `strict_cases=450`, `strict_failures=443`, and
`strict_worst_ratio=11.337108`. Remaining performance milestones are TF32 coop,
BF16 transpose support in the SPIR-V generator, row-level small-shape tuning,
INT8 coop2 throughput tuning, FP64 CUDA-core-style tiling, complex real-kernel
decomposition, and true pointer-array batched tensor dispatch.

## Main Source Layout

- `src/gemm.zig`: public GEMM API, CUDA-shaped enums, scalar references.
- `src/cublas.zig`: CUDA/cuBLAS FFI and reference helpers.
- `src/vulkan_runner.zig`: Vulkan compute runner and shader dispatch.
- `src/manifest.zig`: manifest case enumeration and support/status logic.
- `bench/gemm_perf.zig`: benchmark CLI and JSONL reporting.
- `shaders/*.slang`: Slang GEMM shader families.
- `tests/*.zig`: scalar, cuBLAS, shader, and manifest tests.
