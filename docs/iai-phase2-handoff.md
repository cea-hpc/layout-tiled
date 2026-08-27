# Iai Phase 2 implementation handoff

## Objective

Implement Phase 2, "native-tiled performance maturation," from
[`iai-design-notes.md`](iai-design-notes.md). The goal is to improve generic
`iai::md_reduce` execution for GEMM and tensor contraction without recognizing
either kernel by name, changing the public axis expression, allocating packed
storage, or regressing the already satisfactory large GEMV results.

This handoff is intended for a fresh implementation session. Read the linked
documents and inspect the current tree before editing; the worktree contains
the complete uncommitted Phase 1 extraction and belongs to the user. Do not
reset, restore, or discard unrelated changes.

## Required reading

1. [`iai-design-notes.md`](iai-design-notes.md), especially the locality model,
   execution hierarchy, CuTe prior art, Phase 1 status, and Phase 2 plan.
2. [`iai-phase1-results.md`](iai-phase1-results.md) for controlled final
   measurements and schedule crossovers.
3. [`iai-gemm-tile-sweep.md`](iai-gemm-tile-sweep.md) for the asymmetric
   physical-tile experiment and its interpretation.
4. [`tensor-contraction-sol.md`](tensor-contraction-sol.md) for the contraction
   Speed-of-Light methodology.
5. [`iai-cache-budgets.md`](iai-cache-budgets.md) for the current temporary
   manual policy values.
6. [`iai-gpu-execution.md`](iai-gpu-execution.md) and
   [`iai-std-execution.md`](iai-std-execution.md) only as future compatibility
   constraints. Phase 2 does not implement either backend.

## User priorities and settled decisions

- Thesis results concentrate on GEMV, GEMM, and tensor contraction.
- GEMV at large sizes is already satisfactory and is primarily a regression
  guard.
- GEMM should narrow its same-session gap to BLAS.
- Tensor contraction should demonstrate the flexibility of one generic
  multidimensional executor and improve at cache-capacity-crossing sizes.
- `layout_tiled` owns only native tiled mappings. All execution abstractions
  remain under `include/iai`.
- `in`, `out`, and `inout` are lazy projected bindings. `md_reduce` remains an
  eager terminal function.
- Reduction updates remain ordinary callables. The benchmarks use direct
  call-site lambdas; do not reintroduce named updater wrappers during Phase 2.
- Physical storage tiles, cache blocks, bounded reduction stages, register
  fragments, and parallel work items are distinct concepts.
- Iai does not allocate or persistently pack operand storage.
- Schedule selection derives from axes, projections, mappings, types, extents,
  and backend policy. It must not recognize GEMM or contraction by name.
- Cache budgets must not become universal hardware claims. Phase 2 introduces
  compile-time policy injection; automatic topology discovery remains Phase 7.
- The GEMM benchmark's default physical tile is 64x32x64, selected from the
  Tiger Lake sweep. Existing CMake caches may retain earlier values and should
  be reconfigured explicitly when benchmarking.

## Current implementation map

Public API:

```text
include/iai/bindings.hpp
include/iai/md_reduce.hpp
```

Mapping-independent analysis:

```text
include/iai/detail/axis.hpp
include/iai/detail/projected_view.hpp
include/iai/detail/md_reduce.hpp
include/iai/detail/access_transform.hpp
```

Native tiled recognition and views:

```text
include/iai/detail/layout_tiled_mapping.hpp
include/iai/detail/native_tiled_view.hpp
include/iai/detail/native_tiled_reduce.hpp
```

Optimized execution and policy:

```text
include/iai/detail/native_tiled_fragment_reduce.hpp
include/iai/detail/native_tiled_cache_block.hpp
include/iai/detail/native_tiled_blocked_fragment_reduce.hpp
include/iai/detail/native_tiled_reduce_policy.hpp
```

Tests:

```text
tests/layout_tiled/iai.test.cpp
tests/layout_tiled/iai_reduce.test.cpp
```

Benchmarks:

```text
benchmarks/gemm.cpp
benchmarks/gemv.cpp
benchmarks/tensor_contraction.cpp
```

## Phase 1 summary

Phase 1 separated execution from storage ownership: `layout_tiled` now
provides only its four native tiled mapping families, while the independently
exported `iai::iai` target owns projected bindings and eager multidimensional
execution. It introduced mapping-independent axis validation and projection,
explicit recognition and residual slicing for the known tiled mappings, a
scalar reference reducer, complete-reduction register fragments, and a
cache-blocked fragment path with rectangular physical-tile groups. Invariant
tile binding and address work were hoisted into reusable projected tile-range
descriptors.

Automatic host dispatch now chooses scalar, complete-fragment, or
blocked-fragment execution from compatibility and mapping-derived working-set
metrics, with forced internal paths for deterministic testing. The temporary
cache budgets remain documented, replaceable constants rather than hardware
claims. A short-lived named-updater prototype was removed after controlled GCC
and Clang comparisons found slower code than the equivalent direct lambda.
GEMM, GEMV, and tensor contraction therefore use local updater lambdas.

Controlled Tiger Lake results and the subsequent physical-layout sweep are in
`iai-phase1-results.md` and `iai-gemm-tile-sweep.md`. The blocked schedule
removed most of the original normalized GEMM cache-traffic pathology, large
GEMV remained close to BLAS/DT, and tensor contraction stayed faster than DT
for every measured shape. The final 64x32x64 GEMM layout improved the 4096
case by 19.6% over its 32x32x32 controls and reduced LLC misses by 43.2%; its
same-session time was 1.73x BLAS.

## What Phase 1 already guarantees

- Compile-time validation of binding roles, dense axes, projections, static
  extents, reduction axes, and updater signatures.
- Runtime checks for dynamic extents, tile divisibility, and ordinary pointer
  aliasing in debug builds.
- Recognition of the four supported `layout_tiled` mapping families.
- Lazy native tile grids and `native_tile_range` descriptors preserving source
  accessors and data handles.
- A scalar semantic reference executor.
- A complete-reduction MRxNR fragment executor.
- A reduction-blocked fragment executor that consumes a bound input tile
  across every output fragment before advancing K.
- Rectangular groups of native tiles and a mapping-derived footprint/reuse
  metric.
- Automatic selection between scalar, complete-fragment, and blocked-fragment
  paths, with internal overrides for deterministic tests and benchmarks.
- Generic updater callables, with direct call-site lambdas used by the current
  benchmarks and tests.
- OpenMP ownership of disjoint output blocks and serial fallback.

Preserve these properties throughout Phase 2. Add forced paths for every new
schedule before changing automatic selection.

## Known limitations to address

### Symmetric candidate generation

`native_tiled_reduce_policy.hpp` currently enumerates six pairs of
`(free-axis factor, reduction-axis factor)`. Every free axis receives the same
factor and every reduction axis receives the same factor.

This is too restrictive for GEMM and especially tensor contraction. For
`C[b,i,j] += A[b,i,k] * B[b,k,j]`, a batch block of one may be profitable while
row and column blocks should be larger. Phase 2 must enumerate bounded factors
independently per named axis.

### Conflated reduction stage and fragment lifetime

The blocked executor currently visits one native reduction tile, then every
output fragment. It reuses the bound input tile set, but each fragment reloads
and stores partial output after that one tile. A rectangular reduction group
changes traversal and ownership, yet the register fragment lifetime remains
one native K tile.

The complete executor has the opposite tradeoff: one output fragment remains
live across the complete reduction, but projected inputs are revisited for
each fragment.

Phase 2 must represent and compare both input-stationary and
accumulator-stationary local schedules, including a bounded register-resident
group that may span several native reduction tiles.

### Incomplete hierarchy

The code distinguishes storage tiles and cache-block descriptors, but does not
yet have independent representations for:

- a worker-owned output/cache block;
- a bounded reduction stage;
- a register-resident output-fragment group; and
- an output-block traversal/raster order.

These shapes must be explicit and independently testable before adding a more
complex cost model.

### Initial cost model

The current selector ranks candidates primarily by projected reused bytes,
tile volume, and footprint under one cache-block budget. It does not fully
model partial-accumulator traffic, register pressure, available output
parallelism, distinct cache levels, or traversal order.

For the 64x32x64 FP64 GEMM storage shape, the A, B, and C unit tiles total 64
KiB, already exceeding the temporary 32 KiB cache-block budget. The current
policy therefore falls back to a unit physical block even though a larger
private cache may support useful grouping. Phase 2 should use distinct
register, reduction-stage, and worker-private cache budgets supplied by a
compile-time policy.

## Recommended implementation sequence

### Step 0: protect the baseline

- Configure a fresh build directory with tests, checks, OpenMP, BLAS, and
  `CMAKE_CXX_FLAGS=-march=native`.
- Run all existing Iai tests and checked benchmark kernels at small compatible
  sizes.
- Record the compiler, configuration, and current generated-code evidence for
  vector FMA.
- Do not rerun the complete performance matrix until new execution code exists.

### Step 1: introduce explicit schedule types

Define provider-independent native schedule descriptors for:

- output/cache-block shape in physical tile coordinates;
- reduction-stage shape in physical tile coordinates;
- register-fragment-group shape;
- local stationarity choice;
- output-block traversal order; and
- policy-provided resource budgets.

Keep the current path expressible by these types. Add constexpr tests for rank
2, GEMM rank 3, contraction rank 4, broadcasts, and multiple reduction axes.
Do not change automatic dispatch yet.

### Step 2: complete hierarchical residual slicing

Extend the restricted CuTe-inspired lowering from a range origin to a
rectangular group of native tiles:

```text
source tiled mapping
  -> (local tile, tile grid)
  -> rectangular execution division
  -> fixed output/reduction slice
  -> residual tile and local-element modes
  -> conservative contiguous coalescing
```

Partially evaluate coordinates at output-block, reduction-stage,
physical-tile, and fragment levels. Tests must prove that custom accessors and
handles are preserved and that fixed coordinates do not reappear as global
tiled mapping, division, or remainder in hot local loops.

Also preserve alias provenance through projection and binding. In particular,
an output/input non-overlap fact established by the `md_reduce` contract should
reach the optimized residual access path in a compiler-usable form. Do not
invent a no-alias relationship between two inputs: they may legally share
storage. Any optimization contract must degrade conservatively for custom
accessors whose address ranges cannot be proved.

### Step 3: implement bounded per-axis candidate search

- Generate factors independently for every free and reduction axis.
- Bound the search to a small deterministic set derived from tile counts and
  policy limits; do not build an unbounded runtime autotuner.
- Deduplicate clamped shapes.
- Derive each operand's distinct projected tile count and reuse factor.
- Include output partial traffic, register feasibility, working-set bytes, and
  minimum parallel work count in ranking.
- Ensure the contraction policy can select batch factor one with larger I/J
  factors.
- Keep schedule selection kernel-name independent.

Test candidate generation and ranking with injected synthetic policies rather
than host cache discovery.

### Step 4: add forced local schedule variants

Implement and test:

1. input-stationary blocked execution, retaining a bound input tile/stage
   across the profitable output group; and
2. accumulator-stationary blocked execution, retaining a bounded output
   fragment group across several native reduction tiles.

Register grouping must be constrained by a policy budget and compile-time
fragment types. Avoid a runtime-sized accumulator array in hot code. Retain
full and tail paths, but make them structurally distinct: the full-tile and
full-reduction-stage path should expose compile-time extents and a bounded,
policy-selected K unroll factor, while residual tests and shortened extents
remain in the tail path.

Count input bindings and output accesses in deterministic tests. Verify that
each operation occurs only at the loop level where its coordinates become
variant.

### Step 5: add traversal and parallel work policies

Derive output-block order from operand projections and mapping order. Compare
orders that retain an input region across adjacent blocks, analogous to
cache-aware tile rasterization. Preserve exclusive output ownership and enough
independent work for four OpenMP workers.

Do not introduce a GEMM-specific I/J order. The policy should reason that an
operand omitting a varying output axis is reusable across that axis.

After binding and traversal invariants are explicit, add an optional
mapping-derived software-prefetch decision to the schedule policy. Place
prefetch only at reduction-stage boundaries, select its operand and distance
from the projected layout and traversal, and keep a forced-off control. Treat
prefetch as an independently measured refinement, not as a required property
of every schedule.

### Step 6: connect automatic selection

Only after forced variants are correct and measurable:

- add the new candidates to automatic policy;
- inject separate register, reduction-stage, and private-cache budgets through
  a compile-time policy type;
- keep deterministic conservative defaults;
- preserve force-scalar, force-complete, and force-blocked testing controls;
  and
- add explicit overrides for the new stationarity/traversal choices if needed
  for diagnosis.

Do not add runtime cache discovery, environment variables, global build
macros, or kernel-name exceptions in this phase.

### Step 7: generated-code and performance validation

Inspect assembly before interpreting timings. Confirm:

- updater invocation is inlined;
- compatible paths retain vector FMA;
- accumulator group dimensions remain compile-time constants;
- proven output/input non-overlap survives projection and binding without a
  false no-alias assumption between inputs;
- full-tile/full-stage loops have static bounds and the intended bounded K
  unrolling, with residual branches absent from that hot path;
- global tiled mapping, division, and remainder are absent from hot local
  loops;
- projected tile binding is hoisted;
- policy-enabled prefetch, when present, is issued at the intended
  reduction-stage boundary and can be disabled independently; and
- no allocation or indirect call appears.

Then benchmark forced variants and automatic selection. Use same-session
controls and report time, instructions, L1D misses, LLC references, and LLC
misses normalized by logical work.

## Build and test commands

A fresh multi-configuration build avoids stale GEMM tile cache values:

```sh
cmake -S . -B build/phase2 -G "Ninja Multi-Config" \
  -DCMAKE_CXX_FLAGS=-march=native \
  -DLAYOUT_TILED_BUILD_TESTS=ON \
  -DLAYOUT_TILED_BUILD_BENCHMARKS=ON \
  -DLAYOUT_TILED_BENCH_ENABLE_CHECKS=ON \
  -DLAYOUT_TILED_BENCH_ENABLE_OMP=ON \
  -DLAYOUT_TILED_BENCH_ENABLE_BLAS=ON

cmake --build build/phase2 --config Debug \
  --target layout_tiled.tests.iai layout_tiled.tests.iai_reduce -j 2

ctest --test-dir build/phase2 -C Debug --output-on-failure

cmake --build build/phase2 --config Release \
  --target layout_tiled.benchmarks.gemm \
           layout_tiled.benchmarks.gemv \
           layout_tiled.benchmarks.tensor_contraction -j 2
```

Run checked and measured processes on four distinct physical cores:

```sh
taskset --cpu-list 0,1,2,3 \
  env OMP_NUM_THREADS=4 \
      OMP_DYNAMIC=FALSE \
      OMP_PROC_BIND=close \
      OMP_PLACES='{0},{1},{2},{3}' \
      OPENBLAS_NUM_THREADS=4 \
  build/phase2/benchmarks/Release/layout_tiled.benchmarks.gemm \
      --size 128 --implementation iai-auto
```

The quoted `OMP_PLACES` value is intentional. On the Tiger Lake system, Linux
CPUs 0, 1, 2, and 3 are distinct physical cores even though hwloc logical
indices are L#0, L#2, L#4, and L#6. OpenMP placement uses the Linux processor
IDs in the explicit place list.

Before performance runs, ask the user to confirm that the CPU is again in the
performance governor and locked to the intended fixed frequency. Do not change
frequency policy without authorization.

## Benchmark matrix

Use existing results as historical evidence, not as interchangeable timing
controls:

```text
results/phase1/tigerlake-i5-1135g7-4c-base-2.4ghz/
results/tile-sweep/tigerlake-i5-1135g7-4c-base-2.4ghz/
```

Minimum final matrix:

| Kernel | Sizes/configurations | Comparisons |
|---|---|---|
| GEMM | 512, 1024, 2048, 4096; 64x32x64 default | BLAS, DT, TV, Phase 1 auto, forced new paths |
| GEMV | 1024, 4096, 8192, 16384 | BLAS, DT, TV, Phase 1 auto, new auto |
| Tensor contraction | 32, 64, 128, 256; L1/L2/RECT/BATCH1 | DT, Phase 1 auto, forced new paths, new auto |

The 512 tensor case was intentionally dropped because it is too long. The 256
case is the largest required contraction run.

For rapid development, screen GEMM at 1024 and contraction at 64 or 128, then
advance only promising generic schedules. Always retain bracketing controls
when timing drift is plausible.

Nanobench hardware counters cover the calling worker only, not the entire
OpenMP team. Label that scope and compare normalized relative profiles; do not
describe them as whole-team events.

## Acceptance criteria

- All Phase 1 correctness and access-count tests remain valid.
- Storage tile, output/cache block, reduction stage, register group, and
  traversal order have distinct representations.
- Candidate factors can differ across batch, row, column, and reduction axes.
- Input-stationary and accumulator-stationary paths are independently forced
  and tested.
- Multiple reduction axes, broadcasts, mixed value types, `out`, `inout`, and
  tails remain correct.
- Optimized bound views preserve only alias facts proved by the execution
  contract, and custom accessors retain a conservative fallback.
- Full and residual paths are distinct, and software prefetch remains an
  optional compile-time policy choice with a forced-off baseline.
- No kernel-name recognition, allocation, persistent packing, or public block
  provider is introduced.
- GEMV does not materially regress at 8192 and 16384.
- GEMM narrows its same-session gap to BLAS over the 64x32x64 Phase 1 result.
- Tensor contraction improves at one or more cache-capacity-crossing cases
  while retaining its advantage over DT/TV.
- Every performance conclusion is supported by elapsed time and instruction/
  cache-event context, not cache misses alone.

## Stop conditions and pitfalls

- Do not tune fixed cache constants until forced schedule variants show where
  time is spent.
- Do not assume a larger modeled reuse factor is faster; account for fragment
  reloads, address work, parallelism, and register pressure.
- Do not make a physical tile microkernel-sized merely to imitate BLAS or GPU
  register tiles. The tile sweep showed that small I/J physical tiles are
  substantially slower in the current storage/execution hierarchy.
- Do not interpret sender integration or GPU execution as Phase 2 work. The
  new schedule representations should enable them later, but host performance
  remains this phase's scope.
- Do not run the complete final benchmark matrix until code-generation checks
  and small screens are satisfactory.
