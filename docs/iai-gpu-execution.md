# GPU execution and possible semantic operations

## Status and scope

This document records a possible NVIDIA GPU backend for Iai. It is a future
design direction, not part of the current Phase 2 implementation. Phase 2
should nevertheless preserve the backend-independent schedule information
needed here: named-axis projections, hierarchical residual layouts,
worker-owned output blocks, bounded reduction stages, and register fragments.

The earlier host-only named-operation prototype has been withdrawn because its
multiply-add wrapper generated slower host code than an equivalent call-site
lambda. The current Iai API exposes only generic updater callables. Phase 2
must not add semantic operations; the specialization discussion below is a
possible later GPU-specific design, not an existing API commitment.

The design covers two execution levels:

1. a general CUDA-core path for any supported device-callable updater; and
2. possible specialized paths selected from an explicit future operation
   contract when an architecture offers a compatible accelerated instruction.

It does not propose implicit host/device transfers, device storage ownership,
or recognition of GEMM and tensor contraction by name.

## Motivation

The current updater interface deliberately accepts an arbitrary concrete
callable:

```cpp
[](auto& result, auto lhs, auto rhs) { result += lhs * rhs; }
```

This is sufficient for inlining and host SIMD, but C++ cannot inspect the body
of an arbitrary lambda and prove that it denotes multiply-add. A GPU backend
can invoke such a callable once per logical update on CUDA cores, provided it
is device-callable. It cannot safely replace the callable with a Tensor Core
MMA instruction because that substitution requires knowledge of the
mathematical operation, input arity, accumulator type, contraction behavior,
and permitted reassociation.

A future GPU backend may therefore justify a separate semantic operation
contract. The axis expression would still describe the contraction, while the
operation would describe only the local update. Any such API requires a new
proposal and fresh host code-generation measurements.

## Semantic operation model

An initial operation can retain the existing updater calling convention:

```cpp
namespace iai {

struct semantic_multiply_add {
  template <class Accumulator, class Left, class Right>
  IAI_HOST_DEVICE constexpr void
  operator()(Accumulator& accumulator, Left left, Right right) const {
    accumulator += left * right;
  }
};

} // namespace iai
```

`IAI_HOST_DEVICE` is illustrative. The actual annotation must be provided by
the selected device backend without making CUDA syntax a requirement of the
host-only core. The input operands are intentionally passed by value, matching
the original benchmark updaters and making their local copies independent of
the accumulator for optimization. This does not require trivially copyable
types; ordinary call viability determines whether the concrete operands can be
copied.

Internal traits attach facts that cannot be recovered from a callable body:

```cpp
template <>
struct operation_traits<semantic_multiply_add> {
  static constexpr operation_category category = operation_category::mul_add;
  static constexpr std::size_t input_arity      = 2;
};
```

The first implementation SHOULD keep this trait set closed to Iai-defined
operations. A future extension point for user operations requires a precise
semantic contract; merely allowing a user to claim that an arbitrary callable
is multiply-add would make backend substitution unsound.

No semantic operation vocabulary is currently exposed. A future backend should
introduce and distinguish an operation only when it can preserve the
operation's numerical contract and demonstrate that the wrapper does not
degrade generic host execution.

## Numerical contract

Recognizing `mul_add` does not by itself grant every transformation.
Backend selection must consider:

- whether floating-point contraction into FMA is allowed;
- whether reduction reassociation is allowed by `md_reduce`;
- input, product, and accumulator types;
- intermediate precision and conversion rules;
- NaN, signed-zero, overflow, and saturation behavior; and
- whether an ordered reduction mode has been requested.

A CUDA-core path can preserve the same local updater expression while changing
the axis traversal in ways already permitted by `md_reduce`. An MMA path may
change grouping and intermediate precision. It MUST therefore be guarded by an
explicit math policy and operation/type traits. An eventual ordered or
non-reassociating mode MUST reject any accelerated path that cannot preserve
its order.

## Backend dispatch

The dispatch hierarchy should remain structural:

```text
md_reduce expression
  -> axis, extent, alias, and mapping validation
  -> backend policy
  -> semantic operation classification
  -> schedule and residual-layout selection
  -> generic scalar/CUDA-core fragment executor
     or compatible accelerated operation executor
```

The backend MUST NOT contain branches such as `if GEMM` or `if tensor
contraction`. A multiply-add reduction is eligible because of:

- its semantic operation type;
- the incidence of free and reduction axes in each operand;
- compatible value and accumulator types;
- storage and residual layouts;
- fragment and reduction-stage shapes; and
- target instruction support.

An arbitrary updater remains valid on the generic path if the device compiler
can instantiate it. Unsupported or host-only callables must be rejected at
compile time with a device-backend diagnostic, not silently executed on the
host.

## Memory and execution contract

Iai remains a non-owning execution library. A GPU call assumes that every
operand `mdspan` refers to memory accessible from the selected device. The
caller or an interoperating runtime owns allocation, transfer, and lifetime.
Iai MUST NOT infer pointer residency and MUST NOT copy operands implicitly.

The GPU policy should make the device execution context explicit, for example:

```cpp
iai::md_reduce<K>(
  iai::cuda_policy{stream, math_policy},
  output,
  lhs,
  rhs,
  [] IAI_HOST_DEVICE (auto& result, auto lhs_value, auto rhs_value) {
    result += lhs_value * rhs_value;
  }
);
```

A later sender interface may obtain the stream scheduler from its execution
environment instead. In both forms, the backend must document whether
completion is host-synchronous or only ordered on the supplied stream.

Bindings, mappings, accessors, handles, identities, and operation objects
passed as kernel arguments must be device-copyable. Custom accessors require
an explicit device-compatibility concept. The current pointer-overlap checks
may run on the host, but device code must not dereference host-only state.

NVIDIA's current CCCL includes `cuda::std::mdspan` and memory-space accessors,
including device and managed memory support. Those facilities are a possible
future interop layer rather than a required replacement for the existing
Kokkos `mdspan` implementation. See the
[CCCL mdspan documentation](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/mdspan.html).

## GPU execution hierarchy

After Phase 2, the host execution hierarchy has a natural GPU mapping:

| Iai level | NVIDIA GPU level |
|---|---|
| Output work grid | CUDA grid |
| Worker-owned output/cache block | Thread block or cluster tile |
| Output traversal group | Rasterized neighboring thread blocks |
| Bounded reduction stage | Shared-memory pipeline stage |
| Register-resident output group | Warp/thread accumulator fragments |
| Local update operation | CUDA-core instruction or MMA atom |

A correct first kernel may assign one thread to each output element and
perform its reduction serially. A useful optimized kernel instead requires:

1. one thread block to own a disjoint output block;
2. cooperative, coalesced loading of projected input tiles;
3. a bounded reduction stage in shared memory;
4. synchronization between data movement and consumption;
5. register-resident output fragments;
6. predication for global and reduction tails; and
7. a final coalesced output epilogue.

The Phase 2 choice between input-stationary and accumulator-stationary local
schedules remains applicable. The GPU policy additionally accounts for shared
memory, registers, occupancy, warp count, asynchronous-copy support, and
instruction-level fragment requirements.

## CuTe and CUTLASS reuse

The backend SHOULD evaluate using CuTe internally for three responsibilities:

- translating an Iai residual layout into thread/value partitions;
- constructing cooperative global-to-shared and shared-to-register tiled
  copies; and
- selecting a tiled MMA from an operation/type/architecture-compatible atom.

CUTLASS 3 separates device, kernel, collective, tiled MMA/copy, and atom
levels. Its tiled operations compose hardware atoms across threads and data,
which closely matches Iai's intended separation of storage tiles, execution
blocks, reduction stages, and fragments. See NVIDIA's
[CUTLASS 3 GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html)
and
[efficient GEMM description](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html).

CuTe must remain an implementation detail. Iai's public index space and source
`mdspan` mappings stay authoritative. A backend adaptor may translate a
supported residual layout into CuTe types, but it must not expose a second
public tensor model.

CuTe's `cute::gemm` is appropriate only for compatible multiply-add semantics.
Its layout and tiled-copy machinery may still be useful to a generic
CUDA-core fragment executor.

## Shared-memory staging versus packing

Iai forbids hidden persistent packing and operand storage ownership. That rule
does not forbid an execution backend from loading a bounded working set into
hardware-managed fast storage.

GPU shared-memory staging is acceptable when:

- its lifetime is limited to one kernel work group;
- it is allocated from backend-local shared memory;
- it is not observable as an operand or returned to the caller;
- its cost is represented in schedule selection and measurements; and
- it does not create a persistent alternative layout owned by Iai.

The shared-memory layout may include padding or swizzling to avoid bank
conflicts. This is an ephemeral execution representation, analogous to host
register fragments, rather than a user-visible packed matrix. The main RFC's
"no hidden packing" invariant should be read with this distinction if a GPU
backend is adopted.

## Public API relationship

The host eager API should remain available:

```cpp
iai::md_reduce<K>(output, inputs..., operation);
```

Possible backend selection mechanisms are:

1. an explicit leading execution policy;
2. `make_md_reduce` returning a sender whose environment supplies a scheduler;
3. a sender-domain transformation that selects a device executor; or
4. an explicit `execute(operation, policy)` facility.

Automatic selection based only on pointer values is not acceptable. Backend
choice, memory accessibility, stream ordering, and synchronization must be
visible in a policy or execution environment.

## Implementation stages

1. Repair and test the existing device-offload CMake option and host/device
   annotations for `layout_tiled` mappings.
2. Add a device-compatibility layer for Iai bindings, projections, residual
   layouts, operation invocation, and lightweight tuple/array utilities.
3. Implement a synchronous scalar CUDA reference kernel over device-resident
   tiled operands.
4. Map Phase 2 output blocks and reduction stages to CUDA thread blocks and
   shared memory using CUDA-core fragments.
5. Add a CuTe translation for supported residual layouts and tiled copies.
6. If generic execution establishes a need, propose an explicit semantic
   operation contract and prove that it preserves host code generation.
7. Add one architecture- and type-constrained multiply-add MMA path.
8. Benchmark GEMM and tensor contraction against the generic CUDA-core path
   and relevant CUTLASS references.
9. Add stream-ordered sender integration without changing the eager API's
   completion contract.

## Validation requirements

- The scalar GPU path matches the host semantic reference for every supported
  mapping, broadcast, reduction-axis set, tail, `out`, and `inout` case.
- Semantic operations and equivalent lambdas agree on the generic path.
- Unsupported operation/type/layout combinations fall back to CUDA cores or
  fail at compile time; they never select an incompatible MMA.
- Each output block has one exclusive owner.
- Shared-memory staging is race-free under sanitizer tooling.
- Device code contains no host-only function calls or runtime axis search in
  hot loops.
- Accelerated paths report their accumulation and contraction policy.
- Benchmarks separate transfer time from kernel execution because Iai does not
  own transfers.

## Open questions

1. Should semantic operation traits remain closed or admit a reviewed user
   customization mechanism?
2. Which math-policy levels distinguish strict local evaluation, FMA
   contraction, and MMA reassociation?
3. Should the first device backend use Kokkos `mdspan`, CCCL
   `cuda::std::mdspan`, or a narrow compatibility concept over both?
4. Which CUDA compiler is required for the header-only operation types and
   arbitrary device-callable updater instantiations?
5. Should stream completion be exposed only through the sender interface,
   leaving eager `md_reduce` host-synchronous?
6. How much shared-memory permutation is acceptable before it should be
   described as explicit backend-local packing?

## External references

- [CUDA C++ device and host functions](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)
- [CUDA extended-lambda requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html)
- [CUTLASS overview](https://docs.nvidia.com/cutlass/latest/overview.html)
- [CUTLASS 3 GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html)
