# RFC: Multidimensional algorithms over `mdspan`

- **Status:** Draft
- **Authors:** TBD
- **Target version:** Initial extraction from `layout_tiled::reduce`
- **Tracking issue:** TBD
- **Implementation:** Not started

## Summary

This RFC proposes a header-only library of multidimensional loop-execution
algorithms over `mdspan` operands:

```cpp
iai::md_map(...);
iai::md_reduce<ReductionAxes...>(...);
iai::md_stencil(...);
```

Users describe computation in a **multidimensional index space** using named
axes, operand projections, broadcasts, reductions, and neighborhood offsets.
Each operand's `mdspan` mapping describes how coordinates in that index space
map into a **linear storage space**. The implementation inspects those mapping
types and partitions the multidimensional index space into cache- and
SIMD-friendly execution regions.

`mdspan` is the only public data abstraction. The initial implementation
supports exactly two mapping classes:

1. the standard affine `mdspan` layouts; and
2. the known native tiled mappings from `layout_tiled`.

Storage classification, logical cache blocks, and native tile handles are
internal implementation details. The library does not expose a block-provider
API, optimize arbitrary custom mappings, pack data, allocate operand storage,
or silently rearrange layouts.

The existing `layout_tiled::reduce` implementation is the prototype for typed
axis analysis, native-tile access, output scheduling, scalar execution, and the
register-fragment reduction kernel.

## Motivation

`layout_tiled::reduce` demonstrates that a high-level axis expression can
produce code comparable to a handwritten blocked kernel. GEMM is expressed as:

```cpp
enum : std::size_t { I, J, K };

layout_tiled::reduce<K>(
  layout_tiled::out<I, J>(C, 0.0),
  layout_tiled::in<I, K>(A),
  layout_tiled::in<K, J>(B),
  [](auto& c, auto a, auto b) { c += a * b; }
);
```

This expression contains the mathematical relationships but does not prescribe
loop order, cache blocking, vector direction, thread placement, or register
fragment shape. The compiler sees concrete axis lists, mappings, tile extents,
value types, accessors, and the updater type, allowing the implementation to
derive and specialize the schedule.

The prototype is currently coupled to native tiled mappings and exposed under
the `layout_tiled` namespace. The useful abstraction is broader:

- a pointwise or broadcast map requires the same typed index-space projection;
- a reduction over ordinary row-major data can use logical cache blocking;
- a stencil can use affine neighborhood access and layout-aware tile binding;
- mixed standard and tiled operands can participate in one index space; and
- backend-independent analysis can be reused by serial, OpenMP, and later
  device executors.

A separate library also clarifies responsibility. Layout mappings describe
storage. Multidimensional algorithms describe computation and execution.

## Terminology and normative language

The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are
to be interpreted as normative design requirements.

### Multidimensional index space

The multidimensional index space is the semantic coordinate system in which an
algorithm is expressed. It contains:

- logical axes and extents;
- free and reduction axes;
- operand projections and broadcasts;
- stencil offsets and neighborhoods; and
- coordinates at which outputs are produced.

For GEMM, `(I,J,K)` is the complete index space, `(I,J)` is the output index
space, and `K` is the reduction index space.

### Linear storage space

The linear storage space is the one-dimensional address space consumed by
hardware. An operand's `mdspan` mapping converts its multidimensional
coordinates into offsets in that space.

Relevant storage properties include:

- strides and stride-one dimensions;
- standard versus natively tiled organization;
- physical tile extents;
- tile inner and outer order; and
- accessor and data-handle semantics.

### Execution partition

An execution partition is a region of the multidimensional index space assigned
to a loop nest, worker, logical cache block, physical tile, or register
fragment. It does not imply copied or newly owned storage.

The lowering model is:

```text
multidimensional index space
        |
        | inspect mdspan mapping types
        v
execution partitions and register fragments
        |
        | affine or native-tile address lowering
        v
linear storage space
```

Documentation and implementation comments SHOULD use “iteration space,”
“output index space,” “reduction index space,” “physical tile space,” and
“linear storage space.” They SHOULD avoid using “domain” for these concepts.

## Guide-level explanation

### Bind operands to named axes

Each binding associates operand dimensions with axes in the complete index
space:

```cpp
enum : std::size_t { I, J, K };

auto a = iai::in<I, K>(A);
auto b = iai::in<K, J>(B);
auto c = iai::out<I, J>(C, 0.0);
```

Axis order maps directly to `mdspan` dimension order. An operand that omits an
axis is invariant along that axis and broadcasts naturally. For example,
`in<J>(bias)` broadcasts one bias value across every `I` coordinate.

Bindings are lightweight wrappers around the supplied `mdspan`. They do not
own, allocate, pack, or erase the type of operand storage.

### Map an index space

`md_map` traverses an output index space without removing an axis:

```cpp
iai::md_map(
  iai::out<I, J>(C),
  iai::in<I, J>(A),
  iai::in<J>(bias),
  [](auto a, auto bias_j) { return a + bias_j; }
);
```

The implementation chooses a stride-one output dimension for SIMD, traverses
other dimensions in mapping-preferred order, and binds projected or broadcast
inputs outside the innermost work where possible.

The exact output/callback convention in this example remains unresolved. A
mutable-output form is also under consideration:

```cpp
iai::md_map(
  iai::out<I, J>(C),
  iai::in<I, J>(A),
  [](auto& c, auto a) { c = transform(a); }
);
```

### Reduce axes from an index space

`md_reduce` removes one or more axes and accumulates into each output
coordinate:

```cpp
iai::md_reduce<K>(
  iai::out<I, J>(C, 0.0),
  iai::in<I, K>(A),
  iai::in<K, J>(B),
  [](auto& c, auto a, auto b) { c += a * b; }
);
```

Using `inout` initializes accumulators from existing output values:

```cpp
iai::md_reduce<K>(
  iai::inout<I, J>(C),
  iai::in<I, K>(A),
  iai::in<K, J>(B),
  updater
);
```

The implementation owns complete output execution partitions. It keeps scalar
or register-fragment accumulators live across the full reduction and stores
them after all reduction coordinates have been visited.

### Express a stencil neighborhood

A stencil maps an output coordinate from affine offsets around one or more
inputs. A proposed five-point API is:

```cpp
constexpr auto cross = iai::offsets<
  iai::offset<-1,  0>,
  iai::offset< 0, -1>,
  iai::offset< 0,  0>,
  iai::offset< 0, +1>,
  iai::offset<+1,  0>
>;

iai::md_stencil(
  iai::interior_out<I, J, 1>(output),
  iai::neighborhood<I, J>(input, cross),
  [](auto neighbors) {
    return neighbors[0] + neighbors[1] + neighbors[2]
         + neighbors[3] + neighbors[4];
  }
);
```

An equivalent shifted-binding expression could lower into the same internal
access description:

```cpp
iai::md_map(
  iai::interior_out<I, J, 1>(output),
  iai::shifted<-1,  0>(iai::in<I, J>(input)),
  iai::shifted< 0, -1>(iai::in<I, J>(input)),
  iai::shifted< 0,  0>(iai::in<I, J>(input)),
  iai::shifted< 0, +1>(iai::in<I, J>(input)),
  iai::shifted<+1,  0>(iai::in<I, J>(input)),
  stencil_callable
);
```

The neighborhood footprint is independent of storage layout. For native tiled
storage, the executor binds the center and neighboring physical tiles and
separates tile interiors from edges. For standard affine storage, it operates
over logical cache blocks and ordinary strides.

### Use standard or tiled `mdspan`s

The public expression is unchanged by storage choice:

```cpp
std::mdspan<double, matrix_extents, std::layout_right> standard_A;
std::mdspan<double, matrix_extents,
            layout_tiled::layout_rtr<16, 512>> tiled_A;
```

For a standard mapping, IAI creates logical execution blocks over the existing
strides. For a known tiled mapping, it aligns execution with physical tiles and
uses affine inner-tile views. No storage is packed or copied in either case.

A call MAY mix standard and tiled operands. The shared multidimensional index
space, rather than any one operand, remains the semantic source of truth.

## Reference-level explanation

### Public algorithm set

The initial public API MUST contain:

```cpp
iai::md_map(...);
iai::md_reduce<ReductionAxes...>(...);
iai::md_stencil(...);
```

The initial API MUST NOT contain `md_scan`.

The initial API SHOULD NOT expose `md_iter`. A future expert-oriented primitive
may be proposed separately after the semantic and scheduling model is stable.

### Public data model

Every operand MUST be an `mdspan` or a lightweight binding that directly owns
an `mdspan` value.

The public API MUST NOT expose:

- block or tile providers;
- packed-panel objects;
- internal execution blocks;
- storage ownership abstractions;
- mapping-classification tags; or
- backend-specific fragments.

The implementation MUST preserve source accessor and data-handle semantics.

### Binding model

`in<Axes...>(view)` marks a read-only operand and records its dimension-to-axis
projection.

For reductions, `out<Axes...>(view, identity)` initializes each result from an
explicit identity, and `inout<Axes...>(view)` initializes it from the current
output value.

Bindings MUST retain concrete view, axes, identity, and callable types through
specialization. They MUST NOT allocate, use virtual dispatch, or construct a
runtime expression graph.

### Compile-time validation

For `md_reduce`, the implementation MUST validate:

- exactly one output binding appears first;
- one or more input bindings precede the callable;
- binding rank equals the number of named axes;
- axes are unique within every binding and reduction list;
- the union of axis IDs is dense in `[0, problem_rank)`;
- output axes equal the complement of reduction axes;
- every reduction axis occurs in at least one input;
- shared static logical extents are compatible; and
- the updater has the required argument and result types.

For `md_map`, the implementation MUST validate:

- exactly one output binding;
- every ordinary projected input axis belongs to the output index space; and
- shared static logical extents are compatible.

For `md_stencil`, the implementation MUST additionally validate:

- each offset rank matches the neighborhood rank;
- output axes identify the unshifted neighborhood axes;
- a boundary policy defines every referenced global coordinate; and
- the neighborhood footprint is supported by the selected executor.

Dynamic extents MUST be merged by logical axis and validated before execution.

### Supported mappings

The core implementation MUST support these standard mappings:

- `std::layout_right`;
- `std::layout_left`; and
- `std::layout_stride` when runtime strides satisfy the selected execution
  path.

An optional integration component MUST support the known `layout_tiled`
mappings without making the core standard-layout implementation depend on
`layout_tiled`.

Other custom mappings MUST be rejected at compile time with a useful
diagnostic. The library MUST NOT provide a generic mapping fallback merely
because a type satisfies the broad `mdspan` mapping requirements.

Mapping classification MUST be derived internally from each
`mdspan::mapping_type`. The mechanism MUST NOT appear in public signatures.

### Dependency structure

The intended dependency direction is:

```text
iai core
  -> mdspan
  -> standard-layout execution

layout_tiled
  -> mdspan
  -> tiled mappings

iai-layout-tiled integration
  -> iai
  -> layout_tiled
  -> native tiled execution
```

A possible integration include is:

```cpp
#include <iai/layout_tiled.hpp>
```

The call site still passes only `mdspan` bindings.

### Standard affine lowering

For standard layouts, the implementation MUST partition the multidimensional
index space into logical cache blocks over the original storage. A local block
retains the source handle, accessor, origin, extents, and strides. It MUST NOT
own or copy elements.

For row-major two-dimensional storage, local access has the form:

```text
address(i,j) = block_base + i * leading_stride + j
```

The block need not be contiguous as a whole. A unit-stride inner dimension is
sufficient for the corresponding SIMD fast path.

`layout_stride` SHOULD receive a SIMD path only when the selected SIMD-axis
stride is one. The implementation MAY use a small runtime dispatch or loop
version and MUST otherwise use a correct general path.

Logical cache blocks MUST support shortened global fringes.

### Native tiled lowering

For known tiled mappings, the implementation MUST preserve:

- source element and reference semantics;
- accessor and data-handle type;
- static physical tile extents;
- native inner layout; and
- native outer tile order.

Binding one physical tile SHOULD:

1. form the logical tile origin from tile coordinates and static extents;
2. invoke the outer tiled mapping once;
3. offset the source handle through its accessor; and
4. construct a static-extents affine inner-tile `mdspan`.

Given

```text
x[a] = tile_coordinate[a] * tile_extent[a] + local_coordinate[a],
```

the outer tiled mapping handles the first term at tile-binding time. Hot local
loops use only the affine inner mapping.

Native tiled storage retains its divisible-extent requirement.

### Mixed mappings

A call MAY combine supported standard and tiled operands. Execution partitions
belong to the shared index space. Native tile boundaries constrain axes carried
by native operands; standard operands cover the same coordinates using affine
strides.

The initial implementation SHOULD require native tiled operands sharing an axis
to have identical physical tile extents on that axis. Standard operands do not
participate in this physical compatibility requirement.

Reconciling mismatched native tile extents is deferred.

### Shared execution hierarchy

All algorithms SHOULD lower through this hierarchy:

```text
public expression
  -> compile-time axis/access validation
  -> multidimensional index-space construction
  -> compile-time mapping classification
  -> output-space partitioning
  -> flattened output work items
  -> algorithm-specific block executor
  -> affine or native-tile accesses
  -> fully inlined callable
```

The implementation SHOULD share axis analysis, index-space construction,
mapping analysis, coordinate projection, output work assignment, and backend
dispatch. It MUST retain separate map, reduction, and stencil block executors.

### Output work assignment

The host backend SHOULD flatten only independent output execution partitions.
It SHOULD decode N-D output coordinates in the output mapping's preferred
order.

When OpenMP is enabled, the initial schedule is:

```cpp
#pragma omp parallel for schedule(static)
```

Workers MUST own disjoint output partitions. Reduction axes MUST NOT be split
between workers in the initial implementation.

Without OpenMP, the same work-item order MUST execute serially.

### SIMD and register-unrolled axes

For `layout_right`, the final output dimension is the preferred stride-one
axis. For `layout_left`, the first output dimension is preferred.

The preferred stride-one output axis SHOULD become the SIMD axis. An adjacent
output axis SHOULD become the register-unrolled axis. Remaining output axes
identify SIMD groups or register-fragment planes.

For `layout_stride`, the chosen path MUST validate the relevant runtime stride.

### `md_map` execution

Within one output partition, the map executor SHOULD:

1. bind the output block or physical tile;
2. bind projected input blocks or tiles;
3. traverse non-vector axes in mapping-preferred order;
4. traverse the stride-one output axis with a SIMD loop;
5. project or broadcast input coordinates; and
6. assign or update the output according to the final callback convention.

No accumulator survives between output coordinates.

Every input carrying the SIMD axis MUST have unit stride on that axis for the
SIMD fast path. Inputs omitting the axis are invariant broadcasts. A correct
general path MUST remain available for supported but incompatible mappings.

### `md_reduce` execution

#### Reduction ordering

For every reduction axis, the implementation SHOULD compute the minimum known
inner stride among inputs carrying it. Lower-stride reduction axes SHOULD be
innermost. Ties SHOULD preserve the order supplied to
`md_reduce<Axes...>`.

The same order SHOULD apply across reduction partitions and within each
partition.

#### General scalar executor

For each output coordinate, the scalar executor MUST:

1. initialize from the identity or current output value;
2. traverse all reduction partitions and local reduction coordinates;
3. bind or advance projected inputs;
4. invoke the updater; and
5. store the result once.

The scalar executor is the semantic reference path for all valid reductions
over supported mappings.

#### Register-fragment executor

The fragment path SHOULD be selected when output and inputs are arithmetic and
every input carrying the output SIMD axis is unit-stride on that axis. Inputs
omitting the axis are invariant broadcasts.

The initial portable fragment shape is:

```text
MR = 4 rows for output rank >= 2, otherwise 1
NR = 2 * native SIMD lane count
```

For FP64 on AVX2, `NR` is eight and the fragment is 4-by-8.

For every fragment, the executor MUST:

1. initialize the complete MR×NR accumulator array;
2. retain it across every reduction partition and element;
3. invoke the fully inlined updater for each input tuple; and
4. store the fragment after the complete reduction.

Full fragments SHOULD use compile-time row expansion and constant-bound SIMD
column loops. Short register-block tails MUST work without padded storage.

### `md_stencil` execution

The stencil implementation MUST separate the mathematical neighborhood from
the storage-specific execution strategy.

For a five-point stencil:

```text
out[i,j] = in[i-1,j]
         + in[i,j-1]
         + in[i,j]
         + in[i,j+1]
         + in[i+1,j]
```

the offset footprint is fixed in the index space regardless of mapping.

#### Standard storage

The executor SHOULD partition the output space into logical cache blocks and
bind input origins and strides. A row-major implementation lowers to affine row
accesses:

```text
north  = base + (i-1)*stride_i + j
west   = base + i*stride_i + (j-1)
center = base + i*stride_i + j
east   = base + i*stride_i + (j+1)
south  = base + (i+1)*stride_i + j
```

The contiguous `J` loop SHOULD be vectorized.

#### Native tiled storage

For each output tile, the executor SHOULD:

1. derive the set of input tiles touched by all offsets;
2. bind the center and required neighbor tiles once;
3. partition the output tile into regions where every offset has a fixed source
   tile; and
4. execute each region with affine inner-tile accesses.

For a radius-one cross, the useful shape is:

```text
                 north edge
west edge |      interior      | east edge
                 south edge
```

The interior hot loop MUST NOT perform a global tiled mapping or per-element
source-tile selection. Edge regions SHOULD use statically selected neighboring
tile handles. Physical tile boundaries MUST NOT alter mathematical boundary
semantics.

#### Boundary and aliasing semantics

The initial implementation SHOULD support interior-only output:

```cpp
iai::interior_out<I, J, Radius>(output)
```

Stencil input and output SHOULD be required not to overlap initially, providing
Jacobi-style semantics and independent output-tile parallelism.

In-place Gauss-Seidel semantics require a separately specified ordered
algorithm and are out of scope.

### Code-generation requirements

After specialization and inlining, optimized kernels MUST contain:

- no runtime expression traversal;
- no dynamic allocation;
- no virtual dispatch or type erasure;
- no indirect callback;
- no runtime axis search;
- no hidden packing or copying;
- no full native tiled mapping in inner element loops; and
- no synchronization inside an independently owned output partition.

Compile-time axis recursion SHOULD instantiate fixed-depth loop nests rather
than interpret a generic N-D loop descriptor. `if constexpr` SHOULD eliminate
irrelevant mapping cases, initialization modes, and executor branches.

The reduction fragment path SHOULD expose:

- several independent accumulator chains;
- unit-stride vector loads and stores;
- output loads/stores outside the reduction loop;
- statically recognized broadcasts; and
- constant fragment dimensions matched to the target vector width.

### Host backend boundary

The first backend SHOULD support serial and OpenMP host execution.

Host-specific code owns:

- output work flattening and coordinate decoding;
- OpenMP scheduling;
- target-aware SIMD lane selection;
- scalar, map, reduction-fragment, and stencil executors; and
- optional runtime loop versioning for `layout_stride`.

Bindings, validation, index-space construction, access relations, and mapping
classification SHOULD remain backend-independent.

### Numerical and storage contracts

The initial implementation SHOULD require:

- exactly one output;
- at least one input;
- compatible logical extents for every shared axis;
- only supported standard or known tiled mappings;
- debug validation of incompatible dynamic extents; and
- no overlap except where an algorithm explicitly defines it.

Native tiled operands retain their storage-level divisible-extent requirement.
Logical cache blocks over standard layouts support shortened fringes.

`md_reduce` MAY reassociate updates according to mapping-derived traversal,
fragment shape, compiler target, and thread count. It does not guarantee
bitwise reproducibility.

## Drawbacks

### Limited mapping support

Rejecting other conforming custom mappings narrows generality. This is
intentional: supporting an `mdspan` mapping semantically does not imply that the
library can derive a robust high-performance schedule for it.

### No transparent packing

Some GEMM layouts will not feed the vector microkernel as efficiently as packed
panels. The library intentionally preserves user-provided storage and therefore
may trail BLAS implementations that own packing formats and temporary buffers.

### More algorithm-specific implementation code

Separate map, reduction, and stencil executors duplicate some loop mechanics.
The alternative—a universal iterator—would obscure semantics needed for
optimization and correctness. Shared analysis and scheduling primitives should
limit duplication.

### Compile-time and diagnostic complexity

Typed axes, mapping classification, access relations, and specialized executors
increase template instantiations. Constraints and diagnostics must be designed
carefully so invalid expressions fail near the public call.

### Hybrid storage scheduling

Calls mixing standard and tiled operands require one execution partition that
respects different storage geometries. The initial equal-native-tile-extent
restriction simplifies this but does not solve every useful combination.

## Rationale and alternatives

### Why no public storage provider

A public provider would introduce a second data-access abstraction alongside
`mdspan`, leak blocking mechanics into signatures, and make storage adaptation
part of the user contract. The mapping already describes linear storage. IAI
only needs internal compile-time lowering for the explicitly supported mapping
types.

### Why no packing

Packing introduces temporary ownership, allocation, copy amortization,
backend-specific formats, and hidden costs. Those concerns exceed a
loop-execution library. Users who want physically blocked storage can select a
native tiled layout explicitly.

### Why not a foundational `md_iter`

Map, reduction, and stencil operations expose different initialization,
dependency, reassociation, and register-lifetime semantics. A universal
callback would hide information needed by the scheduler.

The implementation instead shares:

```text
axis analysis
index-space construction
mapping analysis
execution partitioning
coordinate projection
output work assignment
backend dispatch
```

and retains:

```text
execute_map_block
execute_reduce_block
execute_stencil_block
```

An expert primitive may be proposed later, but is not required for zero-cost
implementation.

### Why not express every stencil as `md_reduce`

A neighborhood can be modeled as a synthetic reduction axis, which is useful
for commutative neighborhood sums. Many stencils need named neighbors,
different coefficients, or non-reduction logic. A dedicated neighborhood
access description preserves that structure and enables region-aware tile
binding.

### Why separate execution blocks from storage tiles

Standard mappings have no physical tiles, and register fragments are much
smaller than cache blocks. The implementation must distinguish:

| Concept | Purpose |
|---|---|
| Physical storage tile | Element organization in linear storage |
| Logical cache block | Reuse region in the index space |
| Register fragment | Values retained in registers |
| Parallel work item | Output region owned by one worker |

Native tiled execution may align some of these initially, but the concepts are
not interchangeable.

## Prior art and current prototype

The current prototype is located in:

```text
include/layout_tiled/reduce.hpp
include/layout_tiled/detail/reduce.hpp
include/layout_tiled/detail/tiled_view.hpp
tests/layout_tiled/reduce.test.cpp
```

Reusable components include:

- typed axis lists and membership utilities;
- binding role and updater validation;
- problem-rank and shared-extent analysis;
- projected problem-to-operand coordinates;
- reduction-axis stride ordering;
- output outer-layout flatten/decode;
- scalar reduction traversal;
- target-aware MR×NR fragments;
- register-block tails;
- alias assertions; and
- serial/OpenMP host dispatch separation.

The native tiled adaptor views an N-D tiled `mdspan` as a grid of affine tile
`mdspan`s. It performs the outer tiled lookup when binding a tile and uses
static affine inner mappings afterward.

The following prototype components require generalization rather than direct
copying:

- `valid_binding` currently requires a tiled mapping;
- common tile extents currently conflate storage and execution partitions;
- no logical-block counterpart exists for standard layouts;
- fragment compatibility assumes tiled inner layouts;
- divisibility checks do not distinguish native storage from logical blocks;
  and
- output tile scheduling must become output-space partition scheduling.

## Unresolved questions

1. Should `md_map` use a return-value callback or a mutable output reference?
2. Should the primary stencil API use a neighborhood object, shifted bindings,
   or accept both?
3. Which global boundary policies belong in the initial stencil release?
4. How are automatic logical cache-block shapes selected for standard layouts?
5. How should `layout_stride` unit-stride runtime detection and loop versioning
   be exposed internally?
6. Should scalar-output `md_reduce` be part of the first independent release?
7. Should exact pointwise in-place `md_map` be supported initially?
8. How should the optional integration install internal recognition of the
   known tiled mapping types?
9. Which repository owns the integration header and CMake target?
10. What is the compatibility/deprecation path for
    `layout_tiled::reduce`?
11. Should users eventually receive an explicit schedule object, and which
    controls are safe to expose without leaking internal storage machinery?

## Future possibilities

The following may be proposed after the initial design is validated:

- scalar-output reductions;
- additional mathematical boundary policies;
- an explicit ordered/non-reassociating reduction mode;
- an expert `md_for_each` or `md_kernel` API;
- schedule controls for cache-block shape and parallel axes;
- device backends and device-local fragments;
- split reductions with partial-result combination;
- multiple outputs;
- fusion across compatible map/stencil calls;
- native tile aggregation or subdivision; and
- support for additional mappings through explicit future RFCs.

These extensions must preserve the public `mdspan` data model and must not
introduce hidden packing or storage ownership.

## Implementation plan

### Phase 1: extract tiled reduction semantics

1. Create the new namespace, headers, and interface target.
2. Extract axis lists, bindings, validation, projection, and updater invocation.
3. Add internal recognition of the known tiled mappings.
4. Port `md_reduce` with behavior equivalent to the current implementation.
5. Add a compatibility forwarding entry point if required.
6. Move or duplicate the existing reduction tests.

Acceptance criteria:

- all current reduction tests pass;
- GEMM, GEMV, and tensor contractions remain correct;
- optimized assembly retains vector FMA;
- fragment stores remain outside the reduction loop; and
- tiled benchmark performance does not regress materially.

### Phase 2: standard-layout reductions

1. Recognize standard mappings explicitly.
2. Implement internal affine logical-block access without copying data.
3. Separate cache-block shapes from physical tile extents.
4. Reuse scalar and fragment executors over affine blocks.
5. Support shortened standard-layout fringes.
6. Add mixed standard/tiled operand coverage.

Acceptance criteria:

- standard left/right GEMM and GEMV are correct;
- logical cache blocking appears in generated loops;
- SIMD is selected only for unit-stride compatible operands;
- no allocation or packing appears in implementation or profiles; and
- mixed supported mappings are correct.

### Phase 3: `md_map`

1. Resolve callback and output-binding semantics.
2. Generalize axis validation for zero reduction axes.
3. Implement scalar and SIMD map executors.
4. Add unary, binary, broadcast, heterogeneous, and rank-N tests.
5. Add standard, tiled, and mixed mapping coverage.

Acceptance criteria:

- no reduction machinery remains in map hot loops;
- compatible maps vectorize on the output stride-one axis; and
- output work partitions remain race-free.

### Phase 4: stencil semantics

1. Implement compile-time affine offset descriptions.
2. Resolve the public neighborhood/shift syntax.
3. Implement interior output spaces and the initial boundary policy.
4. Implement a mapping-independent scalar stencil executor.
5. Add five-point, nine-point, higher-radius, and rank-3 tests.

Acceptance criteria:

- results match direct indexed references;
- global boundary behavior is explicit and tested; and
- overlapping stencil input/output is rejected initially.

### Phase 5: optimized stencils

1. Implement standard-layout cache-blocked SIMD traversal.
2. Derive native tile-neighborhood footprints from offsets.
3. Bind neighbor tiles once per output tile.
4. Partition native output tiles into interior and edge regions.
5. Specialize regions so source-tile selection is outside SIMD loops.
6. Replace the local implementation in `benchmarks/stencil_2d.cpp` for
   comparison.

Acceptance criteria:

- standard-layout stencil loops contain contiguous SIMD accesses;
- native-tiled interior loops contain no global tiled mapping;
- tile edges are correct for every supported footprint;
- no halo packing or hidden allocation occurs; and
- performance is compared with naive, cache-blocked, and handwritten
  data-tiled baselines.

## Test plan

### Compile-time rejection

- duplicate or non-dense axes;
- binding rank mismatch;
- reduction axes present in output;
- missing reduction axes;
- incompatible static logical extents;
- incompatible native tile extents;
- invalid callable signatures;
- unsupported custom mappings; and
- invalid stencil offset rank.

### Runtime validation

- incompatible dynamic extents;
- non-divisible native tiled extents;
- shortened standard-layout fringes;
- forbidden aliasing;
- boundary policies; and
- `layout_stride` fast-path guards.

### Correctness kernels

- unary and broadcast maps;
- heterogeneous multi-input maps;
- GEMV and GEMM with `out` and `inout`;
- batched contraction;
- multiple reduction axes;
- standard left and right mappings;
- every supported native tiled mapping;
- mixed standard/tiled operands;
- five- and nine-point 2-D stencils;
- rank-3 stencils; and
- scalar and optimized executor paths.

### Code-generation inspection

- callables inline into hot loops;
- compatible reductions emit vector FMA;
- output fragments remain live across reduction loops;
- fragment loads/stores remain outside reduction loops;
- standard blocks use affine stride/induction arithmetic;
- native tiled inner loops do not invoke global tiled mapping;
- stencil interiors do not select source tiles per element; and
- OpenMP covers only independent output work.

## Implementation invariants

Implementation work under this RFC MUST preserve these invariants:

1. Public operands are `mdspan`s or lightweight typed bindings around them.
2. No public provider, block-view, or storage-policy abstraction is introduced.
3. No algorithm allocates, owns, or packs operand data.
4. Only standard affine mappings and explicitly supported tiled mappings are
   accepted.
5. Mapping classification and local block/tile machinery remain internal.
6. The multidimensional index space is the semantic source of truth.
7. Each operand's `mdspan` mapping is the source of truth for linear storage.
8. Execution blocks and register fragments are distinct from storage tiles.
9. Algorithm semantics remain visible to scheduling and code generation.
10. Every optimized path has a correct general/reference counterpart.
