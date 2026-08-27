# RFC: Multidimensional algorithms over `mdspan`

- **Status:** Draft
- **Authors:** Gabriel dos Santos
- **Target version:** Initial extraction from `layout_tiled::reduce`
- **Tracking issue:** None
- **Implementation:** Started in @include/iai/bindings.hpp, @include/iai/md_reduce.hpp, @include/iai/detail/projected_view.hpp, @include/iai/detail/md_reduce.hpp, @include/iai/detail/access_transform.hpp, @include/iai/detail/layout_tiled_mapping.hpp, @include/iai/detail/native_tiled_view.hpp, @include/iai/detail/native_tiled_reduce.hpp, @include/iai/detail/native_tiled_fragment_reduce.hpp, @include/iai/detail/native_tiled_cache_block.hpp, @include/iai/detail/native_tiled_blocked_fragment_reduce.hpp, and @include/iai/detail/native_tiled_reduce_policy.hpp

## Summary

This RFC proposes a header-only library of multidimensional loop-execution algorithms over `mdspan` operands:

```cpp
iai::md_map(...);
iai::md_reduce<ReductionAxes...>(...);
iai::md_stencil(...);
```

Users describe computation in a **multidimensional index space** using named axes, operand projections, broadcasts, reductions, and neighborhood offsets.
Each operand's `mdspan` mapping describes how coordinates in that index space map into a **linear storage space**.
The implementation inspects those mapping types and partitions the multidimensional index space into cache- and SIMD-friendly execution regions.

`mdspan` is the only public data abstraction.
The initial implementation supports exactly two mapping classes:

1. the standard affine `mdspan` layouts; and
2. the known-native tiled mappings from `layout_tiled`.

Storage classification, logical cache blocks, and native tile handles are internal implementation details.
The library does not expose a block-provider API, optimize arbitrary custom mappings, pack data, allocate operand storage, or silently rearrange layouts.

The original `layout_tiled::reduce` implementation was the prototype for typed axis analysis, native-tile access, output scheduling, scalar execution, and the register-fragment reduction kernel.
Those execution facilities are now being extracted into Iai; `layout_tiled` owns the native tiled mappings, while Iai owns the algorithms that inspect and execute over them.

## Motivation

The original `layout_tiled::reduce` prototype demonstrated that a high-level axis expression can produce code comparable to a handwritten blocked kernel.
The extracted Iai API expresses GEMM as:

```cpp
enum Axes { I, J, K };

iai::md_reduce<K>(
  iai::out<I, J>(C, 0.0),
  iai::in<I, K>(A),
  iai::in<K, J>(B),
  [](auto& result, auto lhs, auto rhs) { result += lhs * rhs; }
);
```

This expression contains the mathematical relationships but does not prescribe loop order, cache blocking, vector direction, thread placement, or register fragment shape.
The compiler sees concrete axis lists, mappings, tile extents, value types, accessors, and the updater type, allowing the implementation to derive and specialize the schedule.

The initial optimized lowering remains coupled to native tiled mappings, but the execution API and implementation are owned by Iai.
The abstraction is broader than those mappings:

- a pointwise or broadcast map requires the same typed index-space projection;
- a reduction over ordinary row-major or col-major ordered data can use logical cache blocking;
- a stencil can use affine neighborhood access and layout-aware tile binding;
- mixed standard and tiled operands can participate in one index space; and
- backend-independent analysis can be reused by serial, OpenMP, and later device executors.

A separate library also clarifies responsibility.
Layout mappings describe storage.
Multidimensional algorithms describe computation and execution.

### Locality model

High performance requires spatial and temporal locality to be designed together.
An operand mapping creates **spatial locality** by determining which logical elements are colocated in main memory, pages, cache lines, physical tiles, etc., and which coordinates admit unit-stride access.
An execution schedule creates **temporal locality** by determining which resident elements are reused before eviction.
The schedule also realizes the spatial locality made available by a mapping by traversing colocated elements together.

Cache or loop blocking changes when existing storage is revisited; it does not change which elements are physically colocated.
Packing changes that physical organization, which is why optimized BLAS implementations combine cache blocking with packed panels.
Iai does not pack implicitly.
Instead, a native tiled mapping provides an ahead-of-time physical organization, and Iai derives cache-block, physical-tile, and register-fragment traversal that reuses that organization directly.

The design principle is therefore:

```text
operand mappings provide spatial locality
              +
axis-derived execution provides temporal locality
              =
layout-aware multidimensional execution
```

Neither component is sufficient by itself.
A tiled mapping used with a poor loop schedule may repeatedly evict useful tiles, while a cache-blocked schedule over an incompatible mapping may still waste cache lines or require strided scalar access.

## Terminology and normative language

The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are to be interpreted as normative design requirements.

### Multidimensional index space

The multidimensional index space is the semantic coordinate system in which an algorithm is expressed.
It contains:

- logical axes and extents;
- free and reduction axes;
- operand projections and broadcasts;
- stencil offsets and neighborhoods; and
- coordinates at which outputs are produced.

For GEMM, `(I,J,K)` is the complete index space, `(I,J)` is the output index space, and `K` is the reduction index space.

### Linear storage space

The linear storage space is the one-dimensional address space consumed by hardware.
An operand's `mdspan` mapping converts its multidimensional coordinates into offsets in that space.

Relevant storage properties include:

- strides and stride-one dimensions;
- standard versus natively tiled organization;
- physical tile extents;
- tile inner and outer order; and
- accessor and data-handle semantics.

### Execution partition

An execution partition is a region of the multidimensional index space assigned to a loop nest, worker, logical cache block, physical tile, or register fragment.
It does not imply copied or newly owned storage.

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

Documentation and implementation comments SHOULD use “iteration space,” “output index space,” “reduction index space,” “physical tile space,” and “linear storage space.”
They SHOULD avoid using “domain” for these concepts.

## Guide-level explanation

### Bind operands to named axes

Each binding associates operand dimensions with axes in the complete index space:

```cpp
enum Axes { I, J, K };

auto a = iai::in<I, K>(A);
auto b = iai::in<K, J>(B);
auto c = iai::out<I, J>(C, 0.0);
```

Axis order maps directly to `mdspan` dimension order.
An operand that omits an axis is invariant along that axis and broadcasts naturally.
For example, `in<J>(bias)` broadcasts one bias value across every `I` coordinate.

Bindings are lightweight wrappers around the supplied `mdspan`.
They do not own, allocate, pack, or erase the type of operand storage.
They are lazy projected operand views: constructing one records an operand role and a dimension-to-axis projection but does not access any element or select an execution partition.

The short factories remain named `in`, `out`, and `inout`.
This follows the standard-library convention in which a short adaptor name may construct a type whose name ends in `_view`.
The concrete binding types SHOULD be named `projected_input_view`, `projected_output_view`, and `projected_inout_view`.

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

The implementation chooses a stride-one output dimension for SIMD, traverses other dimensions in mapping-preferred order, and binds projected or broadcast inputs outside the innermost work where possible.

The exact output/callback convention in this example remains unresolved.

### Reduce axes from an index space

`md_reduce` removes one or more axes and accumulates into each output coordinate:

```cpp
iai::md_reduce<K>(
  iai::out<I, J>(C, 0.0),
  iai::in<I, K>(A),
  iai::in<K, J>(B),
  [](auto& result, auto lhs, auto rhs) { result += lhs * rhs; }
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

The implementation owns complete output execution partitions.
Every logical accumulator belongs to one worker and survives the complete reduction.
For a small reduction, its physical representation may remain in registers until completion.
For a cache-blocked reduction, a partial accumulator may be stored in the exclusively owned output block between reduction blocks and reloaded into a register fragment for the next block.

`md_map`, `md_reduce`, and `md_stencil` are eager terminal operations.
Only their operand views and internal partition enumeration are lazy.
Any future deferred operation or fusion facility MUST use an explicit construction-and-execution interface rather than silently changing these calls to deferred execution.

### Express a stencil neighborhood

A stencil maps an output coordinate from affine offsets around one or more inputs.
A proposed five-point API is:

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

An equivalent shifted-binding expression could lower into the same internal access description:

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

The neighborhood footprint is independent of storage layout.
For native tiled storage, the executor binds the center and neighboring physical tiles and separates tile interiors from edges.
For standard affine storage, it operates over logical cache blocks and ordinary strides.

### Use standard or tiled `mdspan`s

The public expression is unchanged by storage choice:

```cpp
std::mdspan<double, matrix_extents, std::layout_right> standard_A;
std::mdspan<double, matrix_extents,
            layout_tiled::layout_rtr<16, 512>> tiled_A;
```

For a standard mapping, Iai creates logical execution blocks over the existing strides.
For a known tiled mapping, it aligns execution with physical tiles and uses affine inner-tile views.
No storage is packed or copied in either case.

A call MAY mix standard and tiled operands.
The shared multidimensional index space, rather than any one operand, remains the semantic source of truth.

## Reference-level explanation

### Public algorithm set

The initial public API MUST contain:

```cpp
iai::md_map(...);
iai::md_reduce<ReductionAxes...>(...);
```

The initial API SHOULD contain `md_stencil`, depending on the amount of work required.

The initial API MUST NOT contain `md_scan`.

The initial API SHOULD NOT expose `md_iter`. A future expert-oriented primitive may be proposed separately after the semantic and scheduling model is stable.

### Public data model

Every operand MUST be an `mdspan` or a lightweight binding that directly owns an `mdspan` value.

The public API MUST NOT expose:

- block or tile providers;
- packed-panel objects;
- internal execution blocks;
- storage ownership abstractions;
- mapping-classification tags; or
- backend-specific fragments.

The implementation MUST preserve source accessor and data-handle semantics.

### Binding model

`in<Axes...>(view)` marks a read-only operand and records its dimension-to-axis projection.

For reductions, `out<Axes...>(view, identity)` initializes each result from an explicit identity, and `inout<Axes...>(view)` initializes it from the current output value.

Bindings MUST retain concrete view, axes, identity, and callable types through specialization.
They MUST NOT allocate, use virtual dispatch, or construct a runtime expression graph.
Constructing a binding MUST NOT read or write operand elements, bind a physical tile, or enumerate execution work.

The execution algorithms consume these lazy bindings eagerly.
Returning from `md_map`, `md_reduce`, or `md_stencil` means that the operation has completed according to the selected backend's synchronization contract.
A future explicit execution expression MAY be introduced for fusion or backend selection, for example:

```cpp
auto operation = iai::make_md_reduce<K>(...);
iai::execute(operation, policy);
```

Such a facility is distinct from a range view and is not part of the initial API.

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
- `std::layout_stride` when runtime strides satisfy the selected execution path.

An optional integration component MUST support the known `layout_tiled` mappings without making the core standard-layout implementation depend on `layout_tiled`.

Other custom mappings MUST be rejected at compile time with a useful diagnostic.
The library MUST NOT provide a generic mapping fallback merely because a type satisfies the broad `mdspan` mapping requirements.

Mapping classification MUST be derived internally from each `mdspan::mapping_type`.
The mechanism MUST NOT appear in public signatures.

### Dependency structure

The intended dependency direction is:

```text
iai
  -> mdspan
  -> standard-layout execution
  -> tiled-layout execution

layout_tiled
  -> mdspan
  -> tiled mappings
```

A possible integration include is:

```cpp
#include <iai/layout_tiled.hpp>
```

The call site still passes only `mdspan` bindings.

### Standard affine lowering

For standard layouts, the implementation MUST partition the multidimensional index space into logical cache blocks over the original storage.
A local block retains the source handle, accessor, origin, extents, and strides.
It MUST NOT own or copy elements.

The block need not be contiguous as a whole.
A unit-stride inner dimension is sufficient for the corresponding SIMD fast path.

`layout_stride` SHOULD receive a SIMD path only when the selected SIMD-axis stride is one.
The implementation MAY use a small runtime dispatch or loop version and MUST otherwise use a correct general path.

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

the outer tiled mapping handles the first term at tile-binding time.
Hot local loops use only the affine inner mapping.

Native tiled storage retains its divisible-extent requirement.

#### Hierarchical tile decomposition and slicing

Native tiled mappings SHOULD be lowered internally as a hierarchical mapping with separate physical-tile and local-coordinate modes.
For every logical axis:

```text
x[a] = tile_coordinate[a] * tile_extent[a] + local_coordinate[a]
```

and the storage offset has the form:

```text
offset(tile, local)
  = tile_size * outer_layout(tile) + inner_layout(local)
```

This representation exposes a lazy view with conceptual coordinates `(local, tile)` without changing or copying storage.
Selecting a physical-tile coordinate is partial evaluation of that view:

1. evaluate the fixed outer coordinates once;
2. accumulate the resulting offset into the source accessor and handle; and
3. retain the unevaluated inner modes as a static affine local mapping.

Axis binding and partitioning MAY be modeled internally as mapping composition.
If `P` projects complete problem coordinates onto one operand, `L` is the operand storage mapping, and `E` maps local execution coordinates into the problem index space, the effective access is:

```text
L ∘ P ∘ E
```

Fixed output-block, reduction-block, physical-tile, and fragment coordinates SHOULD be partially evaluated at the highest loop level where they become invariant.
The resulting handle offset and residual mapping form a bound operand view.
Broadcast axes become omitted or stride-zero internal modes and MUST NOT require a runtime axis search.

The initial implementation SHOULD provide only the internal layout operations required for this lowering:

- named-axis projection;
- hierarchical native-tile decomposition;
- composition with rectangular execution tilers;
- partial evaluation or slicing;
- removal of extent-one and broadcast modes; and
- coalescing of adjacent provably contiguous modes.

It SHOULD NOT implement or expose a general-purpose layout algebra in the first release.
In particular, general layout products, complements, and inverses are unnecessary for binding a known native tile.
A limited common-sublayout or inverse analysis MAY be added later to derive SIMD widths shared by several operands.

A cache block containing multiple native physical tiles is generally not one affine `mdspan` because tile boundaries introduce discontinuities.
It MUST remain a hierarchical range of native tile views.
The range SHOULD bind its physical-tile origin once, retain the source accessor and handle, and expose affine outer-grid strides for the residual tile coordinates.
Execution MAY then obtain static affine physical-tile views from that range without repeating named-axis projection or a full tiled-layout lookup.
Such a range is an execution-local descriptor, not an owning container; constructing it performs no allocation or element access.
The hot local executor binds one affine physical tile at a time.

### Mixed mappings

A call MAY combine supported standard and tiled operands.
Execution partitions belong to the shared index space.
Native tile boundaries constrain axes carried by native operands; standard operands cover the same coordinates using affine strides.

The initial implementation SHOULD require native tiled operands sharing an axis to have identical physical tile extents on that axis.
Standard operands do not participate in this physical compatibility requirement.

Reconciling mismatched native tile extents is deferred.

### Shared execution hierarchy

All algorithms SHOULD lower through this hierarchy:

```text
public expression
  -> compile-time axis/access validation
  -> multidimensional index-space construction
  -> compile-time mapping classification
  -> projected access composition and simplification
  -> output and reduction cache-block partitioning
  -> flattened output work items
  -> algorithm-specific block executor
  -> physical-tile or affine-block slicing
  -> register fragments
  -> fully inlined callable
```

The implementation SHOULD share axis analysis, index-space construction, mapping analysis, coordinate projection, partial evaluation, cache-block construction, output work assignment, and backend dispatch.
It MUST retain separate map, reduction, and stencil block executors.

### Output work assignment

The host backend SHOULD flatten only independent output execution partitions.
It SHOULD decode N-D output coordinates in the output mapping's preferred order.

When OpenMP is enabled, the initial schedule is:

```cpp
#pragma omp parallel for schedule(static)
```

Workers MUST own disjoint output partitions. Reduction axes MUST NOT be split between workers in the initial implementation.

Without OpenMP, the same work-item order MUST execute serially.

### SIMD and register-unrolled axes

For `layout_right`, the final, or rightmost, output dimension is the preferred stride-one axis.
For `layout_left`, the first, or leftmost, output dimension is preferred.

The preferred stride-one output axis SHOULD become the SIMD axis.
An adjacent output axis SHOULD become the register-unrolled axis.
Remaining output axes identify SIMD groups or register-fragment planes.

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

Every input carrying the SIMD axis MUST have unit stride on that axis for the SIMD fast path.
Inputs omitting the axis are invariant broadcasts.
A correct general path MUST remain available for supported but incompatible mappings.

### `md_reduce` execution

#### Reduction ordering

For every reduction axis, the implementation SHOULD compute the minimum known inner stride among inputs carrying it.
Lower-stride reduction axes SHOULD be innermost.
Ties SHOULD preserve the order supplied to `md_reduce<Axes...>`.

The same order SHOULD apply across reduction partitions and within each partition.
Stride alone is not a complete scheduling model.
The executor SHOULD also inspect operand-axis incidence: an operand omitting an axis is invariant while that axis changes and is therefore a candidate for reuse at that loop level.
Loop interchange and cache-block shapes SHOULD maximize reuse of invariant operand regions while preserving a compatible SIMD direction.

#### General scalar executor

For each output coordinate, the scalar executor MUST:

1. initialize from the identity or current output value;
2. traverse all reduction partitions and local reduction coordinates;
3. bind or advance projected inputs;
4. invoke the updater; and
5. store the result once.

The scalar executor is the semantic reference path for all valid reductions over supported mappings.

#### Cache-blocked reduction schedule

The optimized executor MUST distinguish a physical storage tile from a cache block.
A native cache block is a rectangular collection of one or more complete physical tiles; a standard-layout cache block is a logical affine region.
Neither form owns or copies operand elements.

For native tiled storage, the minimum cache-blocked schedule is:

```text
owned output physical tile
  initialize output tile
  for each native reduction tile
    bind projected input tiles once
    for each output register fragment
      load partial accumulator
      traverse local reduction coordinates
      store partial accumulator
```

This schedule consumes a bound input-tile set across the complete output tile before advancing the reduction-tile coordinate.
For three FP64 32-by-32 tiles, the nominal input-plus-output working set is 24 KiB, so the schedule can retain the spatial locality of the native tiles while creating temporal reuse in a typical private cache.

The executor SHOULD generalize this schedule to cache blocks measured in physical-tile coordinates:

```text
BI = output-block extent along one free axis, in physical tiles
BJ = output-block extent along another free axis, in physical tiles
BK = reduction-block extent, in physical tiles
```

The footprint estimate MUST count the distinct projected operand regions, rather than assume a particular kernel.
For bindings `A(I,K)`, `B(K,J)`, and `C(I,J)`, this naturally gives:

```text
A panel: BI * BK physical tiles
B panel: BK * BJ physical tiles
C block: BI * BJ physical tiles
```

The same calculation applies to higher-rank contractions and broadcasts through their axis projections.
The initial policy MAY use conservative target-specific cache budgets and fixed candidate block shapes.
Future schedule controls or autotuning are not required for correctness.

An output worker MUST own every partial accumulator in its output cache block.
Reduction blocks MUST remain ordered for each output coordinate and MUST NOT be split between workers in the initial implementation.
For `out`, the output block is initialized once before the first reduction block; for `inout`, existing values supply the initial state.
A partial value MAY be written to and reloaded from the owned output block between reduction blocks.

The implementation SHOULD retain a complete-reduction fragment schedule for cases where the projected full-reduction working set is expected to fit private cache.
It SHOULD select the cache-blocked schedule otherwise.
Schedule selection MUST use mapping, extents, operand projections, value sizes, and backend policy; it MUST NOT recognize GEMM or another kernel by name.

#### Register-fragment executor

The fragment path SHOULD be selected when output and inputs are arithmetic and every input carrying the output SIMD axis is unit-stride on that axis.
Inputs omitting the axis are invariant broadcasts.

The initial portable fragment shape is:

```text
MR = 4 rows for output rank >= 2, otherwise 1
NR = 2 * native SIMD lane count
```

For FP64 on AVX2, `NR` is eight and the fragment is 4-by-8.

Within one selected reduction block, the executor MUST:

1. initialize the complete MR×NR accumulator array;
2. retain it across every local reduction partition and element in that block;
3. invoke the fully inlined updater for each input tuple; and
4. store the fragment after the reduction block.

The complete-reduction schedule treats the full reduction as one reduction block and therefore stores once.
The cache-blocked schedule reloads and stores partial fragments at reduction-block boundaries.
Those accesses SHOULD hit in the owned output cache block and are preferred over repeatedly fetching projected input regions that exceed private-cache capacity.

Full fragments SHOULD use compile-time row expansion and constant-bound SIMD column loops.
Short register-block tails MUST work without padded storage.
Fragment shape SHOULD be a backend policy informed by SIMD width, value and accumulator types, input count, and register pressure.
The initial 4-by-two-vectors shape remains a portable default rather than a semantic property of `md_reduce`.

### `md_stencil` execution

The stencil implementation MUST separate the mathematical neighborhood from the storage-specific execution strategy.

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

The executor SHOULD partition the output space into logical cache blocks and bind input origins and strides.
A row-major implementation lowers to affine row accesses:

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
3. partition the output tile into regions where every offset has a fixed source tile; and
4. execute each region with affine inner-tile accesses.

For a radius-one cross, the useful shape is:

```text
                 north edge
west edge |      interior      | east edge
                 south edge
```

The interior hot loop MUST NOT perform a global tiled mapping or per-element source-tile selection.
Edge regions SHOULD use statically selected neighboring tile handles.
Physical tile boundaries MUST NOT alter mathematical boundary semantics.

#### Boundary and aliasing semantics

The initial implementation SHOULD support interior-only output:

```cpp
iai::interior_out<I, J, Radius>(output)
```

Stencil input and output SHOULD be required not to overlap initially, providing Jacobi-style semantics and independent output-tile parallelism.

In-place Gauss-Seidel semantics require a separately specified ordered algorithm and are out of scope.

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

Compile-time axis recursion SHOULD instantiate fixed-depth loop nests rather than interpret a generic N-D loop descriptor.
`if constexpr` SHOULD eliminate irrelevant mapping cases, initialization modes, and executor branches.

The reduction fragment path SHOULD expose:

- several independent accumulator chains;
- unit-stride vector loads and stores;
- output loads/stores outside the innermost local reduction loop;
- input tile or block binding outside the fragment loops when the selected schedule permits reuse;
- statically recognized broadcasts; and
- constant fragment dimensions matched to the target vector width.

### Host backend boundary

The first backend SHOULD support serial and OpenMP host execution.

Host-specific code owns:

- output work flattening and coordinate decoding;
- OpenMP scheduling;
- conservative cache-budget defaults and optimized-schedule selection;
- target-aware SIMD lane selection;
- scalar, map, reduction-fragment, and stencil executors; and
- optional runtime loop versioning for `layout_stride`.

Bindings, validation, index-space construction, access relations, and mapping classification SHOULD remain backend-independent.

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

`md_reduce` MAY reassociate updates according to mapping-derived traversal, fragment shape, compiler target, and thread count.
It does not guarantee bitwise reproducibility.
An updater used with a reorderable optimized executor MUST have no externally observable side effects other than modification of its accumulator argument.
An explicit ordered or non-reassociating mode may relax this requirement in a future release.

## Drawbacks

### Limited mapping support

Rejecting other conforming custom mappings narrows generality.
This is intentional: supporting an `mdspan` mapping semantically does not imply that the library can derive a robust high-performance schedule for it.

### No transparent packing

Some GEMM layouts will not feed the vector microkernel as efficiently as packed panels.
The library intentionally preserves user-provided storage and therefore may trail BLAS implementations that own packing formats and temporary buffers.

### More algorithm-specific implementation code

Separate map, reduction, and stencil executors duplicate some loop mechanics.
The alternative—a universal iterator—would obscure semantics needed for optimization and correctness.
Shared analysis and scheduling primitives should limit duplication.

### Compile-time and diagnostic complexity

Typed axes, mapping classification, access relations, and specialized executors increase template instantiations.
Constraints and diagnostics must be designed carefully so invalid expressions fail near the public call.

### Hybrid storage scheduling

Calls mixing standard and tiled operands require one execution partition that respects different storage geometries.
The initial equal-native-tile-extent restriction simplifies this but does not solve every useful combination.

## Rationale and alternatives

### Why no public storage provider

A public provider would introduce a second data-access abstraction alongside `mdspan`, leak blocking mechanics into signatures, and make storage adaptation part of the user contract.
The mapping already describes linear storage.
Iai only needs internal compile-time lowering for the explicitly supported mapping types.

### Why no packing

Packing introduces temporary ownership, allocation, copy amortization, backend-specific formats, and hidden costs.
Those concerns exceed a loop-execution library.
Users who want physically blocked storage can select a native tiled layout explicitly.

Native tiled storage is not sufficient by itself.
Iai must still choose a cache-blocked execution order that reuses native tiles before eviction.
Conversely, standard logical cache blocking cannot manufacture the spatial locality of a physically suitable native tile or packed panel.

### Why bindings are lazy but algorithms are eager

`in`, `out`, and `inout` describe operand roles and coordinate projections and therefore compose naturally as lightweight lazy views.
They do not have enough information to select a physical tile or execution block until an algorithm constructs a schedule.

`md_map`, `md_reduce`, and `md_stencil` mutate an explicit output, own parallel scheduling, and perform extent and alias validation.
Deferring those effects implicitly would make mutation timing and operand lifetimes surprising.
They are therefore eager terminal operations.
An explicit operation object may be proposed separately if fusion or deferred backend selection justifies it.

### Why not a foundational `md_iter`

Map, reduction, and stencil operations expose different initialization, dependency, reassociation, and register-lifetime semantics.
A universal callback would hide information needed by the scheduler.

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

An expert primitive may be proposed later, but is not required for zero-cost implementation.

### Why not express every stencil as `md_reduce`

A neighborhood can be modeled as a synthetic reduction axis, which is useful for commutative neighborhood sums.
Many stencils need named neighbors, different coefficients, or non-reduction logic.
A dedicated neighborhood access description preserves that structure and enables region-aware tile binding.

### Why separate execution blocks from storage tiles

Standard mappings have no physical tiles, and register fragments are much smaller than cache blocks.
The implementation must distinguish:

| Concept | Purpose |
|---|---|
| Physical storage tile | Element organization in linear storage |
| Logical cache block | Reuse region in the index space |
| Register fragment | Values retained in registers |
| Parallel work item | Output region owned by one worker |

Native tiled execution may align some of these initially, but the concepts are not interchangeable.

## Prior art and current prototype

### CuTe layout representation and algebra

NVIDIA's [CuTe Layout Representation and Algebra](https://arxiv.org/abs/2603.02298) is a primary influence on the internal lowering model.
CuTe treats a layout as a coordinate-to-offset function, represents tiled structures hierarchically, and derives tiling and partitioning through composition.
Its slicing model partially evaluates fixed coordinates into the accessor offset and retains the unevaluated layout modes as a new tensor view.

Iai adopts a deliberately restricted form of these ideas:

- named-axis bindings define projection mappings;
- execution partitions define rectangular tilers;
- supported storage mappings compose with those projections and tilers;
- fixed work-item and tile coordinates are partially evaluated into bound handles;
- residual affine mappings drive hot local loops; and
- compatible modes may be coalesced for SIMD analysis.

Iai does not expose a CuTe-style public layout algebra.
`mdspan` remains the public tensor view, and mapping recognition, composition, slicing, and simplification remain implementation details for explicitly supported mappings.

CuTe's divide-then-slice pattern corresponds directly to viewing native tiled storage as `(local tile, tile grid)` and selecting a grid coordinate to produce an affine local tile.
For standard mappings, the same conceptual operation creates a logical cache block rather than discovering a physical tile.

### Native-tiled reduction implementation

The current implementation is located in:

```text
include/iai/bindings.hpp
include/iai/md_reduce.hpp
include/iai/detail/access_transform.hpp
include/iai/detail/md_reduce.hpp
include/iai/detail/native_tiled_blocked_fragment_reduce.hpp
include/iai/detail/native_tiled_cache_block.hpp
include/iai/detail/native_tiled_fragment_reduce.hpp
include/iai/detail/native_tiled_reduce.hpp
include/iai/detail/native_tiled_reduce_policy.hpp
include/iai/detail/native_tiled_view.hpp
tests/layout_tiled/iai.test.cpp
tests/layout_tiled/iai_reduce.test.cpp
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

The native tiled adaptor views an N-D tiled `mdspan` as a grid of affine tile `mdspan`s.
It performs the outer tiled lookup when binding a tile and uses static affine inner mappings afterward.

The complete-reduction fragment executor uses this temporal order:

```text
output physical tile
  output register fragment
    every reduction physical tile
```

For a 32-by-32 output tile and an AVX2 FP64 4-by-8 fragment, the complete reduction is traversed separately for 32 output fragments.
Projected A-like regions are revisited across column fragments, projected B-like regions are revisited across row fragments, and each input tile is rebound for every fragment.
This minimizes output traffic but makes the full-reduction projected input footprint the unit of temporal reuse.

The supplied four-thread GEMM measurements at size 4096 show the resulting cache-capacity failure:

| Counter | Current reduction | Handwritten DT | Ratio |
|---|---:|---:|---:|
| L1 data misses | 742.6 M | 142.1 M | 5.23x |
| LLC references | 336.5 M | 41.6 M | 8.08x |
| LLC misses | 92.8 M | 22.8 M | 4.07x |

At smaller sizes, the complete-reduction fragment path remains competitive or substantially faster because the projected footprint fits private cache and the executor performs fewer instructions than DT/TV.
This crossover motivates retaining that path while adding native-tile reduction blocking for larger footprints.

The following prototype components require generalization rather than direct copying:

- `valid_binding` currently requires a tiled mapping;
- common tile extents currently conflate storage and execution partitions;
- no logical-block counterpart exists for standard layouts;
- fragment compatibility assumes tiled inner layouts;
- divisibility checks do not distinguish native storage from logical blocks;
- output tile scheduling must become output-space partition scheduling;
- the fragment executor traverses the complete reduction separately for every register fragment;
- input tiles are consequently rebound and revisited for every fragment;
- register-fragment lifetime is incorrectly conflated with logical accumulator lifetime; and
- no cache-blocked native-tile schedule reuses an input-tile set across all relevant output fragments.

## Unresolved questions

1. Should the primary stencil API use a neighborhood object, shifted bindings, or accept both?
2. Which global boundary policies belong in the initial stencil release?
3. How are automatic logical cache-block shapes selected for standard layouts?
4. How should `layout_stride` unit-stride runtime detection and loop versioning be exposed internally?
5. Should scalar-output `md_reduce` be part of the first independent release?
6. Should exact pointwise in-place `md_map` be supported initially?
7. How should the optional integration install internal recognition of the known tiled mapping types?
8. Which repository owns the integration header and CMake target?
9. Should users eventually receive an explicit schedule object, and which controls are safe to expose without leaking internal storage machinery?

## Future possibilities

The following may be proposed after the initial design is validated:

- scalar-output reductions;
- additional mathematical boundary policies;
- an explicit ordered/non-reassociating reduction mode;
- an expert `md_for_each` API;
- eventual public schedule controls for cache-block shape and parallel axes;
- device backends and device-local fragments;
- split reductions with partial-result combination;
- multiple outputs;
- fusion across compatible map/stencil calls;
- support for additional mappings through explicit future RFCs.

These extensions must preserve the public `mdspan` data model and must not introduce hidden packing or storage ownership.

Detailed follow-on notes are recorded separately:

- [`iai-gpu-execution.md`](iai-gpu-execution.md) describes a future GPU
  backend, including generic device-callable execution, CuTe/CUTLASS reuse,
  device execution contracts, and a deferred semantic-operation proposal.
- [`iai-std-execution.md`](iai-std-execution.md) describes a future
  `make_md_reduce` sender, scheduler-aware bulk execution, provider adapters,
  cancellation, and constrained fusion.
- [`iai-phase2-handoff.md`](iai-phase2-handoff.md) is the implementation handoff
  for the next native-tiled performance phase.

## Implementation plan

### Phase 1: extract and correct native-tiled reductions

1. Establish a reproducible baseline before extraction.
   Record correctness, time, instructions, L1 data misses, LLC references, and LLC misses for GEMM, GEMV, and representative tensor contractions across sizes below and above private-cache capacity.
   Report counters per logical update or FLOP as well as absolute values.
   Make the GEMM tile-size option instantiate or select the requested mapping and kernel, or remove the ineffective option.
   Use `benchmarks/scripts/run_phase1_baseline.sh` to isolate implementations in fresh four-thread processes, bind them to distinct physical cores, validate the performance/fixed-base-frequency policy, and retain topology and context-switch metadata.
   Nanobench's calling-thread counters are sufficient for relative Phase 1 comparisons because every implementation uses the same pinned four-thread configuration; reports MUST label that scope and MUST NOT describe the counts as whole-team aggregate events.
   **Status: complete.** The baseline and normalized analysis are recorded in `docs/iai-phase1-baseline.md`. The ineffective single GEMM tile-size option was replaced by independent `TI`, `TJ`, and `TK` CMake controls that instantiate the requested mappings; the measured 64x32x64 shape is now the default.
2. Create the `iai` namespace, public headers, and interface target.
   Keep `in`, `out`, and `inout` as the public factories and give their returned implementation types projected-view names.
   Specify that bindings are lazy and `md_reduce` is eager.
   **Status: complete.** The public API is available through `iai/bindings.hpp` and `iai/md_reduce.hpp`, and consumers can link the exported `iai::iai` interface target. Step 12 removed the temporary `layout_tiled::reduce` forwarding facade so `layout_tiled` owns mappings only.
3. Extract mapping-independent axis lists, operand roles, validation, logical-extent merging, alias checking, updater invocation, and problem-to-operand projection.
   Preserve concrete `mdspan`, accessor, handle, identity, and updater types.
   **Status: complete.** These facilities now live under `iai::detail`; projected operand views retain their concrete source `mdspan` directly, and the native-tiled executor applies its tile-specific compatibility and lowering only after semantic validation.
4. Add internal recognition traits for the known `layout_tiled` mappings.
   Do not accept arbitrary mappings merely because they expose similarly named tile members.
   **Status: complete.** `iai::detail::layout_tiled_mapping_traits` explicitly recognizes the four `layout_tiled` layout families and verifies exact correspondence between each layout policy and its nested mapping type; structurally compatible wrappers are rejected.
5. Introduce the restricted internal access transformations needed by the executor:
   named-axis projection, hierarchical `(local tile, tile grid)` decomposition, composition with rectangular execution tilers, partial evaluation of fixed coordinates, and simple coalescing.
   Refactor the existing tiled adaptor to bind a tile by offsetting through the source accessor once and returning the residual static affine inner mapping.
   **Status: complete.** Restricted coordinate transformations now live under `iai::detail`: named-axis projections compose with non-owning rectangular tilers; recognized tiled mappings decompose into local-tile and tile-grid coordinates; and affine access modes support removal of extent-one/broadcast modes and conservative contiguous coalescing. `native_tiled_view` lazily represents the physical tile grid, while `bind_tile` partially evaluates one fixed tile coordinate through exactly one source-accessor offset and returns an `mdspan` preserving the source element, handle, accessor, static tile extents, and native affine inner layout. The native reducer consumes this path directly.
6. Port a native-tiled scalar `md_reduce` as the semantic reference path.
   It must support current ranks, broadcasts, heterogeneous inputs, multiple reduction axes, inner-left and inner-right mappings, `out`, `inout`, and register-block tails where applicable.
   **Status: complete.** Native-tiled call validation, mapping-derived traversal, projected tile binding, output ownership, scalar accumulation, and common host dispatch now live in `iai::detail::native_tiled_reduce.hpp`. Direct Iai tests cover multiple reduction axes, broadcasts, heterogeneous values, mixed inner-left/inner-right mappings, `out`, and `inout`. This path remains the semantic fallback for calls that are not compatible with the optimized fragment executor.
7. Port the current complete-reduction register-fragment executor as the small-working-set optimized path.
   Preserve compile-time row expansion, constant SIMD bounds, fully inlined updater invocation, and one final output store when the full reduction is selected as one block.
   **Status: complete.** `iai::detail::native_tiled_fragment_reduce.hpp` now owns fragment compatibility analysis, target-derived MR×NR fragment sizing, compile-time row expansion, constant SIMD column bounds, complete-reduction traversal, and tail handling. The full fragment keeps each accumulator live across every native reduction tile and performs one final output access and store; direct tests verify compatibility outcomes and the single-store property. The migrated `layout_tiled/detail/reduce.hpp` executor has been removed. Dispatch was compatibility-based through Step 9 and is now owned by the Step 10 working-set policy. Step 12 subsequently removed the temporary public forwarding facade.
8. Add a native-tile cache-blocked fragment executor.
   The first implementation MUST support one output physical tile by one native reduction tile: initialize the output tile once, bind projected input tiles once per reduction tile, visit every output fragment, and use the owned output tile to carry partial accumulators between reduction tiles.
   **Status: complete.** `iai::detail::native_tiled_blocked_fragment_reduce.hpp` implements the unit native cache-block schedule: a worker binds and exclusively owns one output physical tile, initializes `out` elements once (or preserves the existing `inout` state), walks the native reduction-tile coordinates, binds each projected input tile exactly once at that level, and consumes that bound tuple across every full or tail output fragment. Each fragment loads its partial accumulators before the local reduction elements and stores them afterward, so the owned output tile carries state between reduction tiles without traffic inside the innermost reduction loop. A direct deterministic test forces this executor, covers row and column fragment tails plus `out` and `inout`, verifies numerical results, verifies exactly one binding per input per reduction tile despite multiple output fragments, and checks the expected output accesses at initialization and reduction boundaries. Production selection was connected in Step 10; rectangular multi-tile cache-block descriptors were added in Step 9, and Step 11 subsequently hoisted their invariant address work.
9. Generalize the cache-block descriptor to rectangular groups of native physical tiles.
   Derive distinct operand footprints and reuse opportunities from axis projections.
   Candidate `BI`, `BJ`, and `BK` shapes MUST be expressed in tile coordinates and selected from cache-budget and mapping information, not from a recognized kernel name.
   **Status: complete.** `iai::detail::native_tile_block_shape` expresses a rectangular block entirely in physical-tile coordinates, and `native_tile_block` materializes its origin and shortened boundary extents. Axis-restricted enumeration lets the same descriptor traverse output or reduction projections without introducing GEMM-specific roles. Per-binding metrics derive the physical tile bytes, distinct tile count, working-set bytes, and potential reuse factor solely from the binding's named-axis projection. The initial candidate selector rejects shapes over a supplied cache budget, then ranks the survivors by projected reused bytes, tile volume, and footprint. The grouped executor assigns each output-tile rectangle to one worker, initializes each owned output tile once, and walks rectangular reduction-tile groups including tails. Compile-time tests cover representative projected footprints and budget-based selection; runtime tests cover descriptor enumeration, shortened boundaries, output-block tails, reduction-block tails, and numerical correctness. This executor remained an internal forced path through Step 9 and is connected to automatic dispatch by Step 10; Step 11 subsequently added block-scoped projected tile ranges. At present, the local fragment lifetime is still one native reduction tile, so grouping changes tile traversal and ownership but does not yet promise that a register fragment spans all physical tiles in a multi-tile reduction block.
10. Add an initial schedule policy that chooses between complete-reduction fragments and reduction-blocked fragments.
    The policy SHOULD use projected working-set size, value size, physical tile extents, reduction extent, and conservative backend cache budgets.
    It MUST remain overridable internally so both paths can be tested and benchmarked deterministically.
    **Status: complete.** `iai::detail::native_tiled_reduce_policy.hpp` now owns native reduction dispatch. Fragment-incompatible calls remain on the scalar semantic path. For compatible calls, the policy models a complete-reduction shape with one physical tile on every free axis and the full physical-tile count on every reduction axis; the existing projection metrics therefore include operand value sizes, physical tile extents, broadcasts, and the dynamic reduction extent. The initial portable host policy uses conservative replaceable budgets of 256 KiB for the complete-reduction footprint and 32 KiB for a native cache block. A single reduction tile or a fitting complete footprint selects complete fragments; otherwise the policy selects reduction-blocked fragments. Six generic `(free-axis factor, reduction-axis factor)` candidates are clamped to the actual tile grid and ranked by the Step 9 mapping-derived cost model, with the unit native block as a fallback when no candidate fits. Compile-time internal overrides force scalar, complete-fragment, or reduction-blocked execution without public API or runtime environment state. Tests verify the footprint arithmetic and crossover, force and execute both fragment paths, and exercise automatic blocked dispatch numerically. Performance comparisons are deliberately deferred until all Phase 1 implementation work is complete. The current constants can be changed manually as described in `docs/iai-cache-budgets.md`; richer compile-time policy injection is promoted to Phase 2, while portable topology discovery remains in Phase 7.
11. Hoist binding and address work to its invariant loop level.
    A physical input tile or tile-range descriptor MUST NOT be rebound separately for every register fragment when it is invariant across those fragments.
    **Status: complete.** `native_tile_range` partially evaluates a physical-tile origin through the source accessor once and retains the native outer-grid strides as the residual affine tile mapping. `projected_native_tile_range` composes named-axis projection with that descriptor. The complete-fragment executor constructs one projected range tuple per owned output tile and reuses it for every full and tail register fragment; the unit blocked executor does the same per owned output tile, while the rectangular executor constructs one tuple per current output/reduction block. Binding a tile from the descriptor therefore requires only projected residual coordinates and an accessor-relative offset, with no repeated global tiled mapping, division, remainder, or allocation. Tests cover accessor/handle preservation and affine outer-left range strides, verify that a range origin is bound once, and count complete-fragment input bindings across multiple register fragments and reduction tiles.
12. Decide whether the existing compatibility forwarding entry point for `layout_tiled::reduce` is still required, then retain or remove it and move or duplicate the existing reduction tests under the `iai::md_reduce` API.
    **Status: complete.** The compatibility entry point was removed: `layout_tiled` now installs and exports only the tiled layout mappings, while all multidimensional bindings and execution remain under Iai. The former public reduction coverage moved to `tests/layout_tiled/iai_reduce.test.cpp`, links `iai::iai`, and directly exercises `iai::in`, `iai::out`, `iai::inout`, and `iai::md_reduce`, including compile-time rejection cases and debug contract checks. GEMM, GEMV, and tensor-contraction benchmarks and all documentation examples now use the Iai API; their benchmark targets link `iai::iai`. No deprecated alias is retained because the forwarding header existed only during the extraction and would preserve the ownership ambiguity this phase resolves.
13. Evaluate a small vocabulary of named semantic updater operations.
    **Status: withdrawn.** A host-only prototype exposed named `add`, `mul`, `mul_add`, `min`, and `max` updater types. Controlled GCC and Clang comparisons showed that the named `mul_add` type was consistently slower than the equivalent lambda written at the `md_reduce` call site, with particularly large extra instruction and L1-reference counts under GCC. The prototype, its tests, and its benchmark variants were removed. Ordinary updater lambdas remain the public and benchmark convention; semantic operations may be reconsidered only when a backend can justify them without compromising host code generation.

All Phase 1 implementation steps and the final controlled benchmark comparison are complete. Results are recorded in `docs/iai-phase1-results.md` with raw data under `results/phase1/tigerlake-i5-1135g7-4c-base-2.4ghz/`.

Acceptance criteria:

- all current reduction tests pass;
- GEMM, GEMV, and tensor contractions remain correct;
- lazy binding construction performs no element access or work enumeration, while returning from `md_reduce` implies completion;
- native tile slicing preserves custom accessor and data-handle semantics;
- hot local loops access affine inner-tile mappings and contain no global tiled mapping, division, remainder, runtime axis search, or indirect callback;
- optimized assembly retains vector FMA;
- complete-reduction fragments store once after the full reduction;
- cache-blocked fragments load and store partial accumulators only at reduction-block boundaries, outside the innermost local reduction loop;
- input tiles in the cache-blocked path are bound once per invariant tile coordinate, not once per register fragment;
- at sizes whose complete-reduction fragment footprint exceeds private cache, cache-blocked execution materially reduces normalized L1 misses and LLC traffic relative to the current `layout_tiled::reduce` prototype;
- at small sizes, the selected schedule does not materially regress relative to the current fragment executor;
- the native-tiled optimized path remains competitive with or improves upon the handwritten DT/TV baselines without packing; and
- both schedule paths can be forced in tests and benchmarks so their crossover is measurable rather than inferred.

### Phase 2: native-tiled performance maturation

This phase is next because it can strengthen the principal GEMM and tensor
contraction results without changing the public `md_reduce` expression or
introducing kernel-specific paths. Phase 1 established correctness, native
spatial locality, a register microkernel, and an initial cache-blocked
schedule. It did not yet make the storage tile, worker cache block, bounded
reduction stage, register-resident output group, and parallel work order fully
independent scheduling levels.

The expected GEMM gain comes from retaining projected A-like or B-like tile
sets across several output tiles, reducing partial-output traffic across a
bounded K stage, and choosing output traversal for upper-cache reuse. The
expected tensor-contraction gain is broader: its batch, row, and column axes
must receive independent block factors, so a profitable batch extent of one
does not prevent larger row/column reuse regions. These are consequences of
operand projections and therefore apply to other contractions with the same
structural reuse, not just to the benchmark kernels.

1. Establish performance targets and repeatable shape suites for GEMM and
   representative rank-4 tensor contractions.
   Include sizes that cross private-cache and LLC capacity, compare GEMM with
   BLAS and DT/TV, and compare contractions with DT/TV/cache-blocked baselines
   and a measured compute Speed of Light.
2. Extend the restricted CuTe-inspired lowering from one bound tile to a
   hierarchical residual layout for a rectangular group of native tiles.
   Compose named-axis projection with `(local tile, tile grid)` decomposition,
   rectangular division, slicing, and conservative mode coalescing. Partially
   evaluate coordinates at the output-block, reduction-stage, physical-tile,
   and fragment levels so each hot loop sees only its residual mapping. Carry
   non-overlap information established for projected output/input ranges into
   the bound residual views in a form the compiler can use for optimization.
   Such information must be conservative: output/input non-overlap may be
   propagated when proved by the `md_reduce` contract, but distinct inputs
   must not be described as non-aliasing merely because they have different
   bindings. Custom accessors and data handles must remain supported.
3. Represent three independently tunable execution shapes in addition to the
   physical storage tile:
   a worker-owned output/cache block, a bounded reduction stage, and a
   register-resident output-fragment group. None of these shapes may be
   inferred by treating the physical tile as interchangeable with another
   level.
4. Replace the six symmetric free/reduction-axis candidates with a bounded
   per-axis candidate search. Derive asymmetric shapes from operand
   projections, mapping strides, value sizes, distinct tile footprints,
   prospective reuse, partial-accumulator traffic, available output
   parallelism, and injected cache/register budgets. The search MUST remain
   kernel-name independent and must handle batch and multiple reduction axes
   without assigning every free axis the same factor.
5. Add projection-derived loop-order and output-block traversal choices.
   Consecutive work should retain a profitable invariant input region when
   possible, analogous to cache-aware tile rasterization, while preserving
   exclusive output ownership and balanced OpenMP work. Batch, row, and column
   axes are identified only by their occurrence in operand projections.
6. Generalize the blocked fragment executor to choose between
   input-stationary and accumulator-stationary local schedules. A bounded
   register group SHOULD remain live across several physical reduction tiles
   when its register budget permits; an input tile set SHOULD be consumed
   across the profitable output group before eviction. Binding, slicing, and
   address formation MUST stay at their invariant levels. Separate a static
   full-tile/full-stage hot path from residual tails: full-path extents and
   reduction bounds should be compile-time constants suitable for bounded K
   unrolling, while boundary handling remains outside that path. Once these
   invariants exist, allow the schedule policy to request software prefetch at
   reduction-stage boundaries. Prefetch distances and operands must derive
   from the selected mapping and schedule, remain optional, and be retained
   only when measurement shows a general benefit; there is no unconditional
   prefetch in the semantic executor.
7. Introduce a compile-time injectable native schedule policy with separate
   usable budgets for register fragments, reduction stages, and worker-private
   cache blocks. Keep the documented deterministic fallback and manual tuning
   path. Automatic host cache discovery and deployment calibration remain in
   Phase 7.
8. Inspect generated code and measure each schedule component independently.
   Verify that useful output/input non-overlap reaches the optimized loops,
   that the full-tile path is not contaminated by residual tests, and that any
   selected prefetch occurs at its intended reduction-stage boundary.
   Retain only improvements that generalize across GEMM and tensor contraction
   shapes; do not add recognized-kernel branches, hidden packing, allocation,
   or BLAS-specific semantics.

Acceptance criteria:

- storage tiles, cache blocks, reduction stages, and register groups have
  distinct representations and independently testable shapes;
- multi-tile slices preserve source accessor and data-handle semantics;
- proven output/input non-overlap is available to optimized residual access,
  without imposing a false no-alias contract between input bindings;
- fixed block/stage coordinates disappear from residual hot-loop mappings;
- full tiles and stages use a separately inspectable static hot path with
  compile-time bounds and bounded reduction unrolling, while tails stay
  correct on a distinct path;
- candidate generation can select different factors for different free axes
  and for each reduction axis;
- the selected traversal order is explained by binding projections rather
  than GEMM or contraction names;
- forced input-stationary and accumulator-stationary paths are correct for
  GEMM, batched contraction, broadcasts, tails, and multiple reduction axes;
- software prefetch is policy-controlled and mapping-derived, and disabling it
  preserves identical semantics and a viable schedule;
- GEMM narrows its remaining same-session gap to BLAS without regressing
  GEMV materially; and
- tensor contraction improves against Phase 1 at cache-capacity-crossing
  sizes while retaining its advantage over handwritten DT/TV baselines.

### Phase 3: standard-layout reductions

1. Recognize standard mappings explicitly.
2. Implement internal affine logical-block access without copying data.
3. Reuse the Phase 1 cache-block descriptors without requiring physical tile extents.
4. Reuse scalar, complete-reduction fragment, and reduction-blocked fragment executors over affine blocks.
5. Support shortened standard-layout fringes.
6. Add mixed standard/tiled operand coverage.

Acceptance criteria:

- standard left/right GEMM and GEMV are correct;
- logical cache blocking appears in generated loops;
- SIMD is selected only for unit-stride compatible operands;
- no allocation or packing appears in implementation or profiles; and
- mixed supported mappings are correct.

### Phase 4: `md_map`

1. Resolve callback and output-binding semantics.
2. Generalize axis validation for zero reduction axes.
3. Implement scalar and SIMD map executors.
4. Add unary, binary, broadcast, heterogeneous, and rank-N tests.
5. Add standard, tiled, and mixed mapping coverage.

Acceptance criteria:

- no reduction machinery remains in map hot loops;
- compatible maps vectorize on the output stride-one axis; and
- output work partitions remain race-free.

### Phase 5: stencil semantics

1. Implement compile-time affine offset descriptions.
2. Resolve the public neighborhood/shift syntax.
3. Implement interior output spaces and the initial boundary policy.
4. Implement a mapping-independent scalar stencil executor.
5. Add five-point, nine-point, higher-radius, and rank-3 tests.

Acceptance criteria:

- results match direct indexed references;
- global boundary behavior is explicit and tested; and
- overlapping stencil input/output is rejected initially.

### Phase 6: optimized stencils

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

### Phase 7: portable cache policy and deployment tuning

1. Connect the Phase 2 injectable schedule policy to a portable cache model.
2. Add optional host cache-topology discovery, including cache sharing by the
   active worker placement.
3. Provide an explicit execution-policy override for applications that need
   deterministic deployment tuning.
4. Retain compile-time policy types for controlled targets, testing, and
   benchmarking without relying on global build macros.
5. Calibrate usable-capacity safety factors and fallback behavior across a
   representative set of systems as the library's technology readiness grows.

Acceptance criteria:

- correctness is independent of cache discovery and policy values;
- detected and user-supplied cache models select blocks through the same
  mapping-derived cost model;
- unsupported platforms have a documented deterministic fallback;
- cache sharing and active worker placement are represented explicitly; and
- policy tests inject synthetic cache models and do not depend on the test
  machine's topology.

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
- output fragments remain live across each selected reduction block;
- complete-reduction fragments store once after the full reduction;
- cache-blocked fragment loads/stores occur only at reduction-block boundaries;
- input tile binding is hoisted outside fragment loops when invariant;
- standard blocks use affine stride/induction arithmetic;
- native tiled inner loops do not invoke global tiled mapping;
- stencil interiors do not select source tiles per element; and
- OpenMP covers only independent output work.

### Performance validation

- benchmark sizes cross the expected L1, private-L2, and LLC working-set boundaries;
- complete-reduction and cache-blocked schedules can be selected independently;
- cache counters are normalized by logical work and accompanied by instructions and elapsed time;
- native cache blocking is compared with the previous reduction executor and handwritten DT/TV schedules;
- regressions and improvements are checked in serial and OpenMP configurations; and
- benchmark tile-size controls affect the instantiated or selected mappings under test.

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
11. Operand bindings are lazy descriptions and execution algorithms are eager terminal operations.
12. Logical accumulator lifetime is independent of whether partial state resides in registers or an exclusively owned output cache block.
13. Cache-block schedules derive reuse from operand projections and mappings rather than kernel names.
14. Native tiled execution combines physical spatial locality with schedule-derived temporal locality without hidden packing.
