# Design of layout-aware reductions

`layout_tiled::reduce` describes a reduction in terms of logical axes and lets
the implementation derive an execution schedule from those axes and the
physical tiled layouts of the operands. The public expression contains the
mathematics but does not prescribe loop order, blocking, vector direction,
thread placement, or accumulator shape.

For example, GEMM is written as:

```cpp
enum : std::size_t { I, J, K };

layout_tiled::reduce<K>(
  layout_tiled::out<I, J>(C, 0.0),
  layout_tiled::in<I, K>(A),
  layout_tiled::in<K, J>(B),
  [](auto& c, auto a, auto b) { c += a * b; }
);
```

The axis order in each binding is the dimension order of that operand's
`mdspan`. `I` and `J` are free axes because they appear in the output, while
`K` is a reduction axis because it is a template argument to `reduce`. The
implementation can consequently recover the complete iteration space and each
operand's projection into it without a GEMM-specific interface.

The abstraction is header-only and host-only in its current form. Its design
separates reusable axis and layout analysis from host execution so another
backend can be added without changing the public expression.

## Hierarchical execution model

The implementation follows the same hierarchy as the tiled storage:

```text
reduce expression
  -> compile-time axis graph and validation
  -> logical problem extents and physical tile grid
  -> flattened output-tile work items
  -> one output tile owned by one worker
  -> output fragment or scalar accumulator
  -> reduction-tile traversal
  -> affine accesses within bound operand tiles
  -> updater invocation
```

This hierarchy is important for both generality and performance. Logical axes
define what must be computed. The tiled mappings define where reusable blocks
of data live. The executor then selects a scalar or register-fragment kernel
without changing the operation supplied by the caller.

There are five main implementation layers:

1. Public bindings encode operand roles, axis projections, and output
   initialization in types.
2. Compile-time analysis validates the expression and derives axis properties.
3. The N-D tiled-view adaptor converts global tiled storage into a grid of
   affine physical tiles.
4. The host scheduler assigns disjoint output tiles and orders their axes.
5. A general scalar executor or an optimized fragment executor performs the
   local reduction.

## Bindings are a typed axis graph

The public `in`, `out`, and `inout` functions construct lightweight binding
objects:

- `in<Axes...>(view)` marks an operand read-only and records the axis carried
  by each dimension.
- `out<Axes...>(view, identity)` initializes each accumulator from an explicit
  identity and writes the final value to the view.
- `inout<Axes...>(view)` initializes each accumulator from the current output
  element, allowing operations such as `C += A * B`.

The bindings contain the source `mdspan` through a tiled-view adaptor. They do
not allocate storage, build a runtime expression tree, or erase operand types.
The identity and every axis list remain part of the concrete call type.

The final argument is the updater. It must be callable as

```cpp
void updater(output_value&, const input_value&...);
```

Inputs reach the updater in binding order. The accumulator is deliberately
mutable rather than returned, which supports multiply-adds, minima, maxima,
logical reductions, and user-defined state transitions with the same control
structure.

At compile time, `valid_reduce_call` checks the structural contract:

- the first argument is exactly one output binding;
- one or more input bindings follow it and the updater is last;
- binding rank equals the number of named axes;
- axes within every binding and within the reduction list are unique;
- the union of axis IDs is dense in `[0, problem_rank)`;
- output axes are exactly the complement of the reduction axes;
- every reduction axis occurs in at least one input;
- shared static logical extents and static tile extents agree; and
- the updater has the required argument and `void` return types.

Dynamic logical extents are merged by axis and checked with debug assertions.
The same setup verifies that each logical extent is divisible by its physical
tile extent. Output/input overlap is forbidden by contract and is detected for
ordinary pointer handles in debug builds.

This analysis turns an apparent runtime operation into a specialized template
instantiation. For a particular set of bindings, the compiler knows the
problem rank, operand ranks, axis projections, tile extents, inner and outer
layouts, value types, accessor types, and updater type.

## The N-D tiled-view adaptor

Calling a tiled `mdspan` with global logical coordinates normally asks its
mapping to determine both the physical tile and the element's location inside
that tile. Repeating that full mapping in a hot reduction loop can introduce
integer division, remainder, multiplication, and layout-dependent offset work
for every operand of every update.

`basic_tiled_view` separates those two address levels. It views an N-D tiled
`mdspan` as an N-D grid of ordinary affine tile `mdspan`s while preserving:

- element and reference semantics;
- the original accessor and data-handle type;
- every static physical tile extent; and
- the tiled mapping's `inner_layout_type`.

Binding a physical tile proceeds as follows:

1. Tile coordinates are multiplied by static tile extents to form the logical
   origin of the tile.
2. `tile_offset_outer` is called once to locate that physical tile in the
   source mapping.
3. The source accessor offsets the data handle once.
4. A static-extents `mdspan` with `layout_left` or `layout_right` inner mapping
   is constructed over the tile.

If an axis coordinate is decomposed as

```text
x[a] = tile_coordinate[a] * tile_extent[a] + local_coordinate[a],
```

the tiled mapping handles the first term at tile-binding time. The inner tile
mapping handles only `local_coordinate` afterward. Its strides and extents are
compile-time constants, so ordinary element access becomes affine pointer
arithmetic that the compiler can inline and strength-reduce.

The important claim is not that address arithmetic disappears entirely. It is
that the expensive global tiled mapping is moved out of the innermost element
loop. In the fragment executor, one input-tile binding is amortized over an
entire register fragment and all elements of that reduction tile. The hot FMA
loop sees bound affine tiles rather than repeatedly decomposing global indices.

## Projection from problem coordinates to operand coordinates

All loops use one coordinate array indexed by logical problem axis. Operands
usually carry only a subset of those axes. A GEMM A tile uses `(I,K)`, a B tile
uses `(K,J)`, and the C tile uses `(I,J)`.

`bind_projected_tile` projects problem tile coordinates into the dimension
order of one binding and asks its tiled view for the resulting physical tile.
`projected_tile::at` performs the analogous projection for local coordinates.
An axis absent from an operand is simply not included in the projection, which
implements broadcasting without a special broadcast node or stride-zero view.

For GEMM, a problem coordinate `(i,j,k)` therefore becomes:

```text
A.at(i,j,k) -> A_tile(i,k)
B.at(i,j,k) -> B_tile(k,j)
C.at(i,j,k) -> C_tile(i,j)
```

The projection loops have compile-time bounds and are forced inline on Clang
and GCC. Once specialized, their small coordinate arrays and index selections
are normally scalarized or folded into the surrounding address arithmetic.
There is no virtual lookup, hash table, runtime axis search, or indirect
function call.

This mechanism is what makes the executor N-D. Adding batch axes, omitting an
axis from a broadcast operand, or using two reduction axes changes binding
types, not the core traversal code.

## Deriving the schedule from layouts

`host_reduce` combines all bindings and builds one logical extent and one tile
count per problem axis. Static tile extents come from the binding types;
dynamic logical extents come from the views.

### Output-tile traversal

Only the output-tile grid is flattened into work items. `decode_output_tile`
reconstructs its N-D coordinates according to the output mapping's
`outer_layout_type`, preserving the physical tile traversal order of
`layout_left` or `layout_right`.

When OpenMP is enabled, the flattened grid is executed with

```cpp
#pragma omp parallel for schedule(static)
```

Each worker owns complete output tiles. Reduction axes are never divided
between workers, so there are no atomics, partial output buffers, reduction
trees, or synchronization inside a tile. Without OpenMP, exactly the same work
item order executes serially.

### Output axes

The output inner layout determines local traversal. For an inner
`layout_right`, the final output dimension is stride one; for an inner
`layout_left`, the first output dimension is stride one. That axis becomes the
SIMD axis. The adjacent output axis becomes the row-unrolled axis. A rank-one
output has only the SIMD axis and uses a single fragment row.

If an output has more than two axes, all remaining output axes identify
fragment planes. The executor iterates those planes and applies the same
two-dimensional fragment kernel within each one.

### Reduction axes

For each reduction axis, compile-time analysis finds the minimum static inner
stride among inputs that carry that axis. Reduction axes are sorted so that an
axis with lower input stride becomes the inner recursive loop. Ties preserve
the order supplied to `reduce<Axes...>`.

The same order is used at two levels:

- across physical reduction tiles; and
- across elements inside each reduction tile.

Once all reduction-tile coordinates are known, the executor binds the
projected input tiles and traverses their affine inner mappings. This naturally
rebinds A and B as K changes in GEMM while the output tile remains fixed for
the complete K reduction.

## General scalar executor

The scalar executor is the semantic reference path and supports every valid
binding combination. Within one output tile it:

1. traverses every local output coordinate in inner-layout order;
2. creates a scalar accumulator from the output identity or current value;
3. visits every reduction tile and every local reduction coordinate;
4. binds the projected input tiles and invokes the updater; and
5. stores the accumulator once after the complete reduction.

Even this path benefits from static tile extents, compile-time axis recursion,
inlined projection, and affine inner-tile access. Its priority is complete
coverage: arbitrary operand rank, multiple reduction axes, heterogeneous input
types, broadcasts, custom arithmetic values, and either tiled inner layout.

The scalar executor may bind input tiles more frequently than a
kernel-specialized loop nest because each output element owns its full
reduction. That is an intentional baseline tradeoff. It provides one correct
execution model for all expressions, while the fragment path increases reuse
and address amortization for the important arithmetic case.

## Register-fragment executor

The optimized executor is selected at compile time when:

- the output and all inputs have arithmetic value types; and
- every input that carries the chosen output SIMD axis is contiguous on that
  axis in its inner layout.

An input that does not carry the SIMD axis is invariant along it and is
broadcast naturally. If an input carries the axis with a non-unit inner stride,
the call remains valid but uses the scalar executor.

The fragment shape is target-aware but expressed as portable fixed-size scalar
arrays:

```text
MR = 4 rows for rank >= 2 output, otherwise 1
NR = 2 * native SIMD lane count
```

For FP64 on AVX2, the native lane count is four and `NR` is eight. The logical
fragment is therefore 4-by-8. The implementation does not require a public SIMD
type: fixed extents, forced inlining, compile-time row expansion, and OpenMP
SIMD annotations give the compiler a conventional vectorizable loop.

For each output fragment, the executor performs three phases:

1. Initialize the complete fragment from the identity or current output.
2. Keep the fragment live across every reduction tile and every reduction
   element while invoking the updater.
3. Store the complete fragment once after the reduction finishes.

Full fragments use `static_for<MR>` to expand rows at compile time and a
constant-bound SIMD column loop. This exposes several independent accumulator
chains, allowing an out-of-order core to hide FMA latency and issue multiple
vector operations per cycle. The output load/store traffic and tiled output
mapping are outside the reduction loop.

Physical tile extents do not have to be multiples of `MR` or `NR`. Shortened
row and column loops handle the tail inside a physical tile. Because logical
extents must still be divisible by physical tile extents, these are
register-block tails rather than padded global-layout fringes.

The updater remains generic in this path. A multiply-add lambda is inlined into
vector FMAs; another simple arithmetic updater can specialize into different
instructions. Complex updater bodies are still semantically supported, though
their code generation may not resemble a dense linear-algebra microkernel.

## Why optimized code survives the abstraction

The abstraction deliberately puts all structural choices in types and keeps
the runtime state small. After specialization and inlining there is:

- no expression graph traversal;
- no dynamic allocation;
- no virtual dispatch or type erasure;
- no indirect updater call;
- no runtime search for an axis;
- no full global tiled-layout lookup in the inner element loop; and
- no cross-thread reduction or output synchronization.

Static tile extents make loop bounds and inner strides constants. Recursive
axis walkers instantiate a fixed-depth loop nest rather than interpreting an
N-D loop descriptor. `if constexpr` removes unused initialization modes,
layouts, compatibility checks, and executor branches. The lambda type is known
at the call site, so `std::invoke` disappears and its body becomes part of the
kernel.

The fragment path further gives the optimizer the structure it needs:

- unit-stride input and output access on the vector axis;
- several independent accumulators;
- output values loaded before and stored after the reduction;
- repeated use of already-bound affine input tiles; and
- constant fragment dimensions matched to the target vector width.

This is why a high-level axis expression can produce code similar to a
handwritten blocked microkernel. The API is generic, but each concrete call is
a separately compiled schedule specialized for its layouts and updater.

## Adapting the same executor to other kernels

The core machinery does not identify kernels by name. It only sees projected
axes, reduction axes, layouts, and an updater.

### GEMV

For `y[i] = sum_k A[i,k] * x[k]`:

```cpp
enum : std::size_t { I, K };

layout_tiled::reduce<K>(
  layout_tiled::out<I>(y, 0.0),
  layout_tiled::in<I, K>(A),
  layout_tiled::in<K>(x),
  [](auto& yi, auto aik, auto xk) { yi += aik * xk; }
);
```

`x` has no `I` axis, so it is broadcast across the output. For the fragment
path, `I` should be contiguous in A's inner layout. A suitable arrangement is:

```cpp
constexpr std::size_t TI = 8;
constexpr std::size_t TK = 128;

using matrix_layout = layout_tiled::layout_rtl<TI, TK>;
using output_layout = layout_tiled::layout_ltl<TI>;
using vector_layout = layout_tiled::layout_ltl<TK>;
```

The matrix's left inner layout makes `I` contiguous, so one `x[k]` value can be
broadcast across a SIMD group of output rows. Conventional row-major A remains
correct, but makes `I` strided and therefore selects the scalar path.

### Batched contraction

Adding a batch axis requires no executor change:

```cpp
enum : std::size_t { BATCH, I, J, K };

layout_tiled::reduce<K>(
  layout_tiled::out<BATCH, I, J>(C, 0.0),
  layout_tiled::in<BATCH, I, K>(A),
  layout_tiled::in<BATCH, K, J>(B),
  [](auto& c, auto a, auto b) { c += a * b; }
);
```

The batch axis becomes another output-tile and fragment-plane coordinate. A
physical batch tile of extent one expresses repeated GEMMs directly, while a
larger batch tile allows reuse choices to be encoded in storage.

### Multiple reduction axes and heterogeneous inputs

A rank-four contraction with two reduction axes has the same form:

```cpp
enum : std::size_t { I, J, K, L };

layout_tiled::reduce<K, L>(
  layout_tiled::out<I, J>(C, 0.0),
  layout_tiled::in<I, K, L>(A),
  layout_tiled::in<K, L, J>(B),
  [](auto& c, auto a, auto b) { c += a * b; }
);
```

The recursive reduction walker orders and traverses both axes. Unary
reductions and additional heterogeneous inputs use the same mechanism because
the input tuple and updater argument pack are variadic.

## Choosing layouts and tile shapes

The API guarantees correctness for every valid tiled binding, but layout still
communicates the performance strategy.

- Make the output's stride-one inner axis the dimension intended for SIMD.
- Inputs that carry that output axis should make it stride one as well; inputs
  that omit it are efficient broadcasts.
- Use the outer layout to place consecutively visited output tiles next to one
  another in storage.
- Choose reduction tile depth large enough to amortize tile binding and loop
  overhead, while keeping the active operand tiles in the intended cache.
- Rectangular tiles are often preferable because output-fragment dimensions,
  reduction depth, and cache footprints have different roles.
- Preserve enough output tiles to distribute work evenly across threads.

For GEMM, a physical tile family is `A[TM,TK]`, `B[TK,TN]`, and `C[TM,TN]`.
The complete three-tile working set is

```text
sizeof(value) * (TM*TK + TK*TN + TM*TN).
```

Increasing `TK` can reduce reduction-tile setup, while `TN` controls the amount
of contiguous vector-axis data and `TM*TN` controls output-tile granularity.
Those concerns are not generally balanced by a square tile.

## Backend boundary

The public functions and most detail machinery are backend-independent:

- bindings and axis lists;
- compile-time validation;
- extent and tile-shape analysis;
- tiled-view and projected-tile access; and
- updater invocation.

`dispatch_reduce` currently sends the analyzed expression to `host_reduce`,
which owns OpenMP scheduling and the portable host fragment kernel. A future
device dispatcher can reuse the expression and analysis layers while replacing
output work assignment, fragment storage, and local execution. The public API
does not encode OpenMP, threads, vector registers, or host pointers, so adding a
CUDA backend does not require a second contraction language.

## Numerical and storage contracts

The current implementation has the following deliberate constraints:

- exactly one output binding must be first;
- at least one input is required;
- at least one output axis is required, so scalar-output reductions are not
  implemented yet;
- logical extents must be divisible by static physical tile extents;
- operands sharing an axis must share its logical and tile extents;
- output storage must not overlap input storage; and
- reduction axes are not split across threads.

The executor may reassociate operations according to its layout-derived loop
order, fragment shape, compiler target, and OpenMP configuration. This matches
optimized BLAS-style numerical semantics: results are expected to satisfy an
appropriate numerical tolerance, but bitwise reproducibility across builds,
targets, tile shapes, or thread counts is not guaranteed.

Padded global fringes, scalar outputs, index-aware callbacks, explicit
reproducibility modes, device fragments, and parallel split reductions are
natural extensions but are outside the current host implementation.

## Source organization

The implementation is split into three headers:

- `include/layout_tiled/reduce.hpp` defines `in`, `out`, `inout`, and the
  lambda-last `reduce` entry point.
- `include/layout_tiled/detail/tiled_view.hpp` provides the generic N-D view of
  tiled storage as affine physical tiles.
- `include/layout_tiled/detail/reduce.hpp` contains validation, axis analysis,
  projection, scheduling, scalar execution, fragment execution, and host
  dispatch.

This separation keeps the public surface small while making the address,
schedule, and backend layers independently replaceable.
