# `std::execution` integration for Iai operations

## Status and scope

This document records a future integration between Iai operations and the
C++26 sender/receiver execution model. It proposes that
`iai::make_md_reduce` return a cold sender while the existing
`iai::md_reduce` remains an eager terminal convenience function.

The initial integration should rely on an existing execution implementation
for schedulers, sender algorithms, receivers, operation states, cancellation,
and structured concurrency. Candidate implementations include NVIDIA
[stdexec](https://github.com/NVIDIA/stdexec) and
[beman::execution](https://github.com/bemanproject/execution).

This design distinguishes:

1. task-graph composition;
2. scheduler-aware parallel execution of Iai output blocks; and
3. fusion of compatible Iai operations.

Returning a sender provides the first property. It does not automatically
provide the latter two.

## Motivation

Iai bindings are already lazy descriptions of operand roles and axis
projections. The current algorithm is intentionally eager:

```cpp
iai::md_reduce<K>(output, inputs..., updater);
```

This remains useful for direct scientific kernels and benchmarks. A separate
factory can provide explicit deferred execution without making the existing
API surprising:

```cpp
auto reduction = iai::make_md_reduce<K>(output, inputs..., updater);
```

The returned value describes a mutation but performs no element access, work
enumeration, scheduling, or allocation until it is connected to a receiver and
started. This is the sender equivalent of the operation object anticipated by
the main design notes.

Senders enable sequencing, concurrency, structured lifetime, errors,
cancellation, coroutine interop, and scheduler selection:

```cpp
namespace ex = stdexec; // Or the selected C++26 implementation namespace.

auto work = ex::starts_on(
  scheduler,
  iai::make_md_reduce<K>(
    iai::out<I, J>(C, 0.0),
    iai::in<I, K>(A),
    iai::in<K, J>(B),
    [](auto& result, auto lhs, auto rhs) { result += lhs * rhs; }
  )
) | ex::then([] { /* dependent work */ });

ex::sync_wait(std::move(work));
```

The sender owns its operation description, not the operand storage. Every
referenced allocation must remain valid until the operation completes.

## Shared operation descriptor

The eager and sender APIs should lower through one provider-independent
descriptor:

```text
projected bindings + identity + updater
                 |
                 v
         md_reduce_operation
          /               \
 eager host execution      sender/provider adaptor
```

An illustrative type is:

```cpp
template <class ReductionAxes, class Output, class Inputs, class Updater>
class md_reduce_operation;
```

It stores concrete bindings and the updater by value. It exposes internal
compile-time analysis but is not itself a sender and does not know about an
execution-library namespace. This prevents the core Iai implementation from
depending on stdexec or Beman and prevents duplicate validation/lowering code.

`iai::md_reduce` constructs the descriptor and executes it immediately through
the existing host backend. `iai::make_md_reduce` wraps the same descriptor in a
provider-specific sender.

The eager API SHOULD NOT be implemented as `sync_wait(make_md_reduce(...))`.
Doing so would impose a sender dependency and operation-state overhead on
users who requested direct execution.

## Sender semantics

The reduction sender is cold and single-shot unless its concrete provider and
stored operation types permit copying. Starting it performs the mutation.

Its logical completion channels are:

```text
set_value()             successful completion, no returned values
set_error(error)        scheduling or host execution failure
set_stopped()           stop accepted before mutation begins
```

The initial sender returns no output value because the caller already owns the
output `mdspan`. A later API may optionally forward a view, but that is not
needed for dependency ordering.

### Error behavior

Host updaters may throw. If an exception escapes after output initialization or
partial reduction, the sender reports an error but the output may be partially
modified. This matches the absence of transactional semantics in the eager
API and must be documented.

A device backend normally cannot propagate arbitrary C++ exceptions. Launch
or asynchronous device errors should use the provider's supported error
channel.

### Cancellation behavior

Cancellation is safe before the first output block begins. Once any output is
mutated, cancellation cannot generally restore prior values. The initial
contract should therefore be:

- observe the receiver stop token before enqueue/start;
- complete with `set_stopped` if cancellation wins before execution; and
- treat the reduction as unstoppable after mutation begins.

Cooperative cancellation between output blocks may be considered later only
with an explicit partial-result contract.

### Lifetime and ownership

The sender stores `mdspan` values, projected binding values, identities, and
the updater. These are lightweight non-owning descriptors. The caller must
keep every underlying allocation, custom accessor state referenced by pointer,
and execution context alive through completion.

No runtime expression graph, type erasure, or dynamic allocation is required
by Iai. A selected execution implementation may allocate operation state
according to its own scheduler and receiver environment.

## Integration levels

### Level 1: opaque leaf sender

The smallest implementation creates an operation state whose `start` function
calls the existing blocking `md_reduce` executor, then calls `set_value`.

This immediately supports `then`, `let_value`, `when_all`, scopes,
coroutines, and `sync_wait`. It is a useful semantic reference and inline
scheduler implementation.

It is not a satisfactory parallel scheduler integration. The scheduler sees
one opaque task while Iai starts serial or OpenMP work internally. Running
OpenMP from a thread-pool worker may block a scarce worker or oversubscribe the
machine.

### Level 2: scheduler-aware block execution

Phase 2 will expose a flattened set of independent, worker-owned output
blocks. The proper sender lowering should express those blocks through the
selected execution implementation's standard `bulk` algorithm:

```text
schedule on selected context
  -> bulk(output_block_count)
       execute one Iai block per bulk index
  -> complete after every output block
```

The bulk body decodes its N-D output-block coordinate and invokes the same
mapping-derived block executor used by the direct backend. The scheduler owns
worker assignment; Iai owns block shape, traversal, residual layouts, and the
local reduction schedule.

Requirements:

- each bulk index owns a disjoint output block;
- no OpenMP region is created inside a scheduler-provided parallel bulk;
- an inline or scheduler-unsupported fallback may use the direct host backend;
- output-block count and shape are fixed before mutation begins; and
- completion occurs only after every output block finishes.

The sender may obtain its scheduler and domain from the receiver environment
at connect time. If doing so is not portable across the selected provider
versions, an additional factory accepting a scheduler is preferable to hidden
global state:

```cpp
auto sender = iai::make_md_reduce<K>(scheduler, output, inputs..., updater);
```

The scheduler-agnostic form should remain the desired public model; a
scheduler-taking overload is an implementation bridge, not a separate
mathematical API.

### Level 3: domain-transformable Iai sender

Fusion and accelerator lowering require the operation to remain visible as a
structured sender node. Wrapping eager execution inside `then` hides the axes,
bindings, and operation category from the active execution domain.

The provider adapter should therefore create a sender expression node carrying
an Iai operation tag and the concrete `md_reduce_operation`. At `connect`, an
active domain may transform that node into:

- a host inline execution state;
- a host `bulk` expression;
- a GPU stream kernel launch;
- a specialized multiply-add backend; or
- a fused Iai operation.

stdexec documents domain transformation as the mechanism by which a scheduler
rewrites algorithms for a resource; its GPU `nvexec` domain applies this model
to CUDA-compatible sender algorithms. See the
[stdexec developer guide](https://nvidia.github.io/stdexec/developer/).

An unknown domain must either preserve a correct portable fallback or produce
a clear constraint failure. It must not silently move data or change backend.

## Fusion model

Sender composition creates a dependency graph, not a promise of kernel fusion.
Iai fusion should initially be limited to adjacent structured Iai nodes under
an Iai-aware domain. Foreign senders are valid composition partners and fusion
boundaries.

A first useful case is a future `md_map` producing a value consumed only by
`md_reduce`. Eliminating materialization is legal only when:

- the intermediate has no other observable consumer;
- the producer output and consumer input refer to the same logical index
  space and compatible mappings;
- aliases do not require materialization;
- the map can be evaluated at the consumer's access point;
- initialization, reduction order, exception behavior, and numerical policy
  remain valid;
- the fused register/cache footprint remains within policy; and
- neither operation introduces an execution or synchronization boundary that
  the domain must preserve.

The fused executor should compose axis projections and residual layouts, then
inline the map operation at the reduction input access. It should not create a
general runtime graph optimizer.

Other possible cases include fusing independent pointwise epilogues into a
reduction output store or combining multiple compatible outputs in a future
multi-output API. They require separate legality rules.

## Provider strategy

The core target should not expose a concrete stdexec or Beman type in its
headers. Add optional adapter targets, for example:

```text
iai::iai
iai::execution_stdexec
iai::execution_beman
```

Each adapter supplies provider-specific sender tags, completion signatures,
environment queries, connect/start customization, and algorithm names. The
public factory can live in a provider adapter header or be selected through a
configuration header, but only one provider namespace should be active in a
translation unit.

The implementation should use only the accepted C++26 subset where practical.
Provider extensions, especially domains and GPU schedulers, belong behind
adapter-specific code.

As of this design note:

- NVIDIA stdexec is an experimental reference implementation with broad
  composition, scheduler, domain, and GPU support. Its documented GPU path
  uses `nvc++`, and it does not currently support `nvcc`.
- Beman implements the standard vocabulary and algorithms, including `bulk`,
  but describes itself as under development and not ready for production use.

For initial work, target stdexec and treat Beman as a portability/conformance
adapter once the operation descriptor and semantics are stable.

## Relationship to GPU execution

An execution scheduler chooses where and when work runs; it does not make host
code device-callable or invent a GPU microkernel. A GPU sender lowering still
requires the device backend described in
[`iai-gpu-execution.md`](iai-gpu-execution.md).

For NVIDIA stdexec, an Iai domain integration could translate a structured
`md_reduce` node into an `nvexec` stream operation. Generic `then` wrapping is
insufficient because a host `md_reduce` call cannot become device code merely
by running after a GPU schedule sender.

The sender completion should naturally be stream-ordered. A blocking consumer
such as `sync_wait` provides host synchronization when requested, avoiding an
unconditional device synchronization inside `make_md_reduce`.

## Testing requirements

- Constructing a sender performs no element access, work enumeration, or
  mutation.
- Connecting without starting performs no mutation.
- Starting exactly once produces the same result as eager `md_reduce`.
- Bindings and updater values remain valid when moved into an operation state.
- `set_value` is delivered only after every output block completes.
- Stop requested before start yields `set_stopped` and leaves output unchanged.
- Host exceptions produce the declared error completion and never call
  `set_value`.
- Inline and parallel schedulers produce identical numerical results.
- Parallel bulk assigns every output block exactly once without nested OpenMP.
- `when_all`, `then`, `let_value`, scopes, and `sync_wait` interoperate.
- stdexec and Beman adapters pass the same provider-independent semantic test
  suite.
- Fusion tests compare both values and observable materialization/access counts
  with unfused pipelines.

## Implementation stages and estimate

Assuming Phase 2 is complete, one experienced generic C++ engineer should plan
for:

| Stage | Cumulative effort |
|---|---:|
| Shared operation descriptor and opaque sender for one provider | 1-3 person-weeks |
| Production completion, errors, stopping, and lifetime tests | 3-5 person-weeks |
| stdexec plus Beman adapters | 5-8 person-weeks |
| Scheduler-aware `bulk` output-block execution | 8-13 person-weeks |
| First constrained Iai fusion | 14-23 person-weeks |
| General fusion framework | 6-12 person-months |

The scheduler implementation removes the need for Iai to provide a thread
pool, queue, receiver framework, or structured-concurrency runtime. It does not
remove the work of defining Iai operation state, block parallelism,
cancellation after mutation, fusion legality, or backend lowering.

## Open questions

1. Should `make_md_reduce` be declared only by an optional provider adapter, or
   should the core expose a provider-parameterized customization point?
2. Can the standard receiver environment provide the selected scheduler
   portably enough, or is a scheduler-taking factory required initially?
3. Which error type should represent host exceptions and backend launch
   failures across providers?
4. Should a reduction sender be movable-only to make its single-shot nature
   explicit?
5. Should the first implementation expose only pre-start cancellation?
6. Which Iai sender nodes and domain customizations can stay within the C++26
   standard subset?
7. Where should fusion stop when an external sender preserves dependency order
   but does not expose its memory effects?

## External references

- [stdexec overview](https://nvidia.github.io/stdexec/)
- [stdexec sender and scheduler reference](https://nvidia.github.io/stdexec/reference/index.html)
- [stdexec domain customization guide](https://nvidia.github.io/stdexec/developer/)
- [beman::execution repository and status](https://github.com/bemanproject/execution)
- [P2300R10: `std::execution`](https://wg21.link/P2300R10)
