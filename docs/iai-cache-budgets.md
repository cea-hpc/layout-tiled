# Manual Iai cache-budget tuning

Edit `host_native_tiled_schedule_policy` in
`include/iai/detail/native_tiled_reduce_policy.hpp`:

```cpp
static constexpr std::size_t COMPLETE_REDUCTION_BUDGET_BYTES = 256 * 1024;
static constexpr std::size_t CACHE_BLOCK_BUDGET_BYTES = 32 * 1024;
```

Then rebuild the consuming code.
