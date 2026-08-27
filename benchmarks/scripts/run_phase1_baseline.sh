#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_phase1_baseline.sh [BUILD_DIR] [OUTPUT_DIR]

Run the Phase 1 GEMM, GEMV, and tensor-contraction baseline in fresh,
four-thread processes pinned to four distinct physical cores.

Environment overrides:
  IAI_BASELINE_CPUS              Linux CPU IDs / hwloc P# list [default: 0,1,2,3]
  IAI_BASELINE_GEMM_SIZES        Space-separated extents [default: 512 1024 2048 4096]
  IAI_BASELINE_GEMV_SIZES        Space-separated extents [default: 1024 4096 8192 16384]
  IAI_BASELINE_TENSOR_SIZES      Space-separated extents [default: 32 64 128 256]
  IAI_BASELINE_TENSOR_L1_ONLY_SIZES
                                 Extents run with --l1-only [default: 32]
  IAI_BASELINE_GEMM_IMPLEMENTATIONS
                                 Space-separated selectors [default: blas dt tv reduce]
  IAI_BASELINE_GEMV_IMPLEMENTATIONS
                                 Space-separated selectors [default: blas dt tv reduce]
  IAI_BASELINE_REQUIRE_FIXED_FREQ
                                 Require performance governor and min=max=base
                                 frequency [default: 1]
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

build_dir=${1:-build/benchmark}
output_dir=${2:-results/baseline/"$(date --iso-8601=seconds | tr ':+' '__')"}
cpu_list=${IAI_BASELINE_CPUS:-0,1,2,3}
require_fixed_frequency=${IAI_BASELINE_REQUIRE_FIXED_FREQ:-1}

read -r -a gemm_sizes <<< "${IAI_BASELINE_GEMM_SIZES:-512 1024 2048 4096}"
read -r -a gemv_sizes <<< "${IAI_BASELINE_GEMV_SIZES:-1024 4096 8192 16384}"
read -r -a tensor_sizes <<< "${IAI_BASELINE_TENSOR_SIZES:-32 64 128 256}"
read -r -a gemm_implementations <<< "${IAI_BASELINE_GEMM_IMPLEMENTATIONS:-blas dt tv reduce}"
read -r -a gemv_implementations <<< "${IAI_BASELINE_GEMV_IMPLEMENTATIONS:-blas dt tv reduce}"
tensor_l1_only_sizes=${IAI_BASELINE_TENSOR_L1_ONLY_SIZES:-32}
IFS=',' read -r -a cpus <<< "$cpu_list"

if [[ ${#cpus[@]} -ne 4 ]]; then
  echo "error: IAI_BASELINE_CPUS must name exactly four CPUs" >&2
  exit 1
fi

declare -A physical_cores=()
places=()
for cpu in "${cpus[@]}"; do
  topology=/sys/devices/system/cpu/cpu"${cpu}"/topology
  policy=/sys/devices/system/cpu/cpu"${cpu}"/cpufreq
  if [[ ! -r "$topology/core_id" || ! -r "$topology/physical_package_id" ]]; then
    echo "error: cannot inspect topology for CPU ${cpu}" >&2
    exit 1
  fi
  core_key="$(<"$topology/physical_package_id"):$(<"$topology/core_id")"
  if [[ -n ${physical_cores[$core_key]:-} ]]; then
    echo "error: CPUs ${physical_cores[$core_key]} and ${cpu} are SMT siblings" >&2
    exit 1
  fi
  physical_cores[$core_key]=$cpu
  places+=("{${cpu}}")

  if [[ ! -r "$policy/scaling_governor" || "$(<"$policy/scaling_governor")" != performance ]]; then
    echo "error: CPU ${cpu} is not using the performance governor" >&2
    exit 1
  fi
  if [[ $require_fixed_frequency == 1 ]]; then
    base_frequency=$(<"$policy/base_frequency")
    min_frequency=$(<"$policy/scaling_min_freq")
    max_frequency=$(<"$policy/scaling_max_freq")
    if [[ $min_frequency != "$base_frequency" || $max_frequency != "$base_frequency" ]]; then
      echo "error: CPU ${cpu} is not locked to its base frequency (${base_frequency} kHz)" >&2
      exit 1
    fi
  fi
done

omp_places=$(IFS=,; echo "${places[*]}")
benchmark_dir="$build_dir/benchmarks"
gemm="$benchmark_dir/layout_tiled.benchmarks.gemm"
gemv="$benchmark_dir/layout_tiled.benchmarks.gemv"
tensor="$benchmark_dir/layout_tiled.benchmarks.tensor_contraction"
for executable in "$gemm" "$gemv" "$tensor"; do
  if [[ ! -x $executable ]]; then
    echo "error: missing benchmark executable: ${executable}" >&2
    exit 1
  fi
done

mkdir -p "$output_dir/logs"
if find "$output_dir" -maxdepth 1 -name '*.csv' -print -quit | grep -q .; then
  echo "error: output directory already contains CSV results: ${output_dir}" >&2
  exit 1
fi

metadata="$output_dir/metadata.txt"
{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_dirty=$(if [[ -n $(git status --short) ]]; then echo yes; else echo no; fi)"
  echo "build_dir=$build_dir"
  grep -E '^CMAKE_(BUILD_TYPE|CXX_FLAGS):' "$build_dir/CMakeCache.txt"
  echo "cpu_id_namespace=Linux CPU IDs (hwloc physical indices P#; not hwloc logical indices L#)"
  echo "cpu_list=$cpu_list"
  echo "omp_places=$omp_places"
  echo "threads=4"
  echo "gemm_implementations=${gemm_implementations[*]}"
  echo "gemv_implementations=${gemv_implementations[*]}"
  echo "nanobench_counter_scope=calling thread only (perf events are opened without inheritance)"
  echo "require_fixed_frequency=$require_fixed_frequency"
  echo "no_turbo=$(</sys/devices/system/cpu/intel_pstate/no_turbo)"
  echo
  lscpu
  echo
  lstopo-no-graphics --of console
  echo
  for cpu in "${cpus[@]}"; do
    topology=/sys/devices/system/cpu/cpu"${cpu}"/topology
    policy=/sys/devices/system/cpu/cpu"${cpu}"/cpufreq
    echo "cpu${cpu}_package=$(<"$topology/physical_package_id")"
    echo "cpu${cpu}_core=$(<"$topology/core_id")"
    echo "cpu${cpu}_thread_siblings=$(<"$topology/thread_siblings_list")"
    echo "cpu${cpu}_governor=$(<"$policy/scaling_governor")"
    echo "cpu${cpu}_base_khz=$(<"$policy/base_frequency")"
    echo "cpu${cpu}_min_khz=$(<"$policy/scaling_min_freq")"
    echo "cpu${cpu}_max_khz=$(<"$policy/scaling_max_freq")"
  done
  echo
  ldd "$gemm"
} > "$metadata"

run_one() {
  local benchmark=$1
  local executable=$2
  local size=$3
  local implementation=${4:-}
  local mode=${5:-}
  local result="$output_dir/${benchmark}-${size}.csv"
  local suffix=${implementation:-${mode:-all}}
  local stdout_file="$output_dir/logs/${benchmark}-${size}-${suffix}.stdout"
  local stderr_file="$output_dir/logs/${benchmark}-${size}-${suffix}.stderr"
  local time_file="$output_dir/logs/${benchmark}-${size}-${suffix}.time"
  local command=("$executable" --size "$size")
  if [[ -n $implementation ]]; then
    command+=(--implementation "$implementation")
  fi
  if [[ $mode == l1 ]]; then
    command+=(--l1-only)
  fi

  echo "Running ${benchmark} size=${size} implementation=${suffix} on CPUs ${cpu_list}"
  /usr/bin/time -v -o "$time_file" \
    taskset --cpu-list "$cpu_list" \
    env \
      OMP_NUM_THREADS=4 \
      OMP_DYNAMIC=FALSE \
      OMP_PROC_BIND=close \
      OMP_PLACES="$omp_places" \
      OMP_DISPLAY_AFFINITY=TRUE \
      OMP_AFFINITY_FORMAT="thread=%n affinity=%A" \
      OPENBLAS_NUM_THREADS=4 \
      GOTO_NUM_THREADS=4 \
      BLAS_NUM_THREADS=4 \
      MKL_NUM_THREADS=4 \
      MKL_DYNAMIC=FALSE \
      BLIS_NUM_THREADS=4 \
      VECLIB_MAXIMUM_THREADS=4 \
      NUMEXPR_NUM_THREADS=4 \
      "${command[@]}" > "$stdout_file" 2> "$stderr_file"

  if [[ ! -s $result ]]; then
    cp "$stdout_file" "$result"
  else
    tail -n +2 "$stdout_file" >> "$result"
  fi
}

for size in "${gemm_sizes[@]}"; do
  for implementation in "${gemm_implementations[@]}"; do
    run_one gemm "$gemm" "$size" "$implementation"
  done
done

for size in "${gemv_sizes[@]}"; do
  for implementation in "${gemv_implementations[@]}"; do
    run_one gemv "$gemv" "$size" "$implementation"
  done
done

for size in "${tensor_sizes[@]}"; do
  if [[ " ${tensor_l1_only_sizes} " == *" ${size} "* ]]; then
    run_one tensor_contraction "$tensor" "$size" "" l1
  else
    run_one tensor_contraction "$tensor" "$size"
  fi
done

echo "Baseline results written to ${output_dir}"
