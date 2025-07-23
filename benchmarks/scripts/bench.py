#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess

# fmt: off
PB_DIMS = {
    "gemm":       [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096],
    "stencil_2d": [512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384],
    "stencil_3d": [ 64,  128,  192,  256,  384,  512,  768, 1024],
}
# fmt: on


def parse_nthreads_output(args: tuple):
    try:
        num_threads, output = args.split(",", 1)
        return True, int(num_threads), output
    except ValueError:
        raise argparse.ArgumentTypeError("argument must be in the format: <int>,<str>")


def parse_dims_output(args: tuple):
    try:
        dims, output = args.split(",", 1)
        return True, int(dims), output
    except ValueError:
        raise argparse.ArgumentTypeError("argument must be in the format: <int>,<str>")


def parse_output(args: tuple):
    try:
        output = args[0]
        return True, output
    except ValueError:
        raise argparse.ArgumentTypeError("argument must be in the format: <str>")


def remove_header_duplicates(path: str):
    with open(path, "r") as file:
        lines = file.readlines()
    if len(lines) <= 1:
        return
    header = lines[0]
    filtered_lines = [header] + [line for line in lines[1:] if line != header]
    with open(path, "w") as file:
        file.writelines(filtered_lines)


def problem_scaling(exec: str, bench: str, nthreads: int, output: str):
    dims = PB_DIMS[bench]
    env_vars = os.environ.copy()
    env_vars["OMP_PROC_BIND"] = "spread"
    env_vars["OMP_NUM_THREADS"] = f"{nthreads}"
    for d in dims:
        print(
            f"Running `{bench}` problem scaling study with {nthreads} threads on size: {d}x{d}",
            end="\r",
        )
        command = [
            "numactl",
            f"--physcpubind=0-{nthreads - 1}",
            f"./{exec}",
            f"-n{d}",
            f"-o{output}",
        ]
        subprocess.run(command, env=env_vars)
    remove_header_duplicates(output)


def strong_scaling(exec: str, bench: str, dims: int, output: str):
    nthreads = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    env_vars = os.environ.copy()
    env_vars["OMP_PROC_BIND"] = "spread"
    for t in nthreads:
        print(
            f"\rRunning `{bench}` strong scaling study with size {dims}x{dims} on: {t} threads",
            end="\r",
        )
        env_vars["OMP_NUM_THREADS"] = f"{t}"
        command = [
            "numactl",
            f"--physcpubind=0-{t - 1}",
            f"{exec}",
            f"-n{dims}",
            f"-o{output}",
        ]
        subprocess.run(command, env=env_vars)
    remove_header_duplicates(output)


def weak_scaling(exec: str, bench: str, output: str):
    # fmt: off
    dims     = [512, 1024, 2048, 4096, 8192]
    nthreads = [  1,    4,   16,   64,  128]
    # fmt: on

    env_vars = os.environ.copy()
    env_vars["OMP_PROC_BIND"] = "spread"
    for d, t in zip(dims, nthreads):
        print(
            f"\rRunning `{bench}` weak scaling study with: size {d}x{d}, on: {t} threads",
            end="\r",
        )
        env_vars["OMP_NUM_THREADS"] = f"{t}"
        command = [
            "numactl",
            f"--physcpubind=0-{t - 1}",
            f"{exec}",
            f"-n{d}",
            f"-o{output}",
        ]
        subprocess.run(command, env=env_vars)
    remove_header_duplicates(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiled layout benchmarks suite utility runner")
    parser.add_argument(
        "-e",
        "--executable",
        type=str,
        help="executable binary",
        metavar="EXEC",
    )
    parser.add_argument(
        "-b",
        "--bench",
        type=str,
        choices=["gemm", "stencil_2d", "stencil_3d"],
        help="benchmark name",
        metavar="BENCH",
    )
    parser.add_argument(
        "-p",
        "--problem-scaling",
        required=False,
        dest="p_nthreads_output",
        type=parse_nthreads_output,
        nargs="?",
        const=(True, 64, "./results/data/problem_scaling.csv"),
        default=(False, 64, "./results/data/problem_scaling.csv"),
        help="perform problem scaling study",
        metavar="NUM_THREADS,OUTPUT",
    )
    parser.add_argument(
        "-s",
        "--strong-scaling",
        required=False,
        dest="s_dims_output",
        type=parse_dims_output,
        nargs="?",
        const=(True, 2048, "./results/data/strong_scaling.csv"),
        default=(False, 2048, "./results/data/strong_scaling.csv"),
        help="perform strong scaling study",
        metavar="DIMS,OUTPUT",
    )
    parser.add_argument(
        "-w",
        "--weak-scaling",
        required=False,
        dest="w_output",
        type=parse_output,
        nargs="?",
        const=(True, "./results/data/weak_scaling.csv"),
        default=(False, "./results/data/weak_scaling.csv"),
        help="perform weak scaling study",
        metavar="OUTPUT",
    )

    args = parser.parse_args()
    if args.p_nthreads_output[0]:
        problem_scaling(args.executable, args.bench, args.p_nthreads_output[1], args.p_nthreads_output[2])
    if args.s_dims_output[0]:
        strong_scaling(args.executable, args.bench, args.s_dims_output[1], args.s_dims_output[2])
    if args.w_output[0]:
        weak_scaling(args.executable, args.bench, args.w_output[1])

    return 0


if __name__ == "__main__":
    main()
