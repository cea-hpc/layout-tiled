import argparse
import os
import subprocess


def parse_num_threads_and_output(args: tuple):
    try:
        num_threads, output = args.split(",", 1)
        return True, int(num_threads), output
    except ValueError:
        raise argparse.ArgumentTypeError("argument must be in the format: <int>,<str>")


def parse_dims_and_output(args: tuple):
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


def perf_comp(num_threads: int, output: str):
    dims = [512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096]

    env_vars = os.environ.copy()
    env_vars["OMP_PROC_BIND"] = "spread"
    env_vars["OMP_NUM_THREADS"] = f"{num_threads}"
    for dim in dims:
        command = [
            "numactl",
            f"--physcpubind=0-{num_threads - 1 if num_threads <= 64 else f'{int(num_threads / 2 - 1)},64-{int(64 + num_threads / 2 - 1)}'}",  # balance threads between sockets when above 64
            "./build/benchmarks/layout_tiled.benchmarks.gemm",
            f"-x{dim}",
            f"-y{dim}",
            f"-o{output}",
        ]
        subprocess.run(command, env=env_vars)


def strong_scaling(dims: int, output: str):
    threads = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]

    env_vars = os.environ.copy()
    env_vars["OMP_PROC_BIND"] = "spread"
    for th in threads:
        env_vars["OMP_NUM_THREADS"] = f"{th}"
        command = [
            "numactl",
            f"--physcpubind=0-{th - 1 if th <= 64 else f'{int(th / 2 - 1)},64-{int(64 + th / 2 - 1)}'}",  # balance threads between sockets when above 64
            "./build/benchmarks/layout_tiled.benchmarks.gemm",
            f"-x{dims}",
            f"-y{dims}",
            f"-o{output}",
        ]
        subprocess.run(command, env=env_vars)


def weak_scaling(output: str):
    dims_x = [512, 512, 1024, 1024, 1536, 2048, 2048, 2048, 3072, 4096, 4096, 8192]
    dims_y = [512, 1024, 1024, 2048, 2048, 2048, 3072, 4096, 4096, 4096, 8192, 8192]
    threads = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]

    env_vars = os.environ.copy()
    env_vars["OMP_PROC_BIND"] = "spread"
    for dx, dy, th in zip(dims_x, dims_y, threads):
        env_vars["OMP_NUM_THREADS"] = f"{th}"
        command = [
            "numactl",
            f"--physcpubind=0-{th - 1 if th <= 64 else f'{int(th / 2 - 1)},64-{int(64 + th / 2 - 1)}'}",  # balance threads between sockets when above 64
            "./build/benchmarks/layout_tiled.benchmarks.gemm",
            f"-x{dx}",
            f"-y{dy}",
            f"-o{output}",
        ]
        subprocess.run(command, env=env_vars)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tiled layout benchmarks on a naive GEMM"
    )
    parser.add_argument(
        "-c",
        "--perf-comparison",
        required=False,
        dest="pc_num_threads_and_output",
        type=parse_num_threads_and_output,
        nargs="?",
        const=(True, 64, "./results/gemm/data/perf_comp_1s64c.csv"),
        default=(False, 64, "./results/gemm/data/perf_comp_1s64c.csv"),
        help="Compare layout performance on naive GEMM",
        metavar="NUM_THREADS,OUTPUT",
    )
    parser.add_argument(
        "-s",
        "--strong-scaling",
        required=False,
        dest="ss_dims_and_output",
        type=parse_dims_and_output,
        nargs="?",
        const=(True, 2048, "./results/gemm/data/strong_scaling.csv"),
        default=(False, 2048, "./results/gemm/data/strong_scaling.csv"),
        help="Run strong scaling evaluation",
        metavar="DIMS,OUTPUT",
    )
    parser.add_argument(
        "-w",
        "--weak-scaling",
        required=False,
        dest="ws_output",
        type=parse_output,
        nargs="?",
        const=(True, "./results/gemm/data/weak_scaling.csv"),
        default=(False, "./results/gemm/data/weak_scaling.csv"),
        help="Run weak scaling evaluation",
        metavar="OUTPUT",
    )

    args = parser.parse_args()
    if args.pc_num_threads_and_output[0]:
        perf_comp(args.pc_num_threads_and_output[1], args.pc_num_threads_and_output[2])
    if args.ss_dims_and_output[0]:
        strong_scaling(args.ss_dims_and_output[1], args.ss_dims_and_output[2])
    if args.ws_output[0]:
        weak_scaling(args.ws_output[1])

    return 0


if __name__ == "__main__":
    main()
