# Copyright (C) 2024-2025 CEA
# Copyright (C) 2024-2025 Gabriel Dos Santos
#
# Distributed under the MIT License.
# You may not use this file except in compliance with the License.
# See https://opensource.org/license/MIT for license information.
# SPDX-License-Identifier: MIT

import argparse
from matplotlib import pyplot as plt
import numpy as np
import polars as pl

import common


def process_dataframe(df: pl.DataFrame, bench: str) -> pl.DataFrame:
    results = df.select(
        "name",
        "elapsed",
        "error %",
        pl.col("title").str.extract(common.DIMS_FMT[bench]).cast(pl.Int64).alias("problem size"),
        pl.col("name").str.extract(common.DIMS_FMT[bench]).cast(pl.Int64).alias("tile size"),
        pl.col("title")
        .str.extract(common.DIMS_FMT[bench])
        .cast(pl.List(pl.Float64))
        .list.eval(common.ALGO_COMPLEXITY[bench](pl.element().first()))
        .list.first()
        .alias("Flop"),
    )
    results = results.with_columns(((pl.col("Flop") / 1e9) / pl.col("elapsed")).alias("GFlop/s"))
    results = results.with_columns(
        (pl.col("elapsed") * pl.col("error %")).alias("elapsed err"),
        (pl.col("GFlop/s") - (pl.col("GFlop/s") / (1.0 + (pl.col("error %") / 100.0)))).alias("GFlop/s lo err"),
        ((pl.col("GFlop/s") / (1.0 - (pl.col("error %") / 100.0))) - pl.col("GFlop/s")).alias("GFlop/s up err"),
    )
    return results


def compute_speedups(results: pl.DataFrame) -> pl.DataFrame:
    # Calculate base implementation times for each matrix size
    base_times = (
        results.filter(pl.col("name") == "R naive")
        .select(["problem size", "tile size", "elapsed"])
        .rename({"elapsed": "base_elapsed"})
    )
    # Join with original results and calculate speedup
    results_with_speedup = results.join(base_times, on="problem size", how="left").with_columns(
        (pl.col("base_elapsed") / pl.col("elapsed")).alias("speedup")
    )
    # Filter tiled implementations
    tiled_results = results_with_speedup.filter(
        ~pl.col("name").str.contains("\(runtime\)") & pl.col("name").str.contains("x")
    )
    # Compute speedup stats
    speedups = (
        tiled_results.group_by("name", "tile size")
        .agg(
            pl.col("speedup").min().alias("min speedup"),
            pl.col("speedup").median().alias("med speedup"),
            pl.col("speedup").mean().alias("avg speedup"),
            pl.col("speedup").max().alias("max speedup"),
        )
        .sort("max speedup", descending=True)
    )
    return speedups


def plot_problem_scaling_performance(results: pl.DataFrame, args: argparse.ArgumentParser) -> None:
    # By default, ignore runtime runs (suffixed with `(runtime)`)
    groups = results.filter(
        ~pl.col("name").str.contains("\(runtime\)")
        # Users may add further constraints on filter
        & ~pl.col("name").str.contains("_BL")
    ).group_by("name", maintain_order=True)
    min_size = results.select(pl.col("problem size")).min().item()
    max_size = results.select(pl.col("problem size")).max().item()

    common.configure_plot_fonts(args.use_custom_fonts)
    fig, ax = plt.subplots(figsize=(12, 9))
    common.configure_plot_axes(
        ax,
        x_ticks=np.arange(min_size, max_size + 1, step=min_size),
        x_label="PROBLEM SIZE",
        y_label="PERFORMANCE (GFlop/s)",
    )
    if args.report_blas and args.hardware == "milan":
        ax.axhline(y=2080.58, color="black", linestyle="--")
        ax.annotate("BLAS DGEMM RMax", (ax.get_xlim()[1], 2080.58), ha="right", va="bottom", size=16)

    for name, group in groups:
        ax.errorbar(
            x=group["problem size"],
            y=group["GFlop/s"],
            yerr=[group["GFlop/s lo err"], group["GFlop/s up err"]],
            label=name[0],
            marker="o",
            linewidth=3,
        )
    common.configure_plot_titles(
        fig, ax, "Problem Scaling Performance", args.bench, args.hardware, args.use_custom_fonts
    )

    if args.output is not None:
        fig.savefig(args.output)


def main() -> None:
    parser = common.default_parser("Plot problem scaling performance from CSV data")
    parser.add_argument(
        "--report-blas",
        action="store_true",
        help="Report BLAS performance for GEMM",
    )
    args = parser.parse_args()
    if args.report_blas and args.bench != "gemm":
        parser.error("--report-blas options requires 'gemm' positional argument")

    df = common.read_csv_into_dataframe(args.csv)
    results = process_dataframe(df, args.bench)

    speedups = compute_speedups(results)
    print(speedups)

    plot_problem_scaling_performance(results, args)
    plt.show()


if __name__ == "__main__":
    main()
