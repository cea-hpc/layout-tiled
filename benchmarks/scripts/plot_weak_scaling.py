#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
from matplotlib import pyplot as plt
import numpy as np
import polars as pl

import common


def process_dataframe(df: pl.DataFrame, bench: str) -> pl.DataFrame:
    results = df.with_columns(pl.col("title").str.extract(r"(\d+)C").cast(pl.Int64).alias("cores"))
    baseline = results.filter(pl.col("cores") == 1).select("elapsed").min().item()
    results = results.with_columns((baseline / pl.col("elapsed")).alias("efficiency"))

    return results


def plot_weak_scaling_efficiency(results: pl.DataFrame, args: argparse.ArgumentParser) -> None:
    # By default, ignore runtime runs (suffixed with `(runtime)`)
    filtered = results.filter(
        ~pl.col("name").str.contains("(runtime)")
        # Users may add further constraints on filter
        # & ~pl.col("name").str.contains("_BL")
    )
    groups = filtered.group_by("name", maintain_order=True)
    min_cores = filtered.select(pl.col("cores")).min().item()
    max_cores = filtered.select(pl.col("cores")).max().item()
    cores = np.array(sorted([x for x in filtered["cores"] if (x & (x - 1)) == 0]))

    common.configure_plot_fonts(args.use_custom_fonts)
    fig, ax = plt.subplots(figsize=(12, 9))
    common.configure_plot_axes(
        ax,
        x_label="NUMBER OF CORES",
        y_label="EFFICIENCY",
        x_lim=(min_cores * 0.9, max_cores * 1.1),
        y_lim=(-0.05, 1.05),
        x_scale="log"
    )

    # Ideal efficiency line (perfect speedup)
    ax.plot(cores, np.ones_like(cores), "k--")
    ax.annotate("ideal", (max_cores, 1), ha="right", va="bottom", size=16)

    for name, group in groups:
        ax.errorbar(
            x=group["cores"],
            y=group["efficiency"],
            label=name[0],
            marker="o",
            linewidth=3,
        )
    common.configure_plot_titles(fig, ax, "Weak Scaling Efficiency", args.bench, args.hardware, args.use_custom_fonts)

    if args.output is not None:
        fig.savefig(args.output)
    plt.show()


def main() -> None:
    parser = common.default_parser("Plot weak scaling speedup from CSV data")
    args = parser.parse_args()

    df = common.read_csv_into_dataframe(args.csv)
    results = process_dataframe(df, args.bench)

    plot_weak_scaling_efficiency(results, args)


if __name__ == "__main__":
    main()
