#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
from matplotlib import pyplot as plt
import numpy as np
import polars as pl

import common

EVENTS = [
    "inst",
    "mips",
    "bra",
    "l1d",
    "llc",
]


def process_dataframe(df: pl.DataFrame, size: int, event: str) -> pl.DataFrame:
    res1 = df.select(
        "title",
        "name",
        "elapsed",
    )

    if event == "inst":
        res2 = df.select("instructions").filter(pl.col("title").str.contains(f"{size}x{size}"))
    elif event == "mips":
        res2 = df.select((pl.col("instructions") / pl.col("elapsed") / 1_000_000).alias("mips"))
    else:
        res2 = df.select(event.upper() + " refs", event.upper() + " misses")

    results = pl.concat([res1, res2], how="horizontal").filter(pl.col("title").str.contains(f"{size}x{size}"))
    if results.is_empty():
        raise ValueError(f"No record founds: problem size `{size}` is invalid")
    print(results)

    return results


def plot_event(results: pl.DataFrame, args: argparse.Namespace) -> None:
    # By default, ignore runtime runs (suffixed with `(runtime)`)
    results = results.filter(
        ~pl.col("name").str.contains("\(runtime\)")
        # Users may add further constraints on filter
        # & ~pl.col("name").str.contains("_BL")
    )
    names = results["name"].to_list()

    common.configure_plot_fonts(args.use_custom_fonts)
    fig, ax = plt.subplots(figsize=(12, 9))
    if args.event == "inst":
        instructions = results["instructions"].sort().to_list()
        ax.barh(names, instructions, color="tab:blue")
        ax.set_xlabel("Instructions")
    elif args.event == "mips":
        mips = results["mips"].sort().to_list()
        ax.barh(names, mips, color="tab:blue")
        ax.set_xlabel("MIPS")
    else:
        refs_col = args.event.upper() + " refs"
        misses_col = args.event.upper() + " misses"
        results = results.with_columns((pl.col(misses_col) / pl.col(refs_col)).alias("miss rate")).sort("miss rate")
        refs = results[refs_col].to_list()
        misses = results[misses_col].to_list()
        ax.barh(names, refs, color="tab:blue", label="References")
        ax.barh(names, misses, color="tab:red", label="Misses")
        ax.set_xlabel(f"{args.event.upper()} COUNT")
        ax.legend()

    ax.set_ylabel("CONFIGURATION")
    common.configure_plot_titles(fig, ax, "", args.bench, args.hardware, args.use_custom_fonts)

    if args.output is not None:
        fig.savefig(args.output)


def main() -> None:
    parser = common.default_parser("Plot performance events from CSV data")
    parser.add_argument("--event", type=str, required=True, choices=EVENTS, help="Perf event to plot")
    parser.add_argument("--size", type=int, required=True, help="Problem size to plot")
    print(parser)
    args = parser.parse_args()
    print(args)

    df = common.read_csv_into_dataframe(args.csv)
    results = process_dataframe(df, args.size, args.event)

    plot_event(results, args)
    plt.show()


if __name__ == "__main__":
    main()
