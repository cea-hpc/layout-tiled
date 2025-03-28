# Copyright (C) 2024-2025 CEA
# Copyright (C) 2024-2025 Gabriel Dos Santos
#
# Distributed under the MIT License.
# You may not use this file except in compliance with the License.
# See https://opensource.org/license/MIT for license information.
# SPDX-License-Identifier: MIT

import argparse
from cycler import cycler
from matplotlib import font_manager as fm, pyplot as plt, rcParams
import numpy as np
import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot instruction count from CSV data")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV")
    parser.add_argument("-o", "--output", type=str, help="Filename of the output plot")
    args = parser.parse_args()

    df = pl.read_csv(args.csv_path)
    df = df.with_columns(
        pl.col("title").str.extract(r"(\d+)x\d+").cast(pl.Int64).alias("matrix_size"),
        pl.col("name").str.extract(r"(\d+)x\d+").cast(pl.Int64).alias("tile size"),
    )
    df = df.filter(~pl.col("name").str.contains("OMP")).sort(
        "tile size", maintain_order=True
    )

    # Get unique matrix sizes and sort them
    matrix_sizes = (
        df.select("matrix_size")
        .unique()
        .sort("matrix_size", maintain_order=True)["matrix_size"]
        .to_list()
    )

    # Get unique layouts excluding "right" (which will be our baseline)
    layouts = [layout for layout in df["name"].unique() if layout != "right"]

    # Calculate speedups relative to "right" layout for each matrix size
    speedup_data = []

    for size in matrix_sizes:
        size_data = df.filter(pl.col("title").str.contains(f"{size}x{size}"))

        # Get baseline (right layout) elapsed time
        right_time = size_data.filter(pl.col("name") == "right")["elapsed"].item()

        # Calculate speedup for each layout
        for layout in layouts:
            layout_time = size_data.filter(pl.col("name") == layout)["elapsed"].item()
            speedup = right_time / layout_time
            speedup_data.append(
                {"matrix_size": size, "layout": layout, "speedup": speedup}
            )

    # Configure fonts
    font_dirs = ["/home/dossantosg/.local/share/fonts/palatino"]
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    rcParams["font.family"] = "Palatino Linotype"
    plt.rc("axes", labelsize=24)
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)

    # Set up the plot
    default_cycler = cycler(
        color=[
            "#76b7b2",
            "#59a14f",
            "#f28e2b",
            "#e15759",
            "#edc848",
            "#b07aa1",
            "#4e79a7",
        ]
    )
    plt.figure(figsize=(10, 6))
    plt.rc("axes", prop_cycle=default_cycler)

    # Number of layouts and positioning
    n_layouts = len(layouts)
    width = 0.75 / n_layouts  # Width of each bar

    # Create bars for each layout
    for i, layout in enumerate(layouts):
        # Extract speedup values for this layout
        layout_data = pl.DataFrame(speedup_data).filter(pl.col("layout") == layout)
        layout_speedups = layout_data["speedup"].to_list()

        # Plot bars for this layout
        positions = (
            np.arange(len(matrix_sizes)) + i * width - (n_layouts - 1) * width / 2
        )
        bars = plt.bar(positions, layout_speedups, width, label=layout, alpha=0.9)

        # Add value labels on top of bars
        for bar, speedup in zip(bars, layout_speedups):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{speedup:.1f}x",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Customize plot
    plt.axhline(y=1.0, color="r", linestyle="-", alpha=0.3)  # Baseline reference line
    plt.xlabel("Matrix size")
    plt.ylabel("Speedup over layout right")
    plt.xticks(np.arange(len(matrix_sizes)), [f"{size}" for size in matrix_sizes])
    plt.legend(title="Layouts", title_fontsize=16, fontsize=16, ncol=2)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if args.output is not None:
        plt.savefig(args.output)
    plt.show()


if __name__ == "__main__":
    main()
