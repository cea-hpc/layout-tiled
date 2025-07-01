# Copyright (C) 2024-2025 CEA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

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
    results = df.select(
        "name",
        "elapsed",
        "instructions",
        pl.col("title").str.extract(r"(\d+)x\d+").cast(pl.Int64).alias("matrix size"),
        pl.col("name").str.extract(r"(\d+)x\d+").cast(pl.Int64).alias("tile size"),
    )
    groups = results.sort("tile size").group_by("name", maintain_order=True)

    # Configure fonts
    font_dirs = ["/home/dossantosg/.local/share/fonts/palatino"]
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    rcParams["font.family"] = "Palatino Linotype"
    plt.rc("axes", labelsize=20)
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)

    plt.figure(figsize=(10, 6))
    default_cycler = cycler(
        color=[
            "#4e79a7",
            "#76b7b2",
            "#f28e2b",
            "#e15759",
            "#59a14f",
            "#edc848",
            "#b07aa1",
        ]
    ) + cycler(
        linestyle=[
            "-",
            "-.",
            "--",
            ":",
            "--",
            ":",
            "--",
        ]
    )
    for name, group in groups:
        plt.rc("axes", prop_cycle=default_cycler)
        plt.errorbar(
            x=group["matrix size"],
            y=group["instructions"] / 1e9,
            label=name[0],
            marker="x",
        )
    plt.xlabel("Matrix size")
    plt.ylabel("Executed instructions (billions)")
    plt.xticks(np.arange(512, 2100, step=512))
    plt.legend(title="Layouts", title_fontsize=20, fontsize=20)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tight_layout()

    if args.output is not None:
        plt.savefig(args.output)
    plt.show()


if __name__ == "__main__":
    main()
