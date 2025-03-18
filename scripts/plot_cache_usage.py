import argparse
from cycler import cycler
from matplotlib import font_manager as fm, pyplot as plt, rcParams
import numpy as np
import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot performance over matrix size from CSV data"
    )
    parser.add_argument("csv_path", type=str, help="Path to the input CSV")
    parser.add_argument("-o", "--output", type=str, help="Filename of the output plot")
    args = parser.parse_args()

    df = pl.read_csv(args.csv_path)
    results = df.select(
        "name",
        "elapsed",
        "error %",
        "L1D refs",
        "L1D misses",
        "LLC refs",
        "LLC misses",
        pl.col("title").str.extract(r"(\d+)x\d+").cast(pl.Int64).alias("matrix size"),
        pl.col("name").str.extract(r"(\d+)x\d+").cast(pl.Int64).alias("tile size"),
        # pl.col("L1D misses") / pl.col("L1D refs").alias("L1D misses %"),
        # pl.col("LLC misses") / pl.col("LLC refs").alias("LLC misses %"),
    )

    # Configure fonts
    font_dirs = ["/home/dossantosg/.local/share/fonts/palatino"]
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    rcParams["font.family"] = "Palatino Linotype"
    plt.rc("axes", labelsize=14)
    plt.rc("xtick", labelsize=14)
    plt.rc("ytick", labelsize=14)

    # Configure colors using tableau palette
    tableau_colors = plt.get_cmap("tab10").colors
    color_cycle = cycler(color=tableau_colors)
    plt.rc("axes", prop_cycle=color_cycle)

    # Create subplots
    unique_sizes = results["matrix size"].unique()
    nc = 5
    nr = -(-len(unique_sizes) // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(10 * nc, 6 * nr), sharex=True)
    axes = axes.flatten()

    if len(unique_sizes) == 1:
        axes = [axes]  # Ensure iterable when only one subplot

    for ax, matrix_size in zip(axes, unique_sizes):
        subset = results.filter(pl.col("matrix size") == matrix_size)

        x_labels = subset["name"]
        x = np.arange(len(x_labels))  # X-axis positions
        width = 0.4  # Width of bars

        # Plot L1D refs and L1D misses
        ax.bar(
            x - width / 2,
            subset["L1D refs"],
            width,
            label="L1D refs",
            color=tableau_colors[0],
        )
        ax.bar(
            x - width / 2,
            subset["L1D misses"],
            width,
            label="L1D misses",
            color=tableau_colors[1],
        )

        # Plot LLC refs and LLC misses
        ax.bar(
            x + width / 2,
            subset["LLC refs"],
            width,
            label="LLC refs",
            color=tableau_colors[2],
        )
        ax.bar(
            x + width / 2,
            subset["LLC misses"],
            width,
            label="LLC misses",
            color=tableau_colors[3],
        )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylabel("Cache Accesses")
        ax.set_title(f"Matrix Size {matrix_size}")

    # Add common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=14)

    plt.xlabel("Matrix Layout")
    # plt.tight_layout()
    if args.output is not None:
        plt.savefig(args.output)
    plt.show()


if __name__ == "__main__":
    main()
