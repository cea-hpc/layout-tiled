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
        pl.col("title").str.extract(r"(\d+)x\d+").cast(pl.Int64).alias("matrix size"),
        pl.col("name").str.extract(r"(\d+)x\d+").cast(pl.Int64).alias("tile size"),
        pl.col("title")
        .str.extract(r"(\d+)x\d+")
        .cast(pl.List(pl.Float64))
        .list.eval(
            pl.element().first() * pl.element().last() * pl.element().first() / 1e9
        )
        .list.first()
        .alias("gflop"),
    )
    results = results.with_columns(
        (pl.col("gflop") / pl.col("elapsed")).alias("gflops")
    )
    results = results.with_columns(
        (pl.col("error %") * pl.col("gflops")).alias("errors")
    )
    groups = results.sort("tile size").group_by("name", maintain_order=True)
    max_perf = results.select(pl.col("gflops")).max().item()

    # Calculate base implementation times for each matrix size
    base_times = (
        results.filter(pl.col("name") == "right")
        .select(["matrix size", "tile size", "elapsed"])
        .rename({"elapsed": "base_elapsed"})
    )
    # Join with original results and calculate speedup
    results_with_speedup = results.join(
        base_times, on="matrix size", how="left"
    ).with_columns((pl.col("base_elapsed") / pl.col("elapsed")).alias("speedup"))
    # Filter to show only tiled implementations
    tiled_results = results_with_speedup.filter(pl.col("name").str.contains("tiled"))
    avg_speedups = (
        tiled_results.group_by("tile size")
        .agg(
            pl.col("speedup").max().alias("max_speedup"),
            pl.col("speedup").mean().alias("avg_speedup"),
            pl.col("speedup").std().alias("speedup_std"),
        )
        .sort("tile size")
    )
    print(avg_speedups)

    # Configure fonts
    font_dirs = ["/home/dossantosg/.local/share/fonts/palatino"]
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    rcParams["font.family"] = "Palatino Linotype"
    plt.rc("axes", labelsize=24)
    plt.rc("xtick", labelsize=24)
    plt.rc("ytick", labelsize=24)

    plt.figure(figsize=(10, 6))
    default_cycler = cycler(
        color=["tab:blue", "tab:orange", "tab:green", "tab:red"]
    ) + cycler(linestyle=["-", "--", ":", "-."])
    for name, group in groups:
        plt.rc("axes", prop_cycle=default_cycler)
        plt.errorbar(
            x=group["matrix size"],
            y=group["gflops"],
            yerr=group["errors"],
            label=name[0],
            marker="x",
        )
    plt.xlabel("Matrix size")
    plt.ylabel("Average Gflop/s")
    plt.xticks(np.arange(512, 4200, step=512))
    plt.yticks(np.arange(0, max_perf + 5, step=10))
    plt.legend(title="Layouts", title_fontsize=20, fontsize=20)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tight_layout()

    if args.output is not None:
        plt.savefig(args.output)
    plt.show()


if __name__ == "__main__":
    main()
