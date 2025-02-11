import argparse
from cycler import cycler
import matplotlib.pyplot as plt
from os import path
import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot performance over matrix size from CSV data"
    )
    parser.add_argument("csv_path", type=str, help="Path to the input CSV")

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
        .alias("gflop"),
    )
    results = results.with_columns(
        (pl.col("gflop") / pl.col("elapsed")).alias("gflops")
    )
    results = results.with_columns(
        (pl.col("error %") * pl.col("gflops")).alias("errors")
    )
    groups = results.sort("tile size").group_by("name", maintain_order=True)

    plt.figure(figsize=(10, 4))
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

    plt.xlabel("Matrix Size (NxN)")
    plt.ylabel("GFLOP/s")
    plt.title("Naive GEMM performance\nHigher is Better")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.4)

    plt.tight_layout()
    save_path, _ = path.splitext(
        path.join("./results/plots/", path.basename(args.csv_path))
    )
    plt.savefig(f"{save_path}.pdf")
    plt.show()


if __name__ == "__main__":
    main()
