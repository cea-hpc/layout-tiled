import argparse
from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np
from os import path
import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot weak scaling performance from CSV data"
    )
    parser.add_argument("csv_path", type=str, help="Path to the input CSV")

    args = parser.parse_args()

    df = pl.read_csv(args.csv_path)

    df = df.with_columns(
        pl.col("title").str.extract(r"(\d+)C").cast(pl.Int64).alias("cores")
    )
    baseline = df.filter(pl.col("cores") == 1).select(["name", "elapsed"])

    df = df.join(baseline, on="name", suffix="_base")
    df = df.with_columns(
        (pl.col("elapsed_base") / (pl.col("elapsed") * pl.col("cores"))).alias(
            "efficiency"
        )
    )
    groups = df.group_by("name", maintain_order=True)

    plt.figure(figsize=(10, 4))
    default_cycler = cycler(
        color=["tab:blue", "tab:orange", "tab:green", "tab:red"]
    ) + cycler(linestyle=["-", "--", ":", "-."])
    for name, group in groups:
        plt.rc("axes", prop_cycle=default_cycler)
        plt.errorbar(
            x=group["cores"],
            y=group["efficiency"],
            label=name[0],
            marker="x",
        )

    # Ideal weak scaling line (constant runtime)
    cores = np.array(sorted(df["cores"].unique().to_list()))
    plt.plot(cores, np.ones_like(cores), "k--", label="Ideal Scaling")

    plt.xlabel("Number of Cores")
    plt.ylabel("Efficiency")
    plt.xticks(cores, labels=[str(c) for c in cores])
    plt.yticks(
        np.arange(0, 1.1, 0.2), labels=[f"{v:.1f}" for v in np.arange(0, 1.1, 0.2)]
    )
    # plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.title("Weak Scaling Efficiency")

    plt.tight_layout()
    save_path, _ = path.splitext(
        path.join("./results/plots/", path.basename(args.csv_path))
    )
    plt.savefig(f"{save_path}.pdf")
    plt.show()


if __name__ == "__main__":
    main()
