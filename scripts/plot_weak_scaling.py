import argparse
from cycler import cycler
from matplotlib import font_manager as fm, pyplot as plt, rcParams
import numpy as np
import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot weak scaling effiency from CSV data"
    )
    parser.add_argument("csv_path", type=str, help="Path to the input CSV")
    parser.add_argument("-o", "--output", type=str, help="Filename of the output plot")
    args = parser.parse_args()

    df = pl.read_csv(args.csv_path)
    df = df.with_columns(
        pl.col("title").str.extract(r"(\d+)C").cast(pl.Int64).alias("cores")
    )
    baseline = df.filter(pl.col("cores") == 1).select(["name", "elapsed"])

    df = df.join(baseline, on="name", suffix="_base")
    df = df.with_columns(
        (pl.col("elapsed_base") / pl.col("elapsed")).alias("efficiency")
    )
    groups = df.group_by("name", maintain_order=True)

    # Configure fonts
    font_dirs = ["/home/dossantosg/.local/share/fonts/palatino"]
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    rcParams["font.family"] = "Palatino Linotype"
    plt.rc("axes", labelsize=24)
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)

    plt.figure(figsize=(10, 6))
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
    cores = np.array(sorted([x for x in df["cores"] if (x & (x - 1)) == 0]))
    plt.plot(cores, np.ones_like(cores), "k--", label="ideal")

    plt.xlabel("Number of cores")
    plt.ylabel("Efficiency")
    plt.xscale("log", base=2)
    plt.xticks(cores, cores)
    plt.legend(title="Layouts", title_fontsize=20, fontsize=20)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.tight_layout()

    if args.output is not None:
        plt.savefig(args.output)
    plt.show()


if __name__ == "__main__":
    main()
