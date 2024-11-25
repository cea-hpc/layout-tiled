import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="File containing the MarkDown table")
args = parser.parse_args()

dims = args.input_file[args.input_file.find('x')+1:args.input_file.find(".md")]
with open(args.input_file, "r") as f:
    lines = f.readlines()

table_lines = [line for line in lines if "|" in line]
df = pd.read_csv(io.StringIO("\n".join(table_lines)), sep="|").iloc[1:]
df.columns = df.columns.str.strip()
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
df["op/s"] = pd.to_numeric(df["op/s"])
df["err%"] = pd.to_numeric(df["err%"].str.rstrip("%"))
df["loop"] = df[f"serial {dims}x{dims}"].str.extract(r'(\w+)')[0]
df["type"] = df[f"serial {dims}x{dims}"].str.extract(r'`(\w+\w+|\w+)`')[0]
types_order = ["standard", "tiled", "block standard", "block tiled"]

grouped = df.groupby("loop")
bar_width = 0.12
x_positions = np.arange(len(types_order))

plt.figure(figsize=(10, 6))
for idx, (loop, group) in enumerate(grouped):
    group["loop"] = pd.Categorical(group["loop"], categories=types_order, ordered=True)
    offsets = x_positions + idx * bar_width
    plt.bar(
        offsets, 
        group["op/s"], 
        yerr=(group["op/s"] * group["err%"] / 100), 
        capsize=5, 
        width=bar_width, 
        label=loop
    )

plt.xticks(x_positions + bar_width * (len(grouped) - 1) / 2, types_order)
plt.xlabel("Type")
plt.ylabel("Operations per Second (op/s)")
plt.title(f"{dims}x{dims} DGEMM - Serial loop order performance comparison\nHigher is better")
plt.legend(title="Loop Order")
plt.tight_layout()
plt.show()
