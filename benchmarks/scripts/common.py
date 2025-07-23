# Copyright (C) 2024-2025 CEA
# Copyright (C) 2024-2025 Gabriel Dos Santos
#
# Distributed under the MIT License.
# You may not use this file except in compliance with the License.
# See https://opensource.org/license/MIT for license information.
# SPDX-License-Identifier: MIT

import argparse
import logging
from math import log2
from matplotlib import font_manager as fm, pyplot as plt, rcParams
import numpy as np
import os
import polars as pl
import sys
from typing import Tuple

ALGO_COMPLEXITY = {
    "gemm": lambda n: 2 * n * n * n,
    "stencil_2d": lambda n: 4 * n * n,
    "stencil_3d": lambda n: 6 * n * n * n,
}

DIMS_FMT = {
    "gemm": r"(\d+)x\d+",
    "stencil_2d": r"(\d+)x\d+",
    "stencil_3d": r"(\d+)x(\d+)x\d+",
}

HARDWARE = {
    "default": "Unknown hardware",
    "laptop": "Intel i5-1135G7",
    "workstation": "Intel Ultra 5 135H",
    "milan": "2x AMD EPYC 7763",
}

COLOR_PALETTE = [plt.cm.tab10.colors[7]] + list(plt.cm.tab20.colors)


def default_parser(desc: str) -> argparse.ArgumentParser:
    """
    Create a default argument parser.

    Args:
        desc (str): Description of the parser.
    """
    parser = argparse.ArgumentParser(description=f"{desc}")
    parser.add_argument("--csv", type=str, help="Path to the input CSV")
    parser.add_argument(
        "--bench",
        type=str,
        help="Benchmark name",
        choices=["gemm", "stencil_2d", "stencil_3d"],
    )
    parser.add_argument(
        "--hardware",
        type=str,
        help="Hardware configuration",
        choices=["default", "laptop", "workstation", "milan"],
        default="default",
    )
    parser.add_argument("-o", "--output", type=str, help="File name of the output plot")
    parser.add_argument(
        "--use-custom-fonts",
        action="store_true",
        help="Enable using custom fonts in the output plot",
    )

    return parser


def configure_logging() -> None:
    """
    Configure logging format.
    """
    logging.basicConfig(format="[{asctime}] [{levelname}] {message}", style="{", datefmt="%Y-%m-%d %H:%M:%S")


def configure_plot_fonts(use_custom_fonts: bool) -> None:
    """
    Configure plot aesthetics and fonts.

    Args:
        use_custom_fonts (bool): Whether to use custom fonts.
    """
    # Use context manager for font configuration
    if use_custom_fonts:
        font_dirs = [
            f"{os.path.expanduser('~')}/.local/share/fonts/geist-mono",
            f"{os.path.expanduser('~')}/.local/share/fonts/geist-sans",
        ]
        font_files = fm.findSystemFonts(fontpaths=font_dirs)
        for font_file in font_files:
            fm.fontManager.addfont(font_file)
        rcParams["font.family"] = "Geist Mono"


def configure_plot_axes(
    ax: plt.Axes,
    x_label: str,
    y_label: str,
    x_lim: Tuple[int, int],
    y_lim: Tuple[int, int] = None,
    x_scale: str = "linear",
) -> None:
    """
    Configure axes properties.

    Args:
        ax (Axes): The axes object to configure.
        x_range (int): Tuple of minimum and maximum problem size for x-ticks.
        x_label (str): X-axis label.
        y_label (str): Y-axis label.
    """
    # Set labels and font sizes
    ax.set_xlabel(f"{x_label}", fontsize=24)
    ax.set_ylabel(f"{y_label}", fontsize=24)

    # Configure ticks
    if x_scale != "linear":
        ax.set_xscale(x_scale, base=2)

    ticks = np.array([2**i for i in range(int(log2(x_lim[0])), int(log2(x_lim[1])) + 1) if x_lim[0] <= 2**i <= x_lim[1]])
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    ax.tick_params(axis="both", labelsize=16)

    # Set color cycle and grid
    ax.set_prop_cycle(color=COLOR_PALETTE)
    ax.grid(True, which="major", ls="-", alpha=0.33)


def configure_plot_titles(
    fig: plt.Figure, ax: plt.Axes, title: str, bench: str, hw: str, use_custom_fonts: bool
) -> None:
    """
    Set figure and axes titles.

    Args:
        fig (Figure): Figure object to configure.
        ax (Axes): Axes object to configure.
        title (str): Figure title.
        hw (str): Hardware used.
        bench (str): Benchmark name.
        use_custom_fonts (bool): Whether to use custom fonts.
    """
    title_font = "Geist" if use_custom_fonts else "sans-serif"

    # Set main title
    fig.suptitle(
        f"{title}",
        family=title_font,
        size=28,
        weight="bold",
        x=0.1,
        ha="left",
    )

    # Set subtitle
    ax.set_title(
        f"{HARDWARE[hw]}, {bench.upper()}\nHigher is better",
        loc="left",
        size=20,
        pad=20,
    )

    # Add legend after all plots
    ax.legend(fontsize=20)

    # Set layout
    fig.tight_layout()


def read_csv_into_dataframe(csv: str) -> pl.DataFrame:
    """
    Read CSV file with explicit error handling into a Polars dataframe.

    Args:
        csv (str): Path of the input CSV.
    """
    configure_logging()

    try:
        df = pl.read_csv(csv)
    except FileNotFoundError:
        logging.error(f"CSV file not found at `{csv}`")
        sys.exit(-1)
    except Exception as e:
        logging.error(f"failed to read CSV: {e}")
        sys.exit(-1)

    return df
