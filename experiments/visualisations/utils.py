import pickle as pk
from typing import Dict, List, Literal

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm

# Font Management: Load system fonts for consistent figure rendering
# This ensures Roboto and other fonts are available for matplotlib
for font in fm.findSystemFonts("fonts", "ttf"):
    fm.fontManager.addfont(font)

# Figure Dimensions: All figures are designed at 2× size for high-DPI displays
# Based on LaTeX document column and page widths converted from points to inches
COLUMNWIDTH = 237.13594 / 72 * 2  # Column width in inches (scaled 2×)
FULLWIDTH = 496.85625 / 72 * 2  # Full page width in inches (scaled 2×)


# =============================================================================
# VISUALIZATION UTILITY FUNCTIONS
# =============================================================================


def _fix_venn3_fonts(venn3_plot, set_size=14, subset_size=16):
    """Fix font sizes for Venn diagram labels.

    Args:
        venn3_plot: Matplotlib venn3 plot object
        set_size (int): Font size for set labels (default: 14)
        subset_size (int): Font size for subset labels (default: 16)
    """
    for text in venn3_plot.set_labels:
        text.set_fontsize(set_size)
    for text in venn3_plot.subset_labels:
        text.set_fontsize(subset_size)


def _setup_style():
    """Set up consistent plotting style for all visualizations.

    Configures seaborn and matplotlib with paper-style theme, Roboto font,
    and defines consistent color palette and markers for plots.

    Returns:
        tuple: (colors, markers) - seaborn color palette and marker list
    """
    sb.reset_defaults()
    sb.set_theme(style="whitegrid")
    sb.set_context("paper", font_scale=1.5)
    plt.rcParams["font.family"] = "Roboto"

    colors = sb.color_palette("deep")
    markers = ["o", "s", "d", "^", "v", "<", ">", "P", "x", "*", "h", "H"]
    return colors, markers


def normalize(array):
    """Normalize array values to [0, 1] range using min-max normalization.

    Args:
        array (np.ndarray): Input array to normalize

    Returns:
        np.ndarray: Normalized array with values in [0, 1] range
    """
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def radar_plot(ax, data, categories, labels):
    """Create a radar (spider) plot for multi-dimensional data comparison.

    Args:
        ax (matplotlib.axes.Axes): Polar axes object for plotting
        data (np.ndarray): Data array, shape (n_series, n_categories) or (n_categories,)
        categories (list): List of category names for each axis
        labels (list): List of labels for each data series

    Note:
        Data is expected to be normalized to [0, 1] range for best visualization.
        The plot automatically closes the radar by connecting the last point to the first.
    """
    colors, markers = _setup_style()

    # Ensure data is 2D (n_series, n_categories)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    N = data.shape[1]

    # Create angles for each axis, closing the radar
    # endpoint=False ensures equal spacing without duplicating 0/2π
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Close the polygon

    # Configure polar plot appearance
    ax.set_theta_offset(np.pi / 2)  # Start from top
    ax.set_theta_direction(-1)  # Clockwise direction
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)

    # Set radial limits and ticks
    ax.set_ylim(0, 1)
    ax.set_rticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels([0, "", 0.5, "", 1])  # Only show 0, 0.5, 1 labels
    ax.spines["polar"].set_visible(False)

    # Plot each data series
    for _data, color, label in zip(data, colors, labels):
        # Close the radar by appending the first value
        _data = np.concatenate((_data, [_data[0]]))
        ax.fill(angles, _data, alpha=0.25, color=color)  # Filled area
        ax.plot(angles, _data, color=color, label=label)  # Line plot
