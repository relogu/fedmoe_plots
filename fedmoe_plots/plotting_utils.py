"""Utilities for plotting FedMoE project figures."""

import colorsys
import logging
import math
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import matplotlib.style as mplstyle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

log = logging.getLogger(__name__)


def create_continuous_colormap_for_colors(
    colors: list[str],
    name: str = "custom_cmap",
) -> LinearSegmentedColormap:
    """Create a continuous colormap from a list of color strings.

    Parameters
    ----------
    colors : list[str]
        A list of color strings (e.g., ["red", "blue", "green"] or
        ["#FF0000", "#0000FF", "#00FF00"])
    name : str
        Name for the colormap

    Returns
    -------
    LinearSegmentedColormap
        A continuous colormap created from the input colors.

    """

    def _get_hsv(hexrgb: str) -> tuple[float, float, float]:
        # TODO(Lorenzo): Transform to HSV only if needed
        hexrgb = hexrgb.lstrip("#")  # Remove any leading '#'
        # Convert hex to RGB
        r, g, b = (int(hexrgb[i : i + 2], 16) / 255.0 for i in range(0, 6, 2))
        # Convert RGB to HSV
        return colorsys.rgb_to_hsv(r, g, b)

    # Sort the list by HSV value
    colors.sort(key=_get_hsv)

    return LinearSegmentedColormap.from_list(name, colors)


def configure_logging_for_jupyter(
    level: int = logging.INFO,
    *,
    force: bool = True,
) -> None:
    """Configure logging to display properly in Jupyter notebooks.

    By default, Python's logging module is configured with a WARNING level,
    which means INFO and DEBUG messages won't be displayed in Jupyter notebook
    cell outputs. This function configures logging to display messages at the
    specified level.

    Parameters
    ----------
    level : int, optional
        The logging level. Use logging.DEBUG, logging.INFO, logging.WARNING,
        or logging.ERROR. Default is logging.INFO.
    force : bool, optional
        Whether to force reconfiguration of existing handlers. Default is True.

    Examples
    --------
    >>> # Configure to show INFO and above
    >>> configure_logging_for_jupyter(logging.INFO)
    >>>
    >>> # Configure to show DEBUG and above
    >>> configure_logging_for_jupyter(logging.DEBUG)

    """
    logging.basicConfig(
        level=level,
        format="%(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=force,
    )


def _select_style(
    *,
    custom_fonts: bool = True,
    use_inverted_style: bool = False,
    backend: str = "pgf",
) -> None:
    """Select the matplotlib style for FedMoE project figures.

    Parameters
    ----------
    custom_fonts : bool
        Whether to use custom fonts in the matplotlib style. Default is True.
    use_inverted_style : bool
        Whether to use the inverted style for the matplotlib figures. Default is False.
    backend : str, optional
        The matplotlib backend to use. Default is "pgf".

    Raises
    ------
    FileNotFoundError
        If the branding styles directory cannot be found.

    """
    # Get the directory where to find the styles
    # Try different methods to find the branding styles directory
    possible_paths = [
        Path(__file__).parent / "notebooks",  # Local notebooks directory
        Path.cwd() / "fedmoe_plots" / "notebooks",  # Current working directory
    ]

    styles_path = None
    for path in possible_paths:
        if path.exists():
            styles_path = path
            break

    if styles_path is None:
        msg = f"Could not find branding styles directory. Tried: {possible_paths}"
        raise FileNotFoundError(msg)
    # Get style paths
    camlsys_conf_style_path = (
        styles_path / "camlsys_matplotlib_style_conference.mplstyle"
    )
    camlsys_style_path = styles_path / "camlsys_matplotlib_style.mplstyle"
    camlsys_inv_style_path = styles_path / "camlsys_matplotlib_style_inv.mplstyle"

    # Load style sheet
    mpl.use(backend)
    if custom_fonts:
        if use_inverted_style:
            plt.style.use(camlsys_inv_style_path)
        else:
            plt.style.use(camlsys_style_path)
    else:
        plt.style.use(camlsys_conf_style_path)


def run_matplotlib_preamble(
    *,
    custom_fonts: bool = True,
    use_inverted_style: bool = False,
    backend: str = "pgf",
) -> tuple[list[str], list[tuple], list[str]]:
    """Run the matplotlib preamble for FedMoE project figures.

    Parameters
    ----------
    custom_fonts : bool
        Whether to use custom fonts in the matplotlib style. Default is True.
    use_inverted_style : bool
        Whether to use the inverted style for the matplotlib figures. Default is False.
    backend : str, optional
        The matplotlib backend to use. Default is "pgf".

    Returns
    -------
    tuple
        A tuple containing the color palette, line styles, and patterns used in the
        figures.

    Notes
    -----
    This function includes logging messages that may not appear in Jupyter notebooks
    by default. To see these messages, configure logging first:

    >>> import logging
    >>> configure_logging_for_jupyter(logging.INFO)

    """
    _select_style(
        custom_fonts=custom_fonts,
        use_inverted_style=use_inverted_style,
        backend=backend,
    )

    # Fix font configuration to avoid LaTeX warnings
    # Provide fallback fonts that are LaTeX-compatible
    if mpl.rcParams.get("text.usetex", False):
        if custom_fonts:
            # When using custom fonts, preserve them and add LaTeX fallbacks
            current_serif_fonts = mpl.rcParams.get("font.serif", [])
            if isinstance(current_serif_fonts, str):
                current_serif_fonts = [current_serif_fonts]

            # Add LaTeX-compatible fallbacks only if not already there
            latex_fallbacks = [
                "Computer Modern Roman",  # Default LaTeX font
                "Times",
                "Times New Roman",
                "DejaVu Serif",
                "serif",  # Generic fallback
            ]

            # Combine custom fonts with fallbacks, removing duplicates
            combined_fonts = []
            for font in current_serif_fonts + latex_fallbacks:
                if font not in combined_fonts:
                    combined_fonts.append(font)

            mpl.rcParams["font.serif"] = combined_fonts
            log.info(
                "Preserved custom fonts with LaTeX fallbacks: %s...",
                combined_fonts[:3],
            )
        else:
            # For non-custom fonts, use only LaTeX-compatible fonts
            mpl.rcParams["font.serif"] = [
                "Computer Modern Roman",  # Default LaTeX font
                "Times",
                "Times New Roman",
                "DejaVu Serif",
                "serif",  # Generic fallback
            ]
            log.info("Configured LaTeX-compatible fonts with fallbacks")

        # Ensure font family is properly set
        if "serif" not in mpl.rcParams["font.family"]:
            mpl.rcParams["font.family"] = ["serif"]

    # Colors
    color_palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # Line styles
    line_styles = [
        (0, ()),  # Solid
        (0, (1, 1)),  # Dotted
        (0, (5, 5)),  # Dashed
        (0, (3, 5, 1, 5)),  # Dashdotted
        (0, (3, 5, 1, 5, 1, 5)),  # Dashdotdotted
        (0, (1, 10)),  # Loosely dotted
        (0, (1, 1)),  # Densely dotted
        (5, (10, 3)),  # Long dash with offset
        (0, (5, 10)),  # Loosely dashed
        (0, (5, 1)),  # Densely dashed
        (0, (3, 10, 1, 10)),  # Loosely dashdotted
        (0, (3, 1, 1, 1)),  # Densely dashdotted
        (0, (3, 10, 1, 10, 1, 10)),  # Loosely dashdotdotted
        (0, (3, 1, 1, 1, 1, 1)),  # Densely dashdotdotted
    ]
    # Patterns for bar plots
    patterns = ["|", "", "/", "+", "-", ".", "*", "x", "o", "O"]
    mplstyle.use("fast")
    return color_palette, line_styles, patterns


def plot_colortable(colors: list[str], *, ncols: int = 4) -> Figure:
    """Plot a color table with the given colors.

    Parameters
    ----------
    colors : list[str]
        A list of color strings (e.g., hex codes or color names).
    ncols : int, optional
        The number of columns in the color table, by default 4.

    Returns
    -------
    Figure
        A matplotlib Figure containing the color table.

    """
    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 2

    n = len(colors)
    nrows = math.ceil(n / ncols)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(
        margin / width,
        margin / height,
        (width - margin) / width,
        (height - margin) / height,
    )
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.0)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, color in enumerate(colors):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col

        ax.add_patch(
            Rectangle(
                xy=(swatch_start_x, y - 9),
                width=swatch_width,
                height=18,
                facecolor=color,
                edgecolor="0.7",
            ),
        )

    return fig


def bold_all_ticks(
    *,
    format_fn: Callable[[float], str] = lambda x: rf"\textbf{{{int(x)}}}",
) -> None:
    """Bold all ticks on the current axis.

    Parameters
    ----------
    format_fn : Callable[[float], str], optional
        A function to format the tick labels. Default is a function that converts
        the tick value to an integer and wraps it in LaTeX bold formatting.

    """
    x_ticks_values, _ticks_labels = plt.xticks()
    assert isinstance(
        x_ticks_values,
        np.ndarray,
    ), f"Expected x_ticks_values to be a numpy array, got {type(x_ticks_values)}"
    plt.xticks(
        x_ticks_values,
        [format_fn(x) for x in x_ticks_values],
    )
    y_ticks_values, _ticks_labels = plt.yticks()
    assert isinstance(
        y_ticks_values,
        np.ndarray,
    ), f"Expected y_ticks_values to be a numpy array, got {type(y_ticks_values)}"
    plt.yticks(y_ticks_values, [format_fn(y) for y in y_ticks_values])


def bold_all_ticks_on_axis(
    ax: Axes,
    *,
    format_fn: Callable[[float], str] = lambda x: rf"\textbf{{{int(x)}}}",
) -> None:
    """Bold all ticks on the given axis.

    Parameters
    ----------
    ax : Axes
        The matplotlib Axes object to modify.
    format_fn : Callable[[float], str], optional
        A function to format the tick labels. Default is a function that converts
        the tick value to an integer and wraps it in LaTeX bold formatting.

    """
    x_ticks_values = ax.get_xticks()
    ax.set_xticks(
        x_ticks_values,
        [format_fn(x) for x in x_ticks_values],
    )
    y_ticks_values = ax.get_yticks()
    ax.set_yticks(y_ticks_values, [format_fn(y) for y in y_ticks_values])


def apply_improved_style() -> None:
    """Apply improved matplotlib styling for better visual hierarchy.

    This function applies enhancements to the current matplotlib style:
    - Better grid appearance with dashed lines and reduced opacity
    - Cleaner axes with no top/right spines
    - Improved tick styling with larger, more prominent ticks
    - Enhanced legend appearance with shadows and borders
    - Better typography hierarchy
    - Larger default figure size for better readability
    """
    mpl.rcParams.update(
        {
            # Better default figure size for readability
            "figure.figsize": [8, 6],
            "figure.dpi": 100,
            # Improved grid appearance
            "grid.linewidth": 0.5,  # Thinner grid lines
            "grid.alpha": 0.7,  # More subtle grid
            "grid.linestyle": "--",  # Dashed instead of solid
            # Better axes styling
            "axes.linewidth": 1.2,  # Slightly thicker axis lines
            "axes.spines.top": False,  # Remove top spine for cleaner look
            "axes.spines.right": False,  # Remove right spine for cleaner look
            "axes.labelpad": 8,  # More space for labels
            # Improved tick styling
            "xtick.major.size": 5,  # Larger tick marks
            "ytick.major.size": 5,
            "xtick.major.width": 1.2,  # Thicker tick marks
            "ytick.major.width": 1.2,
            "xtick.major.pad": 6,  # More padding for tick labels
            "ytick.major.pad": 6,
            "xtick.minor.visible": True,  # Show minor ticks
            "ytick.minor.visible": True,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            # Better line widths
            "lines.linewidth": 2.0,  # Thicker lines for better visibility
            "lines.markersize": 6,  # Larger markers
            # Improved legend
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.fancybox": True,
            "legend.shadow": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "gray",
            "legend.borderpad": 0.5,
            # Better text rendering
            "font.size": 12,  # Slightly smaller default font
            "axes.titlesize": 14,  # Larger title
            "axes.labelsize": 12,  # Consistent label size
            "xtick.labelsize": 10,  # Smaller tick labels
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        },
    )


def create_publication_ready_plot(
    figsize: tuple[float, float] = (8, 6),
    *,
    use_improved_style: bool = True,
    enable_grid: bool = True,
    grid_which: str = "both",
) -> tuple[Figure, Axes]:
    """Create a publication-ready plot with improved styling.

    Parameters
    ----------
    figsize : tuple[float, float], optional
        Figure size in inches, by default (8, 6)
    use_improved_style : bool, optional
        Whether to apply improved styling, by default True
    enable_grid : bool, optional
        Whether to enable grid, by default True
    grid_which : str, optional
        Which grid lines to show ('major', 'minor', 'both'), by default "both"

    Returns
    -------
    tuple[Figure, Axes]
        A tuple containing the figure and axes objects

    Examples
    --------
    >>> fig, ax = create_publication_ready_plot()
    >>> ax.plot([1, 2, 3], [1, 4, 2], label='Data')
    >>> ax.set_xlabel('X Label')
    >>> ax.set_ylabel('Y Label')
    >>> ax.set_title('My Plot')
    >>> ax.legend()
    >>> plt.tight_layout()
    >>> plt.show()

    """
    if use_improved_style:
        apply_improved_style()

    fig, ax = plt.subplots(figsize=figsize)

    if enable_grid:
        # Type-safe grid which parameter
        valid_which = ["major", "minor", "both"]
        if grid_which in valid_which:
            ax.grid(which=grid_which, alpha=0.3)  # type: ignore[arg-type]
        else:
            ax.grid(which="both", alpha=0.3)

    return fig, ax


def enhance_plot_for_presentation(
    ax: Axes,
    *,
    bold_ticks: bool = False,
    increase_linewidth: float = 1.5,
    legend_shadow: bool = True,
) -> None:
    """Enhance an existing plot for presentation purposes.

    Parameters
    ----------
    ax : Axes
        The matplotlib Axes object to enhance
    bold_ticks : bool, optional
        Whether to make tick labels bold, by default False
    increase_linewidth : float, optional
        Factor to increase line width by, by default 1.5
    legend_shadow : bool, optional
        Whether to add shadow to legend, by default True

    """
    # Enhance line widths
    for line in ax.get_lines():
        current_width = line.get_linewidth()
        line.set_linewidth(current_width * increase_linewidth)

    # Bold ticks if requested
    if bold_ticks:
        bold_all_ticks_on_axis(ax)

    # Enhance legend if present - using proper matplotlib API
    legend = ax.get_legend()
    if legend and legend_shadow:
        # Access the legend frame for styling
        frame = legend.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("gray")
        frame.set_alpha(0.9)
