
import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


def use_style():
    # Call before every experiment for high resolution image/formatting
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 8,
        "errorbar.capsize": 2,
    })

LABELS = {"gumbel": "Gumbel", "exp": "Exponential", "renyi": "Rényi"}
MARKERS = {"gumbel": "o", "exp": "^", "renyi": "s"}
COLOURS = {"gumbel": "C0", "exp": "C1", "renyi": "C2"}
LINESTYLES = {"gumbel": "-", "exp": "--", "renyi": "-."}


def tilt_x_ticks(ax, rotation=30):
    for label in ax.get_xticklabels(which="both"):
        label.set_rotation(rotation)
        label.set_horizontalalignment("right")


def label_ticks(ax, values, labelled, axis="x", fmt="{:g}", **kwargs):
    set_ticks, set_labels = ((ax.set_xticks, ax.set_xticklabels) if axis == "x"
                             else (ax.set_yticks, ax.set_yticklabels))
    set_ticks(values)
    set_labels([fmt.format(v) if v in labelled else "" for v in values], **kwargs)
    ax.minorticks_off()


def plain_line_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, ErrorbarContainer) else h for h in handles]
    return ax.legend(handles, labels, **kwargs)


def plain_log_ticks(ax, values, axis="y", fmt="{:g}"):
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(FixedLocator(list(values)))
    target.set_major_formatter(FuncFormatter(lambda value, _: fmt.format(value)))
    target.set_minor_locator(NullLocator())


def reference_line(ax, value, axis="y", color="gray", dotted=False, label=None):
    draw = ax.axvline if axis == "x" else ax.axhline
    return draw(value, color=color, linestyle=":" if dotted else "--", linewidth=0.8, label=label)
