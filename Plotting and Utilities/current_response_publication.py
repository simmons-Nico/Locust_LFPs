"""Presentation only: current-response figures retain independent observations."""
from pathlib import Path
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties


def publication_options(options=None):
    cfg = dict(size="double", legend_order=[], legend_note="", note_overrides={}, horizontal_offset_na=0.)
    if options is not None:
        if not isinstance(options, dict) or set(options) - set(cfg):
            raise ValueError("Unknown current-response presentation settings.")
        cfg.update(options)
    if cfg["size"] not in ("single", "double"):
        raise ValueError("Publication size must be single or double.")
    if not isinstance(cfg["legend_order"], list) or not all(isinstance(k, str) for k in cfg["legend_order"]):
        raise ValueError("Legend order must be a list of source paths or preparation IDs.")
    cfg["legend_order"] = list(dict.fromkeys(str(Path(k).resolve()) if Path(k).is_absolute() else k for k in cfg["legend_order"]))
    if not isinstance(cfg["legend_note"], str) or not isinstance(cfg["note_overrides"], dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in cfg["note_overrides"].items()):
        raise ValueError("Legend notes must be text, with per-plot overrides keyed by plot ID.")
    cfg["horizontal_offset_na"] = float(cfg["horizontal_offset_na"])
    if not np.isfinite(cfg["horizontal_offset_na"]) or cfg["horizontal_offset_na"] < 0:
        raise ValueError("Horizontal offset must be finite and nonnegative (zero disables it).")
    return cfg


def current_colors(currents):
    values = sorted(set(float(c) for c in currents), reverse=True)
    # Avoid near-white and near-black endpoints of this sequential single-hue map.
    levels = np.linspace(.48, .88, len(values)) if len(values) > 1 else [.68]
    return {current: to_hex(plt.get_cmap("Blues")(level)) for current, level in zip(values, levels)}


def legend_ids(points, order):
    records = points[["preparation_id", "source_file"]].drop_duplicates().sort_values("source_file")
    lookup = dict(zip(records.source_file, records.preparation_id))
    lookup.update({prep: prep for prep in records.preparation_id})
    selected = [lookup[key] for key in order if key in lookup]
    return list(dict.fromkeys(selected + records.preparation_id.tolist()))


def fitted_text(text, columns, width_in, font_size):
    """Wrap by rendered width as well as characters, including long unbroken IDs."""
    renderer = RendererAgg(1, 1, 72)
    font = FontProperties(family="DejaVu Sans", size=font_size)
    while True:
        result = "\n".join(textwrap.fill(line, columns) for line in text.splitlines())
        if columns <= 1 or all(renderer.get_text_width_height_descent(line, font, False)[0] <= width_in*72 for line in result.splitlines()):
            return result
        columns -= 1


def response_figure(points, summary, title, styles, options=None, colors=None, plot_id=""):
    # Legacy presentation options remain accepted for saved configurations.
    cfg = publication_options(options)
    small = cfg["size"] == "single"
    width, font, title_font = (3.5, 8., 11.) if small else (7.2, 10., 14.)
    height = 3.4 if small else 4.8
    fill, edge = "#6c9fba", "#203040"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": font, "text.usetex": False}):
        fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    fig.suptitle(fitted_text(title, 40 if small else 72, width-.25, title_font),
                 fontsize=title_font, fontfamily="DejaVu Sans", y=.96)
    groups = list(points.groupby("current_na", sort=True))
    if groups:
        positions = np.array([current for current, _ in groups])
        box_width = .28 * np.min(np.diff(positions)) if len(positions) > 1 else 28.
        ax.boxplot([g.normalized_rate_pct.to_numpy() for _, g in groups],
                   positions=positions, widths=box_width, whis=1.5,
                   showfliers=False, manage_ticks=False, patch_artist=True, zorder=2,
                   boxprops=dict(facecolor="none", edgecolor=".4", linewidth=1.),
                   whiskerprops=dict(color=".4", linewidth=1.),
                   capprops=dict(color=".4", linewidth=1.),
                   medianprops=dict(color=".25", linewidth=1.2))
    ax.axhline(100, color=".5", ls=(0, (4, 3)), lw=.8)
    ax.set(xlabel="Applied cathodic current (nA)", ylabel="Firing rate (% of baseline)")
    ax.set_xticks(sorted(points.current_na.unique(), reverse=True))
    ax.invert_xaxis()
    ax.set_ylim(min(0, ax.get_ylim()[0]), max(105, ax.get_ylim()[1]))
    ax.margins(x=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3, width=.7, labelsize=font)
    for label in [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontfamily("DejaVu Sans")
        label.set_fontsize(font)
    ax.xaxis.label.set_size(font+1)
    ax.yaxis.label.set_size(font+1)
    fig.subplots_adjust(left=.20 if small else .13, right=.95,
                        bottom=.18 if small else .14, top=.80 if small else .84)
    # Freeze limits before measuring marker overlap in display coordinates.
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    fig.canvas.draw()
    separation_px = 6.0 * fig.dpi / 72  # 5 pt marker plus outline/clearance.
    pixels_per_na = abs(ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0])
    effective = []
    for current, group in groups:
        group = group.sort_values(["normalized_rate_pct", "preparation_id", "source_file"], kind="stable")
        placed = []
        for row in group.itertuples():
            y_px = ax.transData.transform((current, row.normalized_rate_pct))[1]
            # Test exact collision boundaries: choose the smallest feasible shift.
            candidates = [0., -5., 5.]
            for dx, other_y in placed:
                dy = abs(y_px-other_y)
                if dy < separation_px:
                    gap = np.sqrt(separation_px**2-dy**2) / pixels_per_na
                    candidates.extend([dx-gap, dx+gap])
            candidates = sorted((x for x in candidates if abs(x) <= 5.), key=lambda x: (abs(x), x))
            def clearance(dx):
                return min((np.hypot((dx-x)*pixels_per_na, y_px-y) for x, y in placed), default=float("inf"))
            shift = next((x for x in candidates if clearance(x) >= separation_px-1e-8),
                         max(candidates, key=clearance))
            placed.append((shift, y_px))
            ax.scatter([current+shift], [row.normalized_rate_pct], facecolors=fill,
                       edgecolors=edge, linewidths=.55, marker="o", s=25, zorder=4)
            effective.append(dict(preparation_id=row.preparation_id, source_file=row.source_file,
                                  display_name=row.display_name, display_label=row.display_label,
                                  current_na=float(current), plotted_current_na=float(current+shift),
                                  normalized_rate_pct=float(row.normalized_rate_pct),
                                  marker="o", color=fill))
    fig.publication_metadata = dict(settings={"size": cfg["size"]}, legend_note="", legend_order=[],
                                    color_mapping={str(c): fill for c, _ in groups}, observations=effective,
                                    title=title, width_in=width, height_in=height, dpi=600,
                                    font="DejaVu Sans", font_size_pt=font, title_size_pt=title_font,
                                    svg_text="editable", palette="uniform muted blue",
                                    marker_size_pt=5, marker_edge=edge, marker_linewidth_pt=.55,
                                    jitter="deterministic collision avoidance; maximum +/-5 nA",
                                    formats=["png", "svg", "pdf"])
    return fig
