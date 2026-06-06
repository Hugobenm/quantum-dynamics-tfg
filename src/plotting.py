from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from .gp2d_split_step import Grid2D
from .observables import find_curve_crossings


def configure_matplotlib(use_tex: bool = False) -> None:
    """Set a compact publication-style Matplotlib theme."""
    plt.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
    })


def plot_density(psi: np.ndarray, grid: Grid2D, ax=None, component: int | None = None):
    """Plot a density map."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.4), constrained_layout=True)
    density = np.sum(np.abs(psi) ** 2, axis=0) if component is None else np.abs(psi[component]) ** 2
    im = ax.imshow(
        density,
        extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]],
        origin="lower",
        cmap="inferno",
        interpolation="bilinear",
        aspect="equal",
    )
    ax.set_xlabel(r"$\tilde{x}$")
    ax.set_ylabel(r"$\tilde{y}$")
    return im


def plot_current_panels(grid: Grid2D, current_data: dict[str, np.ndarray], filename: str | None = None):
    """Plot orbital, spin and total currents in a single row."""
    rho = current_data["rho"]
    current_pairs = [
        (current_data["J_orb_x"], current_data["J_orb_y"]),
        (current_data["J_spin_x"], current_data["J_spin_y"]),
        (current_data["J_x"], current_data["J_y"]),
    ]
    titles = [r"$\mathbf{J}_{\mathrm{orb}}$", r"$\mathbf{J}_{\sigma}$", r"$\mathbf{J}$"]
    colors = ["cyan", "white", "lime"]
    j_scale = max(np.max(np.sqrt(jx**2 + jy**2)) for jx, jy in current_pairs)
    j_scale = j_scale if j_scale > 0 else 1.0

    fig = plt.figure(figsize=(8.8, 3.0))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.045], wspace=0.11)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    for i, (ax, (jx, jy), title, color) in enumerate(zip(axes, current_pairs, titles, colors)):
        im = ax.imshow(
            rho,
            extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]],
            origin="lower",
            cmap="inferno",
            interpolation="bilinear",
            vmin=0,
            vmax=np.max(rho),
            aspect="equal",
        )
        j_mag = np.sqrt(jx**2 + jy**2)
        linewidth = np.clip(1.65 * j_mag / j_scale, 0, 1.65)
        mask = j_mag > 0.005 * j_scale
        ax.streamplot(
            grid.X,
            grid.Y,
            np.where(mask, jx, np.nan),
            np.where(mask, jy, np.nan),
            color=color,
            linewidth=linewidth,
            density=1.0,
            arrowsize=0.5,
            arrowstyle="-|>,head_length=0.75,head_width=0.28",
        )
        ax.set_title(title, pad=8)
        ax.tick_params(direction="out", top=True, right=True, labelleft=(i == 0))
        ax.set_xlabel("")
        ax.set_ylabel("")

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$|\tilde{\chi}|^2$", rotation=270, labelpad=17)
    fig.supxlabel(r"$\tilde{x}$", y=0.0)
    fig.supylabel(r"$\tilde{y}$", x=0.07)
    if filename:
        fig.savefig(filename, bbox_inches="tight")
    return fig, axes


def extract_current_cross_section(
    grid: Grid2D,
    current_data: dict[str, np.ndarray],
    y_section: float = 0.0,
) -> dict[str, np.ndarray | float | int]:
    """Extract the y-current cross-section nearest to `y_section`."""
    idx_y = int(np.argmin(np.abs(grid.y - y_section)))
    return {
        "x": grid.x,
        "J_sigma": current_data["J_spin_y"][idx_y, :],
        "J_orb": current_data["J_orb_y"][idx_y, :],
        "J": current_data["J_y"][idx_y, :],
        "idx_y": idx_y,
        "y_value": float(grid.y[idx_y]),
        "Lx": grid.Lx,
    }


def plot_current_cross_sections(
    cuts: list[dict[str, np.ndarray | float | int | str]],
    *,
    mode: str = "auto",
    x_zoom_factor: float = 1.5,
    filename: str | None = None,
):
    """Plot normalized transverse current cross-sections.

    Each entry in `cuts` should contain the keys returned by
    `extract_current_cross_section`. An optional `panel_label` key is used for
    in-panel labels.
    """
    if not cuts:
        raise ValueError("At least one cross-section is required.")

    if mode == "auto":
        mode = "single" if len(cuts) == 1 else ("four_panel" if len(cuts) >= 4 else "horizontal")

    j_max = max(
        np.max(np.abs(cut[name]))
        for cut in cuts
        for name in ["J_sigma", "J_orb", "J"]
    )
    if j_max == 0:
        raise ValueError("The selected cross-sections have zero current.")

    if mode == "single":
        fig, ax = plt.subplots(1, 1, figsize=(3.55, 2.55))
        axes = np.array([ax])
        selected = cuts[:1]
    elif mode == "four_panel":
        fig, axes_grid = plt.subplots(
            2, 2,
            figsize=(6.8, 4.55),
            sharex=True,
            sharey=False,
            gridspec_kw={"wspace": 0.13, "hspace": 0.10},
        )
        axes = axes_grid.ravel()
        selected = cuts[:4]
    elif mode == "horizontal":
        fig, axes = plt.subplots(
            1, len(cuts),
            figsize=(3.4 * len(cuts), 2.55),
            sharey=False,
            gridspec_kw={"wspace": 0.12},
        )
        axes = np.atleast_1d(axes)
        selected = cuts
    elif mode == "vertical":
        fig, axes = plt.subplots(
            len(cuts), 1,
            figsize=(3.55, 2.2 * len(cuts)),
            sharex=True,
            gridspec_kw={"hspace": 0.05},
        )
        axes = np.atleast_1d(axes)
        selected = cuts
    else:
        raise ValueError("mode must be 'auto', 'single', 'horizontal', 'vertical' or 'four_panel'.")

    for ax, cut in zip(axes, selected):
        x = cut["x"]
        xlim = (-float(cut["Lx"]) / x_zoom_factor, float(cut["Lx"]) / x_zoom_factor)
        normalized_curves = [
            cut["J_sigma"] / j_max,
            cut["J_orb"] / j_max,
            cut["J"] / j_max,
        ]

        ax.plot(x, normalized_curves[0], label=r"$J_{\sigma}$", lw=1.65, ls="--")
        ax.plot(x, normalized_curves[1], label=r"$J_{\mathrm{orb}}$", lw=1.65, ls=":")
        ax.plot(x, normalized_curves[2], label=r"$J$", lw=1.65, ls="-")

        panel_label = str(cut.get("panel_label", ""))
        if panel_label:
            ax.text(
                0.94,
                0.87,
                panel_label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=11,
            )

        ax.tick_params(direction="in", top=True, right=True)
        ax.set_xlim(*xlim)

        visible = (x >= xlim[0]) & (x <= xlim[1])
        y_values = np.concatenate([curve[visible] for curve in normalized_curves])
        y_min, y_max = np.min(y_values), np.max(y_values)
        y_pad = 0.10 * (y_max - y_min) if y_max > y_min else 0.05
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    for ax in axes[len(selected):]:
        ax.axis("off")

    if mode == "single":
        fig.supxlabel(r"$\tilde{x}$", x=0.56, y=0.08, fontsize=10)
        fig.supylabel(r"$J_y/J_{\mathrm{max}}$", x=0.050, fontsize=10)
        axes[0].legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.25))
        fig.subplots_adjust(left=0.20, right=0.98, bottom=0.22, top=0.78)
    elif mode == "four_panel":
        fig.supxlabel(r"$\tilde{x}$", x=0.54, y=0.055, fontsize=10)
        fig.supylabel(r"$J_y/J_{\mathrm{max}}$", x=0.045, fontsize=10)
        for ax in axes:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axes[0].legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(1.08, 1.26))
        for ax in axes[:2]:
            ax.tick_params(labelbottom=False)
        fig.subplots_adjust(left=0.13, right=0.99, bottom=0.14, top=0.88, wspace=0.13, hspace=0.10)
    elif mode == "horizontal":
        fig.supxlabel(r"$\tilde{x}$", x=0.54, y=0.08, fontsize=10)
        fig.supylabel(r"$J_y/J_{\mathrm{max}}$", x=0.050, fontsize=10)
        axes[0].legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(1.08, 1.25))
        fig.subplots_adjust(left=0.13, right=0.99, bottom=0.22, top=0.78, wspace=0.12)
    else:
        fig.supxlabel(r"$\tilde{x}$", y=0.055, fontsize=10)
        fig.supylabel(r"$J_y/J_{\mathrm{max}}$", x=0.032, fontsize=10)
        axes[0].legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.20))
        axes[0].tick_params(labelbottom=False)
        fig.subplots_adjust(left=0.20, right=0.985, bottom=0.13, top=0.86, hspace=0.05)

    if filename:
        fig.savefig(filename, bbox_inches="tight")
    return fig, axes, j_max


def plot_energy_branches(
    x: np.ndarray,
    energy_no_vortices: np.ndarray,
    energy_vortices: np.ndarray,
    xlabel: str,
    critical_label: str,
    ax=None,
):
    """Plot two energy branches and mark their crossing."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    ax.plot(x, energy_no_vortices, "o-", color="tab:orange", markersize=4, label="No vortices")
    ax.plot(x, energy_vortices, "s-", color="tab:blue", markersize=4, label="Vortices")
    crossings = find_curve_crossings(x, energy_vortices, energy_no_vortices)
    if crossings:
        x_cross, y_cross = crossings[0]
        ax.scatter(x_cross, y_cross, color="black", s=55, zorder=5)
        ax.axvline(x_cross, color="0.3", linestyle=":", linewidth=1.1)
        ax.text(
            x_cross,
            0.92,
            critical_label.format(x_cross),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$E$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return crossings


def animate_density(frames: list[np.ndarray], grid: Grid2D, interval: int = 50):
    """Create a Matplotlib density animation from precomputed frames."""
    fig, ax = plt.subplots(figsize=(4.5, 4.0), constrained_layout=True)
    vmax = max(np.max(frame) for frame in frames)
    im = ax.imshow(
        frames[0],
        extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]],
        origin="lower",
        cmap="inferno",
        vmin=0,
        vmax=vmax,
    )
    ax.set_xlabel(r"$\tilde{x}$")
    ax.set_ylabel(r"$\tilde{y}$")
    fig.colorbar(im, ax=ax, label=r"$|\tilde{\chi}|^2$")

    def update(index):
        im.set_data(frames[index])
        return (im,)

    return FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=True)
