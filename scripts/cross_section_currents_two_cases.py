import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from pathlib import Path

# ============================================================
# Cross-section plot for two Pauli spinor wave functions
# ============================================================
# The script computes J_orb, J_sigma and J = J_orb + J_sigma
# for two .npz wave functions, then plots the transverse section
# at y = y_section in two stacked panels with a common J_max.
#
# Expected .npz structure:
#   - Either each wave-function file contains x, y, Lx, Ly, N, B, psi
#   - Or use GRID_FILE to load x, y, Lx, Ly, N, B from a separate file.
#
# The spinor is assumed to be stored as psi[spin, y, x], with spin=0,1.
# ============================================================

# --- Style: same spirit as your current plotting script ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "font.size": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

# --- Files ---
ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT

# Set to None if each ground-state file already contains x, y, Lx, Ly, N.
GRID_FILE = "data/visualization_inputs/datos_tfg_gaugeLandau_g=0.npz"

CASES = [
    {
        "file": "data/visualization_inputs/datos_tfg_gaugeLandau_g=0.npz",
        "panel_label": r"",
    },
    {
        "file": "data/visualization_inputs/ground_state_split_step_B1_g18.npz", 
        "panel_label": r"",
    },
    {
        "file": "data/visualization_inputs/datos_tfg_gaugeLandau_g=0_E=0.5.npz",
        "panel_label": r"",
    },
    {
        "file": "data/visualization_inputs/ground_state_split_step_B1_g18_E0.5.npz",
        "panel_label": r"",
    },
]

# --- Parameters ---
B_DEFAULT = 1.0
q = 1.0
hbar = 1.0
m = 1.0

y_section = 0.0          # transverse cut at y = 0
x_zoom_factor = 1.5      # xlim = [-Lx/x_zoom_factor, Lx/x_zoom_factor]
PLOT_MODE = "four_panel"         # "all", "single", or "four_panel"
SINGLE_CASE_INDEX = 0     # 0 for first case, 1 for second case
PANEL_LAYOUT = "horizontal"  # "vertical" or "horizontal"
OUTPUT_STEM = "results/figures/cross_section_currents"


def load_npz(filename):
    """Load an npz file using a path relative to the script location."""
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{filename}' in:\n{BASE_DIR}\n"
            "Move the script to the data folder or edit the file paths."
        )
    return np.load(path)


def get_key(data, key, default=None):
    """Small helper to read optional npz keys."""
    return data[key] if key in data.files else default


def load_grid_and_spinor(case_file, grid_file=None):
    """Load psi plus the numerical grid."""
    data_case = load_npz(case_file)

    # Spinor
    if "psi" in data_case.files:
        psi = data_case["psi"]
    elif "psi_up" in data_case.files and "psi_down" in data_case.files:
        psi = np.stack([data_case["psi_up"], data_case["psi_down"]], axis=0)
    else:
        raise KeyError(
            f"File '{case_file}' must contain either 'psi' or 'psi_up'/'psi_down'."
        )

    # Grid: either from GRID_FILE or from the same case file
    if grid_file is not None:
        data_grid = load_npz(grid_file)
    else:
        data_grid = data_case

    try:
        x = data_grid["x"]
        y = data_grid["y"]
    except KeyError as exc:
        raise KeyError(
            "Could not find x,y grid arrays. Either include them in each .npz "
            "or set GRID_FILE to a file containing x and y."
        ) from exc

    N = int(get_key(data_grid, "N", len(x)))
    Lx = float(get_key(data_grid, "Lx", np.max(np.abs(x))))
    Ly = float(get_key(data_grid, "Ly", np.max(np.abs(y))))

    # Prefer B stored in the wave-function file; otherwise in the grid file; otherwise default.
    B_case = get_key(data_case, "B", None)
    B_grid = get_key(data_grid, "B", None)
    B = float(B_case if B_case is not None else (B_grid if B_grid is not None else B_DEFAULT))

    return psi, x, y, Lx, Ly, N, B


def compute_currents(psi, x, y, B, q=1.0, hbar=1.0, m=1.0):
    """
    Compute orbital and spin probability currents for a Pauli spinor.

    Gauge:
        A = (0, B x, 0)

    Current:
        J_orb = Re{ psi^dagger (p - q A) psi } / m
        J_sigma = hbar/(2m) curl(psi^dagger sigma psi)
    """
    N_y, N_x = len(y), len(x)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y)
    KX, KY = np.meshgrid(
        np.fft.fftfreq(N_x, d=dx) * 2 * np.pi,
        np.fft.fftfreq(N_y, d=dy) * 2 * np.pi,
    )

    A_x = np.zeros_like(X)
    A_y = B * X

    # Fourier derivatives for the momentum operator p = hbar k
    psi_k = np.fft.fft2(psi, axes=(1, 2))
    kx_psi = np.fft.ifft2(KX * psi_k, axes=(1, 2))
    ky_psi = np.fft.ifft2(KY * psi_k, axes=(1, 2))

    psi_up = psi[0]
    psi_down = psi[1]
    psi_up_conj = np.conj(psi_up)
    psi_down_conj = np.conj(psi_down)

    # Orbital current
    J_orb_x = (
        psi_up_conj * (hbar * kx_psi[0] - q * A_x * psi_up)
        + psi_down_conj * (hbar * kx_psi[1] - q * A_x * psi_down)
    ).real / m

    J_orb_y = (
        psi_up_conj * (hbar * ky_psi[0] - q * A_y * psi_up)
        + psi_down_conj * (hbar * ky_psi[1] - q * A_y * psi_down)
    ).real / m

    # Spin density C = psi^dagger sigma psi
    psi_up_conj_psi_down = psi_up_conj * psi_down
    C_z = np.abs(psi_up)**2 - np.abs(psi_down)**2

    # Spin current: J_sigma = hbar/(2m) curl(C)
    C_z_k = np.fft.fft2(C_z)

    J_spin_x = (hbar / (2 * m)) * np.fft.ifft2(1j * KY * C_z_k).real
    J_spin_y = (hbar / (2 * m)) * np.fft.ifft2(-1j * KX * C_z_k).real

    return {
        "J_orb_x": J_orb_x,
        "J_orb_y": J_orb_y,
        "J_spin_x": J_spin_x,
        "J_spin_y": J_spin_y,
        "J_total_x": J_orb_x + J_spin_x,
        "J_total_y": J_orb_y + J_spin_y,
    }


def prepare_case(case):
    """Load data, compute currents and extract the y = y_section cut."""
    psi, x, y, Lx, Ly, N, B = load_grid_and_spinor(case["file"], GRID_FILE)
    currents = compute_currents(psi, x, y, B, q=q, hbar=hbar, m=m)

    idx_y = int(np.argmin(np.abs(y - y_section)))

    return {
        "x": x,
        "J_sigma": currents["J_spin_y"][idx_y, :],
        "J_orb": currents["J_orb_y"][idx_y, :],
        "J": currents["J_total_y"][idx_y, :],
        "Lx": Lx,
        "Ly": Ly,
        "N": N,
        "B": B,
        "idx_y": idx_y,
        "y_value": y[idx_y],
        "panel_label": case["panel_label"],
    }


if PLOT_MODE not in {"all", "single", "four_panel"}:
    raise ValueError("PLOT_MODE must be either 'all', 'single' or 'four_panel'.")

if PLOT_MODE == "single":
    if not 0 <= SINGLE_CASE_INDEX < len(CASES):
        raise ValueError("SINGLE_CASE_INDEX must select an entry from CASES.")
    selected_cases = [CASES[SINGLE_CASE_INDEX]]
elif PLOT_MODE == "four_panel":
    if len(CASES) < 4:
        raise ValueError("PLOT_MODE='four_panel' needs at least four entries in CASES.")
    selected_cases = CASES[:4]
else:
    selected_cases = CASES

output_suffix = PLOT_MODE
if PLOT_MODE == "all":
    output_suffix = f"all_{PANEL_LAYOUT}"
elif PLOT_MODE == "single":
    output_suffix = f"single_{SINGLE_CASE_INDEX + 1}"

OUTPUT_PDF = f"{OUTPUT_STEM}_{output_suffix}.pdf"
OUTPUT_PNG = f"{OUTPUT_STEM}_{output_suffix}.png"

# --- Compute selected cases ---
all_cuts = [prepare_case(case) for case in selected_cases]

# Common Jmax for both panels and all three curves.
J_max = max(
    np.max(np.abs(cuts[name]))
    for cuts in all_cuts
    for name in ["J_sigma", "J_orb", "J"]
)

if J_max == 0:
    raise ValueError("J_max is zero. The selected cross-section has no current.")

print(f"Common J_max = {J_max:.6e}")
for i, cuts in enumerate(all_cuts):
    print(
        f"Case {i+1}: file={selected_cases[i]['file']}, B={cuts['B']}, "
        f"y_index={cuts['idx_y']}, y={cuts['y_value']:.4g}"
    )


# --- Plot ---
if PLOT_MODE == "single":
    fig, ax = plt.subplots(
        1, 1,
        figsize=(3.55, 2.55),
    )
    axes = np.array([ax])
elif PLOT_MODE == "four_panel":
    fig, axes_grid = plt.subplots(
        2, 2,
        figsize=(6.8, 4.55),
        sharex=True,
        sharey=False,
        gridspec_kw={"wspace": 0.13, "hspace": 0.10},
    )
    axes = axes_grid.ravel()
elif PANEL_LAYOUT == "horizontal":
    fig, axes = plt.subplots(
        1, 2,
        figsize=(6.8, 2.55),
        sharey=False,
        gridspec_kw={"wspace": 0.12},
    )
elif PANEL_LAYOUT == "vertical":
    fig, axes = plt.subplots(
        2, 1,
        figsize=(3.55, 4.35),
        sharex=True,
        gridspec_kw={"hspace": 0.05},
    )
else:
    raise ValueError("PANEL_LAYOUT must be either 'vertical' or 'horizontal'.")

for ax, cuts in zip(axes, all_cuts):
    x = cuts["x"]
    xlim = (-cuts["Lx"] / x_zoom_factor, cuts["Lx"] / x_zoom_factor)
    normalized_curves = [
        cuts["J_sigma"] / J_max,
        cuts["J_orb"] / J_max,
        cuts["J"] / J_max,
    ]

    ax.plot(x, normalized_curves[0], label=r"$J_{\sigma}$", lw=1.65, ls="--")
    ax.plot(x, normalized_curves[1], label=r"$J_{\mathrm{orb}}$", lw=1.65, ls=":")
    ax.plot(x, normalized_curves[2], label=r"$J$", lw=1.65, ls="-")

    ax.text(
        0.94, 0.87,
        cuts["panel_label"],
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )

    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xlim(*xlim)
    ax.grid(False)

    visible = (x >= xlim[0]) & (x <= xlim[1])
    y_values = np.concatenate([curve[visible] for curve in normalized_curves])
    y_min, y_max = np.min(y_values), np.max(y_values)
    y_pad = 0.10 * (y_max - y_min) if y_max > y_min else 0.05
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

if PLOT_MODE == "single":
    fig.supxlabel(r"$\tilde{x}$", x=0.56, y=0.08, fontsize=10)
    fig.supylabel(r"$J_y/J_{\mathrm{max}}$", x=0.050, fontsize=10)
    axes[0].legend(
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.25),
        handlelength=2.2,
        columnspacing=1.25,
    )
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.22, top=0.78)
elif PLOT_MODE == "four_panel":
    fig.supxlabel(r"$\tilde{x}$", x=0.54, y=0.055, fontsize=10)
    fig.supylabel(r"$J_y/J_{\mathrm{max}}$", x=0.045, fontsize=10)
    for ax in axes:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axes[0].legend(
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(1.08, 1.26),
        handlelength=2.2,
        columnspacing=1.25,
    )
    for ax in axes[:2]:
        ax.tick_params(labelbottom=False)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.14, top=0.88, wspace=0.13, hspace=0.10)
elif PANEL_LAYOUT == "horizontal":
    fig.supxlabel(r"$\tilde{x}$", x=0.54, y=0.08, fontsize=10)
    fig.supylabel(r"$J_y/J_{\mathrm{max}}$", x=0.050, fontsize=10)
    axes[0].legend(
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(1.08, 1.25),
        handlelength=2.2,
        columnspacing=1.25,
    )
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.22, top=0.78, wspace=0.12)
else:
    fig.supxlabel(r"$\tilde{x}$", y=0.055, fontsize=10)
    fig.supylabel(r"$J_y/J_{\mathrm{max}}$", x=0.032, fontsize=10)
    axes[0].legend(
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.20),
        handlelength=2.2,
        columnspacing=1.25,
    )
    # Avoid duplicated x tick labels on the top panel
    axes[0].tick_params(labelbottom=False)
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.13, top=0.86, hspace=0.05)

# Save
fig.savefig(BASE_DIR / OUTPUT_PDF, bbox_inches="tight")
fig.savefig(BASE_DIR / OUTPUT_PNG, dpi=300, bbox_inches="tight")

plt.show()

