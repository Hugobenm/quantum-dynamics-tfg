import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ============================================================
# Four-panel figure:
#   - Top row: sweep in g for J_y cross-sections (two electric fields)
#   - Bottom row: J_sigma, J_orb and J for two selected ground states
#
# Output:
#   - four_panel_currents.pdf
#   - four_panel_currents.png
#
# Expected files:
#   1) Sweep data, each containing at least:
#         currents_g   with shape (n_g, N_x)
#      Optionally:
#         g           interaction values
#
#   2) Ground-state files, each containing either:
#         psi
#      or:
#         psi_up, psi_down
#
#   3) A grid file containing:
#         x, y, Lx, Ly, N, B
#      (or those can be inside the ground-state files)
# ============================================================

# -----------------------------
# Plot style
# -----------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 12,
})

# -----------------------------
# User inputs
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT

# Grid file used to recover x, y, Lx, Ly, N if needed
GRID_FILE = "data/visualization_inputs/datos_tfg_gaugeLandau_g=0.npz"

# Sweep files (top row)
SWEEP_CASES = [
    {
        "file": "data/visualization_inputs/datos_currents_barrido_g_E=0.npz",
        "label": r"$\tilde{E}_x=0$",
    },
    {
        "file": "data/visualization_inputs/datos_currents_barrido_g_E=0.5.npz",
        "label": r"$\tilde{E}_x=0.5$",
    },
]

# Ground-state files (bottom row)
GROUND_CASES = [
    {
        "file": "data/visualization_inputs/datos_tfg_gaugeLandau_g=50.npz",
        "label": r"$\tilde{E}_x=0,\ \tilde{g}=50$",
        "E_x": 0.0,
        "g": 50.0,
    },
    {
        "file": "data/visualization_inputs/datos_tfg_g=50_E=0.5.npz",
        "label": r"$\tilde{E}_x=0.5,\ \tilde{g}=50$",
        "E_x": 0.5,
        "g": 50.0,
    },
]

# If the sweep files do NOT store the interaction values,
# they will be generated with np.linspace(G_MIN, G_MAX, n_curves)
G_MIN = 0.0
G_MAX = 100.0

# Physical constants / conventions
q = 1.0
hbar = 1.0
m = 1.0
B_DEFAULT = 1.0

# Thomas-Fermi parameters for the comparison curve.
TF_N = 1.0
TF_K = 0.0
TF_BZ = 1.0
TF_WB = 1.0
SHOW_TF_CURRENT = False

# Cross-section position
y_section = 0.0
x_zoom_factor = 1.5
FIGURE_MODE = "sweep_only"  # "full" or "sweep_only"

# Output names
OUTPUT_PDF = "results/figures/four_panel_currents.pdf"
OUTPUT_PNG = "results/figures/four_panel_currents.png"

# Colormap for the sweep
CMAP_NAME = "viridis"
SWEEP_LINEWIDTH = 2.65


# -----------------------------
# Helpers
# -----------------------------
def load_npz(filename):
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{filename}' in:\n{BASE_DIR}\n"
            "Edit the file names at the top of the script or place the script in the same folder."
        )
    return np.load(path)


def get_key(data, key, default=None):
    return data[key] if key in data.files else default


def load_grid():
    data = load_npz(GRID_FILE)
    x = data["x"]
    y = data["y"]
    Lx = float(get_key(data, "Lx", np.max(np.abs(x))))
    Ly = float(get_key(data, "Ly", np.max(np.abs(y))))
    N = int(get_key(data, "N", len(x)))
    B = float(get_key(data, "B", B_DEFAULT))
    return x, y, Lx, Ly, N, B


def load_ground_state(case_file, grid_file=GRID_FILE):
    data_case = load_npz(case_file)

    if "psi" in data_case.files:
        psi = data_case["psi"]
    elif "psi_up" in data_case.files and "psi_down" in data_case.files:
        psi = np.stack([data_case["psi_up"], data_case["psi_down"]], axis=0)
    else:
        raise KeyError(
            f"File '{case_file}' must contain either 'psi' or 'psi_up'/'psi_down'."
        )

    # Recover grid either from the case file or the generic grid file
    if "x" in data_case.files and "y" in data_case.files:
        x = data_case["x"]
        y = data_case["y"]
        Lx = float(get_key(data_case, "Lx", np.max(np.abs(x))))
        Ly = float(get_key(data_case, "Ly", np.max(np.abs(y))))
        N = int(get_key(data_case, "N", len(x)))
        B = float(get_key(data_case, "B", B_DEFAULT))
    else:
        x, y, Lx, Ly, N, B_grid = load_grid()
        B = float(get_key(data_case, "B", B_grid))

    return psi, x, y, Lx, Ly, N, B


def compute_currents(psi, x, y, B, q=1.0, hbar=1.0, m=1.0):
    """
    Compute J_orb, J_sigma and J = J_orb + J_sigma
    in the Landau gauge A=(0, Bx, 0).
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y)
    KX, KY = np.meshgrid(
        np.fft.fftfreq(len(x), d=dx) * 2 * np.pi,
        np.fft.fftfreq(len(y), d=dy) * 2 * np.pi,
    )

    A_x = np.zeros_like(X)
    A_y = B * X

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

    # Spin current: only C_z is needed for the in-plane curl
    C_z = np.abs(psi_up)**2 - np.abs(psi_down)**2
    C_z_k = np.fft.fft2(C_z)

    J_spin_x = (hbar / (2 * m)) * np.fft.ifft2(1j * KY * C_z_k).real
    J_spin_y = (hbar / (2 * m)) * np.fft.ifft2(-1j * KX * C_z_k).real

    J_x = J_orb_x + J_spin_x
    J_y = J_orb_y + J_spin_y

    return {
        "J_orb_x": J_orb_x,
        "J_orb_y": J_orb_y,
        "J_sigma_x": J_spin_x,
        "J_sigma_y": J_spin_y,
        "J_x": J_x,
        "J_y": J_y,
    }


def load_sweep_case(case_file):
    data = load_npz(case_file)

    if "currents_g" not in data.files:
        raise KeyError(f"File '{case_file}' must contain 'currents_g'.")

    currents_g = data["currents_g"]

    if "g" in data.files:
        g = data["g"]
    else:
        g = np.linspace(G_MIN, G_MAX, len(currents_g))

    return currents_g, g


def compute_tf_current(x, y, E_x, g_tf):
    """
    Thomas-Fermi current for k=0 and the dimensionless constants used here.

    The TF density is normalized with int rho_TF dx dy = TF_N, assuming a
    uniform y profile over the numerical y-domain.
    """
    y_length = float(y[-1] - y[0])
    if y_length <= 0:
        raise ValueError("The y grid must have a positive length.")

    trap_strength = m * TF_WB**2
    R_tf = (3.0 * g_tf * TF_N / (2.0 * trap_strength * y_length)) ** (1.0 / 3.0)
    mu_eff = 0.5 * trap_strength * R_tf**2
    x_k = hbar * TF_K / (q * TF_BZ) - m * E_x / (q * TF_BZ**2)
    x_shifted = x - x_k
    inside_tf = np.abs(x_shifted) <= R_tf
    rho_tf = (mu_eff - 0.5 * trap_strength * x_shifted**2) / g_tf
    rho_tf = np.where(inside_tf, rho_tf, 0.0)

    J_tf = np.full_like(x, np.nan, dtype=float)
    J_tf[inside_tf] = (
        (hbar * TF_WB / (2.0 * g_tf) - rho_tf) * TF_WB * x_shifted
        + rho_tf * E_x / TF_BZ
    )[inside_tf]
    return J_tf


def prepare_bottom_case(case):
    psi, x, y, Lx, Ly, N, B = load_ground_state(case["file"])
    idx_y = int(np.argmin(np.abs(y - y_section)))

    currents = compute_currents(psi, x, y, B, q=q, hbar=hbar, m=m)
    J_tf = compute_tf_current(x, y, case["E_x"], case["g"])

    return {
        "x": x,
        "J_sigma": currents["J_sigma_y"][idx_y, :],
        "J_orb": currents["J_orb_y"][idx_y, :],
        "J": currents["J_y"][idx_y, :],
        "J_TF": J_tf,
        "label": case["label"],
    }


# -----------------------------
# Load data
# -----------------------------
x_grid, y_grid, Lx, Ly, N, B_grid = load_grid()

# Top row
top_data = []
all_g_values = []
for case in SWEEP_CASES:
    currents_g, g_values = load_sweep_case(case["file"])
    top_data.append({
        "x": x_grid,
        "currents_g": currents_g,
        "g": g_values,
        "label": case["label"],
    })
    all_g_values.append(g_values)

all_g_values = np.concatenate(all_g_values)
g_min_global = float(np.min(all_g_values))
g_max_global = float(np.max(all_g_values))

if FIGURE_MODE not in {"full", "sweep_only"}:
    raise ValueError("FIGURE_MODE must be either 'full' or 'sweep_only'.")

# Selected cross-sections
bottom_data = [] if FIGURE_MODE == "sweep_only" else [prepare_bottom_case(case) for case in GROUND_CASES]

# -----------------------------
# Common J_max across ALL panels
# -----------------------------
candidates = []

for item in top_data:
    candidates.append(np.max(np.abs(item["currents_g"])))

for item in bottom_data:
    candidates.append(np.max(np.abs(item["J_sigma"])))
    candidates.append(np.max(np.abs(item["J_orb"])))
    candidates.append(np.max(np.abs(item["J"])))
    if SHOW_TF_CURRENT:
        candidates.append(np.nanmax(np.abs(item["J_TF"])))

J_max = max(candidates)

print(f"Common J_max = {J_max:.8e}")

# -----------------------------
# Plot
# -----------------------------
cmap = cm.get_cmap(CMAP_NAME)
norm = mcolors.Normalize(vmin=g_min_global, vmax=g_max_global)

xlim = (-Lx / x_zoom_factor, Lx / x_zoom_factor)

if FIGURE_MODE == "sweep_only":
    fig = plt.figure(figsize=(7.1, 2.75))
    gs = GridSpec(
        1, 3, figure=fig,
        width_ratios=[1.0, 1.0, 0.035],
        wspace=0.30,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])

    for ax, item in zip(axes, top_data):
        for curve, g_val in zip(item["currents_g"], item["g"]):
            ax.plot(item["x"], curve / J_max, color=cmap(norm(g_val)), lw=SWEEP_LINEWIDTH)

        ax.text(
            0.95, 0.93, item["label"],
            transform=ax.transAxes,
            ha="right", va="top", fontsize=12
        )
        ax.tick_params(direction="in", top=True, right=True)
        ax.set_xlim(*xlim)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$\tilde{g}$")

    fig.supxlabel(r"$\tilde{x}$", y=0.07)
    fig.supylabel(r"$J_y/J_{\mathrm{max}}$", x=0.028)
    fig.subplots_adjust(left=0.12, right=0.93, bottom=0.22, top=0.95, wspace=0.30)

    pos_right = axes[1].get_position()
    cax.set_position([pos_right.x1 + 0.010, pos_right.y0, 0.012, pos_right.height])
else:
    fig = plt.figure(figsize=(9.2, 5.8))
    gs = GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.0, 1.0, 0.035],
        height_ratios=[1.0, 1.0],
        wspace=0.205, hspace=0.04
    )

    # Axes
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_empty = fig.add_subplot(gs[0, 2])
    ax_empty.axis("off")

    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])
    cax   = fig.add_subplot(gs[1, 2])

    # --- Top row: selected cross-sections ---
    handles_for_legend = None

    for ax, item in zip([ax_tl, ax_tr], bottom_data):
        line_sigma, = ax.plot(item["x"], item["J_sigma"] / J_max, lw=1.65, ls="--", label=r"$J_{\sigma}$")
        line_orb,   = ax.plot(item["x"], item["J_orb"]   / J_max, lw=1.65, ls=":", label=r"$J_{\mathrm{orb}}$")
        line_tot,   = ax.plot(item["x"], item["J"]       / J_max, lw=1.65, ls="-", label=r"$J$")

        if handles_for_legend is None:
            handles_for_legend = [line_sigma, line_orb, line_tot]

        if SHOW_TF_CURRENT:
            line_tf, = ax.plot(item["x"], item["J_TF"] / J_max, lw=1.65, ls="-.", color="0.25", label=r"$J_{\mathrm{TF}}$")
            if len(handles_for_legend) == 3:
                handles_for_legend.append(line_tf)

        ax.text(
            0.95, 0.93, item["label"],
            transform=ax.transAxes,
            ha="right", va="top", fontsize=12
        )
        ax.tick_params(direction="in", top=True, right=True)
        ax.set_xlim(*xlim)
        ax.tick_params(labelbottom=False)

    # One single legend for the top row, using the empty slot above the colorbar
    ax_empty.legend(
        handles_for_legend,
        [handle.get_label() for handle in handles_for_legend],
        loc="center left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.0, 0.52),
        handlelength=1.9,
        borderaxespad=0.0,
        labelspacing=0.85,
    )

    # --- Bottom row: sweeps ---
    for ax, item in zip([ax_bl, ax_br], top_data):
        for curve, g_val in zip(item["currents_g"], item["g"]):
            ax.plot(item["x"], curve / J_max, color=cmap(norm(g_val)), lw=SWEEP_LINEWIDTH)

        ax.text(
            0.95, 0.93, item["label"],
            transform=ax.transAxes,
            ha="right", va="top", fontsize=12
        )
        ax.tick_params(direction="in", top=True, right=True)
        ax.set_xlim(*xlim)

    # Single colorbar for the bottom row
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$\tilde{g}$")

    # Shared axis labels for all four panels
    fig.supxlabel(r"$\tilde{x}$", y=0.04)
    fig.supylabel(r"$J_y/J_{\mathrm{max}}$", x=0.035)

    # Layout
    fig.subplots_adjust(left=0.11, right=0.93, bottom=0.11, top=0.96)

    # Keep the right-side guides close to the right panels while leaving a little
    # more breathing room between the two plot columns.
    pos_tr = ax_tr.get_position()
    pos_br = ax_br.get_position()
    ax_empty.set_position([pos_tr.x1 + 0.010, pos_tr.y0, 0.070, pos_tr.height])
    cax.set_position([pos_br.x1 + 0.010, pos_br.y0, 0.012, pos_br.height])

# Save
out_pdf = BASE_DIR / OUTPUT_PDF
out_png = BASE_DIR / OUTPUT_PNG
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")

plt.show()

print(f"Saved: {out_pdf.name}")
print(f"Saved: {out_png.name}")

