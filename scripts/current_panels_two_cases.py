import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ============================================================
# Two-row current-panel figure
# ============================================================
# For each input ground-state spinor, this script computes:
#   - Orbital current     J_orb
#   - Spin current        J_sigma
#   - Total current       J = J_orb + J_sigma
#
# It then produces a 2 x 3 panel figure:
#   row 1: first ground-state file
#   row 2: second ground-state file
#   columns: J_orb, J_sigma, J
#
# The density rho = |psi_up|^2 + |psi_down|^2 is used as background.
# Streamlines represent the corresponding current.
#
# Expected .npz structure:
#   - Either each ground-state file contains x, y, Lx, Ly, N, B, psi
#   - Or use GRID_FILE to load x, y, Lx, Ly, N, B from a separate file.
#
# The spinor is assumed to be stored as psi[spin, y, x], with spin=0,1.
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

# If the ground-state files do not store x, y, Lx, Ly, N, use this file.
GRID_FILE = "data/visualization_inputs/datos_tfg_gaugeLandau_g=0.npz"

CASES = [
    {
        "file": "data/visualization_inputs/ground_state_E0.25_g50_vortices.npz",
        "row_label": r"(a) $\tilde{E}_x=0$",
    },
    {
        "file": "data/visualization_inputs/ground_state_E0.25_g50_vortices.npz",
        "row_label": r"(b) $\tilde{E}_x=0.25$",
    },
]

# Physical constants / conventions
q = 1.0
hbar = 1.0
m = 1.0
B_DEFAULT = 1.0

# Output
OUTPUT_PDF = "results/figures/current_panels_two_cases.pdf"
OUTPUT_PNG = "results/figures/current_panels_two_cases.png"

# Streamplot settings
STREAM_DENSITY = 1.0
ARROWSIZE = 0.5
LINEWIDTH_MAX = 1.65
CURRENT_THRESHOLD = 0.05   # hides streamlines with |J| < CURRENT_THRESHOLD * J_scale

# Use "row" to normalize streamline widths row by row,
# or "global" to normalize all six panels using the same current scale.
STREAM_NORMALIZATION = "row"

# Density color scale:
#   False -> each row has its own density color scale and colorbar.
#   True  -> both rows share the same density scale.
COMMON_DENSITY_SCALE = False

# Figure size
FIGSIZE = (9.0, 5.8)
SINGLE_ROW_FIGSIZE = (8.8, 3.0)
FIGURE_MODE = "single_row"  # "all" or "single_row"
SINGLE_ROW_INDEX = 0  # 0 for first case, 1 for second case

# Density background.
DENSITY_CMAP = "inferno"


# -----------------------------
# Helpers
# -----------------------------
def load_npz(filename):
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{filename}' in:\n{BASE_DIR}\n"
            "Edit the file names at the top of the script or place the script in the data folder."
        )
    return np.load(path)


def get_key(data, key, default=None):
    return data[key] if key in data.files else default


def load_grid_from_file():
    data = load_npz(GRID_FILE)
    x = data["x"]
    y = data["y"]
    Lx = float(get_key(data, "Lx", np.max(np.abs(x))))
    Ly = float(get_key(data, "Ly", np.max(np.abs(y))))
    N = int(get_key(data, "N", len(x)))
    B = float(get_key(data, "B", B_DEFAULT))
    return x, y, Lx, Ly, N, B


def load_ground_state(case_file):
    data_case = load_npz(case_file)

    if "psi" in data_case.files:
        psi = data_case["psi"]
    elif "psi_up" in data_case.files and "psi_down" in data_case.files:
        psi = np.stack([data_case["psi_up"], data_case["psi_down"]], axis=0)
    else:
        raise KeyError(
            f"File '{case_file}' must contain either 'psi' or 'psi_up'/'psi_down'."
        )

    # Prefer grid stored in the same file. Otherwise, use GRID_FILE.
    if "x" in data_case.files and "y" in data_case.files:
        x = data_case["x"]
        y = data_case["y"]
        Lx = float(get_key(data_case, "Lx", np.max(np.abs(x))))
        Ly = float(get_key(data_case, "Ly", np.max(np.abs(y))))
        N = int(get_key(data_case, "N", len(x)))
        B = float(get_key(data_case, "B", B_DEFAULT))
    else:
        x, y, Lx, Ly, N, B_grid = load_grid_from_file()
        B = float(get_key(data_case, "B", B_grid))

    return psi, x, y, Lx, Ly, N, B


def compute_currents(psi, x, y, B, q=1.0, hbar=1.0, m=1.0):
    r"""
    Landau gauge A = (0, Bx, 0).

    J_orb   = Re{psi^\dagger (p - qA) psi}/m
    J_sigma = hbar/(2m) curl(psi^\dagger sigma psi)
    J       = J_orb + J_sigma
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

    # Momentum-space derivatives: p psi = hbar k psi
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

    # Spin current
    # C_z = psi^\dagger sigma_z psi
    C_z = np.abs(psi_up)**2 - np.abs(psi_down)**2
    C_z_k = np.fft.fft2(C_z)

    J_sigma_x = (hbar / (2 * m)) * np.fft.ifft2(1j * KY * C_z_k).real
    J_sigma_y = (hbar / (2 * m)) * np.fft.ifft2(-1j * KX * C_z_k).real

    J_x = J_orb_x + J_sigma_x
    J_y = J_orb_y + J_sigma_y

    rho = np.abs(psi_up)**2 + np.abs(psi_down)**2

    return {
        "X": X,
        "Y": Y,
        "rho": rho,
        "J_orb_x": J_orb_x,
        "J_orb_y": J_orb_y,
        "J_sigma_x": J_sigma_x,
        "J_sigma_y": J_sigma_y,
        "J_x": J_x,
        "J_y": J_y,
    }


def prepare_case(case):
    psi, x, y, Lx, Ly, N, B = load_ground_state(case["file"])
    currents = compute_currents(psi, x, y, B, q=q, hbar=hbar, m=m)

    norm = np.sum(currents["rho"]) * (x[1] - x[0]) * (y[1] - y[0])
    print(
        f"Loaded {case['file']}: grid={len(x)}x{len(y)}, "
        f"B={B:.6g}, norm={norm:.6f}"
    )

    return {
        **case,
        "psi": psi,
        "x": x,
        "y": y,
        "Lx": Lx,
        "Ly": Ly,
        "N": N,
        "B": B,
        **currents,
    }


def plot_stream_panel(ax, data, Jx, Jy, color_stream, density_vmax, J_scale):
    rho = data["rho"]
    X = data["X"]
    Y = data["Y"]

    im = ax.imshow(
        rho,
        extent=[data["x"][0], data["x"][-1], data["y"][0], data["y"][-1]],
        origin="lower",
        cmap=DENSITY_CMAP,
        interpolation="bilinear",
        vmin=0.0,
        vmax=density_vmax,
        aspect="equal",
    )

    J_mag = np.sqrt(Jx**2 + Jy**2)
    lw = LINEWIDTH_MAX * (J_mag / J_scale)
    lw = np.clip(lw, 0.0, LINEWIDTH_MAX)

    threshold = CURRENT_THRESHOLD * J_scale
    Jx_plot = np.where(J_mag > threshold, Jx, np.nan)
    Jy_plot = np.where(J_mag > threshold, Jy, np.nan)

    ax.streamplot(
        X,
        Y,
        Jx_plot,
        Jy_plot,
        color=color_stream,
        linewidth=lw,
        density=STREAM_DENSITY,
        arrowsize=ARROWSIZE,
        arrowstyle="-|>,head_length=0.75,head_width=0.28",
    )

    ax.tick_params(direction="out", top=True, right=True)
    ax.set_xlim(data["x"][0], data["x"][-1])
    ax.set_ylim(data["y"][0], data["y"][-1])

    return im


# -----------------------------
# Load and compute
# -----------------------------
if FIGURE_MODE not in {"all", "single_row"}:
    raise ValueError("FIGURE_MODE must be either 'all' or 'single_row'.")

if FIGURE_MODE == "single_row":
    if not 0 <= SINGLE_ROW_INDEX < len(CASES):
        raise ValueError("SINGLE_ROW_INDEX must select an entry from CASES.")
    selected_cases = [CASES[SINGLE_ROW_INDEX]]
else:
    selected_cases = CASES

rows = [prepare_case(case) for case in selected_cases]

# Current scale for stream widths
row_J_scales = []
for row in rows:
    candidates = [
        np.max(np.sqrt(row["J_orb_x"]**2 + row["J_orb_y"]**2)),
        np.max(np.sqrt(row["J_sigma_x"]**2 + row["J_sigma_y"]**2)),
        np.max(np.sqrt(row["J_x"]**2 + row["J_y"]**2)),
    ]
    row_J_scales.append(max(candidates))

global_J_scale = max(row_J_scales)

print(f"Row current scales: {[f'{v:.8e}' for v in row_J_scales]}")
print(f"Global current scale: {global_J_scale:.8e}")

if STREAM_NORMALIZATION.lower() == "global":
    J_scales = [global_J_scale for _ in rows]
else:
    J_scales = row_J_scales

# Density scales
if COMMON_DENSITY_SCALE:
    common_rho_max = max(np.max(row["rho"]) for row in rows)
    rho_vmax = [common_rho_max for _ in rows]
else:
    rho_vmax = [np.max(row["rho"]) for row in rows]

print(f"Density colorbar maxima: {[f'{v:.8e}' for v in rho_vmax]}")


# -----------------------------
# Figure
# -----------------------------
n_rows = len(rows)
fig = plt.figure(figsize=SINGLE_ROW_FIGSIZE if FIGURE_MODE == "single_row" else FIGSIZE)

# 3 current panels + one colorbar column
gs = GridSpec(
    n_rows, 4,
    figure=fig,
    width_ratios=[1.0, 1.0, 1.0, 0.045],
    wspace=0.11,
    hspace=0.08 if n_rows > 1 else 0.0,
)

axes = np.empty((n_rows, 3), dtype=object)
caxes = []

for i in range(n_rows):
    for j in range(3):
        axes[i, j] = fig.add_subplot(gs[i, j])
    caxes.append(fig.add_subplot(gs[i, 3]))

# Column titles only on the top row
column_titles = [
    r"$\mathbf{J}_{\mathrm{orb}}$",
    r"$\mathbf{J}_{\sigma}$",
    r"$\mathbf{J}$",
]
for j, title in enumerate(column_titles):
    axes[0, j].set_title(title, pad=8)

# Plot rows
stream_colors = {
    "orb": "cyan",
    "sigma": "white",
    "total": "lime",
}

for i, row in enumerate(rows):
    im0 = plot_stream_panel(
        axes[i, 0],
        row,
        row["J_orb_x"], row["J_orb_y"],
        stream_colors["orb"],
        rho_vmax[i],
        J_scales[i],
    )
    plot_stream_panel(
        axes[i, 1],
        row,
        row["J_sigma_x"], row["J_sigma_y"],
        stream_colors["sigma"],
        rho_vmax[i],
        J_scales[i],
    )
    plot_stream_panel(
        axes[i, 2],
        row,
        row["J_x"], row["J_y"],
        stream_colors["total"],
        rho_vmax[i],
        J_scales[i],
    )

    if FIGURE_MODE != "single_row":
        # Row label: (a), (b)
        axes[i, 0].text(
            -0.28, 0.50,
            row["row_label"],
            transform=axes[i, 0].transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=12,
        )

    # One colorbar per row
    cbar = fig.colorbar(im0, cax=caxes[i])
    colorbar_labelpad = 17 if FIGURE_MODE == "single_row" else 14
    cbar.set_label(r"$|\tilde{\chi}|^2$", rotation=270, labelpad=colorbar_labelpad)
    cbar.ax.tick_params(direction="out")

# Remove repeated individual axis labels
for i in range(n_rows):
    for j in range(3):
        axes[i, j].set_xlabel("")
        axes[i, j].set_ylabel("")
        axes[i, j].tick_params(
            labelbottom=(i == n_rows - 1),
            labelleft=(j == 0),
            labeltop=False,
            labelright=False,
        )

# Shared labels for all panels
if FIGURE_MODE == "single_row":
    fig.supxlabel(r"$\tilde{x}$", y=0.000)
    fig.supylabel(r"$\tilde{y}$", x=0.070)
else:
    fig.supxlabel(r"$\tilde{x}$", y=0.04)
    fig.supylabel(r"$\tilde{y}$", x=0.045)

# Layout and save
fig.savefig(BASE_DIR / OUTPUT_PDF, bbox_inches="tight")
fig.savefig(BASE_DIR / OUTPUT_PNG, dpi=300, bbox_inches="tight")

plt.show()

print(f"Saved: {OUTPUT_PDF}")
print(f"Saved: {OUTPUT_PNG}")

