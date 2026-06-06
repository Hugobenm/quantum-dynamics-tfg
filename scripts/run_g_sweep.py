"""Regenerate a coarse interaction-strength sweep.

This script is intentionally conservative: it demonstrates the workflow used
in the notebooks, while the included `results/sweeps/*.npz` files provide
ready-to-plot data for the GitHub demo.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gp2d_split_step import GPParameters, gaussian_spinor, make_grid, normalize, x_wall_potential, evolve_imaginary
from src.io_utils import load_npz
from src.observables import energy_functional, chemical_potential


def main() -> None:
    grid = make_grid(Lx=5, Ly=5, N=128)
    params = GPParameters(g=50, B=1, E=0, V0=100, wall_fraction_x=0.5)
    potential = x_wall_potential(grid, params)

    g_values = np.linspace(50, 1, 25)
    dtau = 3e-3
    tolerance = 1e-7

    seed_path = ROOT / "data" / "example_states" / "ground_state_split_step_B1_g50_E0_Lx0.5.npz"
    if seed_path.exists():
        psi_v = normalize(load_npz(seed_path)["psi"], grid)
    else:
        psi_v = gaussian_spinor(grid, spin_down=True)
    psi_w = gaussian_spinor(grid, spin_down=False)

    energy_v, energy_w = [], []
    mu_v, mu_w = [], []
    steps_v, steps_w = [], []
    psi_v_array, psi_w_array = [], []

    for g_value in g_values:
        params.g = float(g_value)
        energy, psi_v, energies, steps = evolve_imaginary(
            psi_v, grid, params, energy_functional, dtau=dtau,
            max_steps=100_000, energy_tolerance=tolerance, check_every=100,
            potential=potential,
        )
        energy_v.append(energy)
        mu_v.append(chemical_potential(psi_v, grid, params, potential))
        steps_v.append(steps[-1])
        psi_v_array.append(psi_v.copy())

        energy, psi_w, energies, steps = evolve_imaginary(
            psi_w, grid, params, energy_functional, dtau=dtau,
            max_steps=100_000, energy_tolerance=tolerance, check_every=100,
            potential=potential,
        )
        energy_w.append(energy)
        mu_w.append(chemical_potential(psi_w, grid, params, potential))
        steps_w.append(steps[-1])
        psi_w_array.append(psi_w.copy())
        print(f"g={g_value:.3f}: E_v={energy_v[-1]:.8f}, E_w={energy_w[-1]:.8f}")

    out = ROOT / "results" / "sweeps"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "g_sweep_regenerated.npz",
        g_array=g_values,
        energy_v_g=np.asarray(energy_v),
        energy_w_g=np.asarray(energy_w),
        mu_v_g=np.asarray(mu_v),
        mu_w_g=np.asarray(mu_w),
        steps_v_g=np.asarray(steps_v),
        steps_w_g=np.asarray(steps_w),
        psi_v_g_array=np.asarray(psi_v_array),
        psi_w_g_array=np.asarray(psi_w_array),
        B_fixed=params.B,
        V_0=params.V0,
    )


if __name__ == "__main__":
    main()

