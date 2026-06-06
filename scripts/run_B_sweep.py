"""Regenerate a coarse magnetic-field sweep."""

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

    b_values = np.linspace(0.1, 1.0, 20)
    dtau = 3e-3
    tolerance = 1e-7

    seed_path = ROOT / "data" / "example_states" / "ground_state_split_step_B1_g50_E0_Lx0.5.npz"
    psi_v = normalize(load_npz(seed_path)["psi"], grid) if seed_path.exists() else gaussian_spinor(grid, spin_down=True)
    psi_w = gaussian_spinor(grid, spin_down=False)

    energy_v, energy_w = [], []
    mu_v, mu_w = [], []

    # Continue from high B to low B, then reverse to match the plotted array.
    for b_value in b_values[::-1]:
        params.B = float(b_value)
        energy, psi_v, _, _ = evolve_imaginary(
            psi_v, grid, params, energy_functional, dtau=dtau,
            max_steps=100_000, energy_tolerance=tolerance, check_every=100,
            potential=potential,
        )
        energy_v.append(energy)
        mu_v.append(chemical_potential(psi_v, grid, params, potential))

        energy, psi_w, _, _ = evolve_imaginary(
            psi_w, grid, params, energy_functional, dtau=dtau,
            max_steps=100_000, energy_tolerance=tolerance, check_every=100,
            potential=potential,
        )
        energy_w.append(energy)
        mu_w.append(chemical_potential(psi_w, grid, params, potential))
        print(f"B={b_value:.3f}: E_v={energy_v[-1]:.8f}, E_w={energy_w[-1]:.8f}")

    out = ROOT / "results" / "sweeps"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "B_sweep_regenerated.npz",
        B_array=b_values,
        energy_v=np.asarray(energy_v[::-1]),
        energy_w=np.asarray(energy_w[::-1]),
        mu_v=np.asarray(mu_v[::-1]),
        mu_w=np.asarray(mu_w[::-1]),
        g=params.g,
        V_0=params.V0,
    )


if __name__ == "__main__":
    main()

