from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from .gp2d_split_step import (
    GPParameters,
    gaussian_spinor,
    make_grid,
    normalize,
    x_wall_potential,
    evolve_imaginary,
)
from .io_utils import load_npz, save_state
from .observables import chemical_potential, energy_functional


SeedKind = Literal["gaussian", "noise", "file"]


def make_initial_state(
    grid,
    seed: SeedKind = "gaussian",
    seed_file: str | Path | None = None,
    spin_down: bool = False,
    sigma: float = 2.5,
    noise_amplitude: float = 0.2,
    random_seed: int = 43,
) -> np.ndarray:
    """Build or load an initial spinor state."""
    if seed == "file":
        if seed_file is None:
            raise ValueError("seed_file must be provided when seed='file'.")
        data = load_npz(seed_file)
        return normalize(data["psi"].copy(), grid)

    if seed == "gaussian":
        return gaussian_spinor(grid, sigma=sigma, spin_down=spin_down)

    if seed == "noise":
        rng = np.random.default_rng(random_seed)
        psi = np.zeros((2, grid.N, grid.N), dtype=complex)
        noise = noise_amplitude * (rng.normal(size=(grid.N, grid.N)) + 1j * rng.normal(size=(grid.N, grid.N)))
        psi[0] = noise
        psi[1] = noise if spin_down else 0.0
        return normalize(psi, grid)

    raise ValueError(f"Unknown seed '{seed}'. Use 'gaussian', 'noise' or 'file'.")


def run_single_imaginary_time_state(
    output: str | Path,
    *,
    Lx: float = 5.0,
    Ly: float = 5.0,
    N: int = 128,
    g: float = 50.0,
    B: float = 1.0,
    E: float = 0.0,
    V0: float = 100.0,
    wall_fraction_x: float = 0.5,
    dtau: float = 3e-3,
    max_steps: int = 100_000,
    energy_tolerance: float = 1e-7,
    check_every: int = 100,
    seed: SeedKind = "gaussian",
    seed_file: str | Path | None = None,
    spin_down: bool = False,
    sigma: float = 2.5,
    noise_amplitude: float = 0.2,
    random_seed: int = 43,
) -> dict[str, object]:
    """Run one imaginary-time evolution and save the resulting state.

    This helper is the programmatic version of `scripts/run_single_state.py`.
    It returns the final state and diagnostics, and also writes an `.npz` file.
    """
    grid = make_grid(Lx=Lx, Ly=Ly, N=N)
    params = GPParameters(
        g=g,
        B=B,
        E=E,
        V0=V0,
        wall_fraction_x=wall_fraction_x,
    )
    potential = x_wall_potential(grid, params)
    psi0 = make_initial_state(
        grid,
        seed=seed,
        seed_file=seed_file,
        spin_down=spin_down,
        sigma=sigma,
        noise_amplitude=noise_amplitude,
        random_seed=random_seed,
    )

    energy, psi, energies, checked_steps = evolve_imaginary(
        psi0,
        grid,
        params,
        energy_functional,
        dtau=dtau,
        max_steps=max_steps,
        energy_tolerance=energy_tolerance,
        check_every=check_every,
        potential=potential,
    )
    mu = chemical_potential(psi, grid, params, potential)
    norm = float(np.sum(np.abs(psi) ** 2) * grid.dx * grid.dy)
    last_delta_energy = float(abs(energies[-1] - energies[-2])) if len(energies) > 1 else np.nan

    save_state(
        output,
        psi,
        x=grid.x,
        y=grid.y,
        Lx=grid.Lx,
        Ly=grid.Ly,
        N=grid.N,
        dx=grid.dx,
        dy=grid.dy,
        g=params.g,
        B=params.B,
        E=params.E,
        V_0=params.V0,
        wall_fraction_x=params.wall_fraction_x,
        energy=energy,
        mu=mu,
        norm=norm,
        dtau=dtau,
        max_steps=max_steps,
        energy_tolerance=energy_tolerance,
        check_every=check_every,
        checked_steps=checked_steps,
        energies=energies,
        last_delta_energy=last_delta_energy,
        seed=seed,
        seed_file="" if seed_file is None else str(seed_file),
        spin_down=spin_down,
        sigma=sigma,
        noise_amplitude=noise_amplitude,
        random_seed=random_seed,
    )

    return {
        "psi": psi,
        "grid": grid,
        "params": params,
        "energy": energy,
        "mu": mu,
        "norm": norm,
        "energies": energies,
        "checked_steps": checked_steps,
        "last_delta_energy": last_delta_energy,
        "output": Path(output),
    }

