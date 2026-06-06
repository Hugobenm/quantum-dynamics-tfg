from __future__ import annotations

import numpy as np

from .gp2d_split_step import GPParameters, Grid2D, x_wall_potential


def kinetic_hamiltonian(
    psi: np.ndarray,
    grid: Grid2D,
    params: GPParameters,
) -> np.ndarray:
    """Apply the minimally coupled kinetic Hamiltonian in Landau gauge."""
    psi_k = np.fft.fft2(psi, axes=(1, 2))
    dpsi_x = np.fft.ifft2(1j * grid.KX * psi_k, axes=(1, 2))
    dpsi_y = np.fft.ifft2(1j * grid.KY * psi_k, axes=(1, 2))
    laplacian = np.fft.ifft2(-(grid.KX**2 + grid.KY**2) * psi_k, axes=(1, 2))

    A_x = np.zeros_like(grid.X)
    A_y = params.B * grid.X
    a_dot_grad = A_x * dpsi_x + A_y * dpsi_y
    a_sq = A_x**2 + A_y**2

    return (
        -(params.hbar**2 / (2 * params.m)) * laplacian
        + 1j * (params.hbar * params.q / params.m) * a_dot_grad
        + (params.q**2 / (2 * params.m)) * a_sq * psi
        + params.q * params.E * grid.X * psi
    )


def zeeman_hamiltonian(psi: np.ndarray, params: GPParameters) -> np.ndarray:
    """Zeeman splitting for the two spin components."""
    out = np.zeros_like(psi)
    omega_b = params.q * params.B / params.m
    out[0] = -params.hbar * omega_b * psi[0] / 2
    out[1] = params.hbar * omega_b * psi[1] / 2
    return out


def energy_functional(
    psi: np.ndarray,
    grid: Grid2D,
    params: GPParameters,
    potential: np.ndarray | None = None,
) -> float:
    """Gross-Pitaevskii energy functional."""
    if potential is None:
        potential = x_wall_potential(grid, params)
    density = np.sum(np.abs(psi) ** 2, axis=0)
    h_linear = kinetic_hamiltonian(psi, grid, params) + zeeman_hamiltonian(psi, params) + potential * psi
    linear_energy = np.sum(np.conj(psi) * h_linear).real * grid.dx * grid.dy
    interaction_energy = 0.5 * params.g * np.sum(density**2) * grid.dx * grid.dy
    return float(linear_energy + interaction_energy)


def chemical_potential(
    psi: np.ndarray,
    grid: Grid2D,
    params: GPParameters,
    potential: np.ndarray | None = None,
) -> float:
    """Expectation value of the effective GP Hamiltonian."""
    if potential is None:
        potential = x_wall_potential(grid, params)
    density = np.sum(np.abs(psi) ** 2, axis=0)
    h_eff = (
        kinetic_hamiltonian(psi, grid, params)
        + zeeman_hamiltonian(psi, params)
        + potential * psi
        + params.g * density * psi
    )
    return float(np.sum(np.conj(psi) * h_eff).real * grid.dx * grid.dy)


def currents(
    psi: np.ndarray,
    grid: Grid2D,
    params: GPParameters,
) -> dict[str, np.ndarray]:
    """Compute orbital, spin and total currents."""
    psi_k = np.fft.fft2(psi, axes=(1, 2))
    kx_psi = np.fft.ifft2(grid.KX * psi_k, axes=(1, 2))
    ky_psi = np.fft.ifft2(grid.KY * psi_k, axes=(1, 2))

    psi_up, psi_down = psi[0], psi[1]
    up_conj, down_conj = np.conj(psi_up), np.conj(psi_down)
    A_x = np.zeros_like(grid.X)
    A_y = params.B * grid.X

    j_orb_x = (
        up_conj * (params.hbar * kx_psi[0] - params.q * A_x * psi_up)
        + down_conj * (params.hbar * kx_psi[1] - params.q * A_x * psi_down)
    ).real / params.m
    j_orb_y = (
        up_conj * (params.hbar * ky_psi[0] - params.q * A_y * psi_up)
        + down_conj * (params.hbar * ky_psi[1] - params.q * A_y * psi_down)
    ).real / params.m

    c_z = np.abs(psi_up) ** 2 - np.abs(psi_down) ** 2
    c_z_k = np.fft.fft2(c_z)
    j_spin_x = (params.hbar / (2 * params.m)) * np.fft.ifft2(1j * grid.KY * c_z_k).real
    j_spin_y = (params.hbar / (2 * params.m)) * np.fft.ifft2(-1j * grid.KX * c_z_k).real

    return {
        "rho": np.abs(psi_up) ** 2 + np.abs(psi_down) ** 2,
        "J_orb_x": j_orb_x,
        "J_orb_y": j_orb_y,
        "J_spin_x": j_spin_x,
        "J_spin_y": j_spin_y,
        "J_x": j_orb_x + j_spin_x,
        "J_y": j_orb_y + j_spin_y,
    }


def find_curve_crossings(x: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> list[tuple[float, float]]:
    """Find linear-interpolated crossings between two curves."""
    diff = y1 - y2
    crossings: list[tuple[float, float]] = []
    for i in range(len(x) - 1):
        if diff[i] == 0:
            crossings.append((float(x[i]), float(y1[i])))
        elif diff[i] * diff[i + 1] < 0:
            x0, x1 = x[i], x[i + 1]
            d0, d1 = diff[i], diff[i + 1]
            x_cross = x0 - d0 * (x1 - x0) / (d1 - d0)
            y_cross = y1[i] + (x_cross - x0) * (y1[i + 1] - y1[i]) / (x1 - x0)
            crossings.append((float(x_cross), float(y_cross)))
    return crossings

