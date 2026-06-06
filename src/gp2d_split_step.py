from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid2D:
    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    KX: np.ndarray
    KY: np.ndarray
    dx: float
    dy: float
    Lx: float
    Ly: float
    N: int


@dataclass
class GPParameters:
    g: float = 50.0
    B: float = 1.0
    E: float = 0.0
    q: float = 1.0
    m: float = 1.0
    hbar: float = 1.0
    V0: float = 100.0
    wall_fraction_x: float = 0.5
    wall_steepness: float = 9.0


def make_grid(Lx: float = 5.0, Ly: float = 5.0, N: int = 128) -> Grid2D:
    """Build a periodic square grid and its Fourier wave numbers."""
    x = np.linspace(-Lx, Lx, N, endpoint=False)
    y = np.linspace(-Ly, Ly, N, endpoint=False)
    X, Y = np.meshgrid(x, y)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    return Grid2D(x=x, y=y, X=X, Y=Y, KX=KX, KY=KY, dx=dx, dy=dy, Lx=Lx, Ly=Ly, N=N)


def normalize(psi: np.ndarray, grid: Grid2D) -> np.ndarray:
    """Normalize the two-component spinor to unit total norm."""
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * grid.dx * grid.dy)
    if norm == 0:
        raise ValueError("Cannot normalize a zero wave function.")
    return psi / norm


def x_wall_potential(grid: Grid2D, params: GPParameters) -> np.ndarray:
    """One-dimensional soft-wall confinement along x."""
    return params.V0 * (
        1.0
        + np.tanh((np.abs(grid.X) - params.wall_fraction_x * grid.Lx) * params.wall_steepness)
    )


def gaussian_spinor(grid: Grid2D, sigma: float = 2.5, spin_down: bool = False) -> np.ndarray:
    """Smooth normalized initial condition."""
    psi = np.zeros((2, grid.N, grid.N), dtype=complex)
    psi[0] = np.exp(-(grid.X**2) / (2 * sigma**2))
    if spin_down:
        psi[1] = psi[0]
    return normalize(psi, grid)


def split_step_imaginary(
    psi: np.ndarray,
    grid: Grid2D,
    params: GPParameters,
    dtau: float,
    potential: np.ndarray | None = None,
) -> np.ndarray:
    """One Strang split-step step in imaginary time.

    The operator splitting is

        exp(-V dt/2) exp(-T_x dt/2) exp(-T_{y,A} dt)
        exp(-T_x dt/2) exp(-V dt/2).

    The magnetic field is represented in Landau gauge A=(0, Bx).
    """
    if potential is None:
        potential = x_wall_potential(grid, params)

    density = np.sum(np.abs(psi) ** 2, axis=0)
    scalar_potential = params.q * params.E * grid.X
    zeeman_up = -params.hbar * params.q * params.B / (2 * params.m)
    zeeman_down = params.hbar * params.q * params.B / (2 * params.m)
    common = potential + scalar_potential + params.g * density

    psi = psi.copy()
    psi[0] *= np.exp(-0.5 * dtau * (common + zeeman_up))
    psi[1] *= np.exp(-0.5 * dtau * (common + zeeman_down))

    tx_half = np.exp(-0.5 * dtau * (params.hbar**2 * grid.KX**2 / (2 * params.m)))
    psi_kx = np.fft.fft(psi, axis=2)
    psi = np.fft.ifft(psi_kx * tx_half, axis=2)

    tya = np.exp(-dtau * ((params.hbar * grid.KY - params.q * params.B * grid.X) ** 2 / (2 * params.m)))
    psi_ky = np.fft.fft(psi, axis=1)
    psi = np.fft.ifft(psi_ky * tya, axis=1)

    psi_kx = np.fft.fft(psi, axis=2)
    psi = np.fft.ifft(psi_kx * tx_half, axis=2)

    density = np.sum(np.abs(psi) ** 2, axis=0)
    common = potential + scalar_potential + params.g * density
    psi[0] *= np.exp(-0.5 * dtau * (common + zeeman_up))
    psi[1] *= np.exp(-0.5 * dtau * (common + zeeman_down))
    return normalize(psi, grid)


def split_step_real_time(
    psi: np.ndarray,
    grid: Grid2D,
    params: GPParameters,
    dt: float,
    potential: np.ndarray | None = None,
) -> np.ndarray:
    """One real-time split-step step using unitary phase factors."""
    if potential is None:
        potential = x_wall_potential(grid, params)

    density = np.sum(np.abs(psi) ** 2, axis=0)
    scalar_potential = params.q * params.E * grid.X
    zeeman_up = -params.hbar * params.q * params.B / (2 * params.m)
    zeeman_down = params.hbar * params.q * params.B / (2 * params.m)
    common = potential + scalar_potential + params.g * density

    psi = psi.copy()
    psi[0] *= np.exp(-0.5j * dt * (common + zeeman_up) / params.hbar)
    psi[1] *= np.exp(-0.5j * dt * (common + zeeman_down) / params.hbar)

    tx_half = np.exp(-0.5j * dt * (params.hbar**2 * grid.KX**2 / (2 * params.m)) / params.hbar)
    psi = np.fft.ifft(np.fft.fft(psi, axis=2) * tx_half, axis=2)

    tya = np.exp(-1j * dt * ((params.hbar * grid.KY - params.q * params.B * grid.X) ** 2 / (2 * params.m)) / params.hbar)
    psi = np.fft.ifft(np.fft.fft(psi, axis=1) * tya, axis=1)
    psi = np.fft.ifft(np.fft.fft(psi, axis=2) * tx_half, axis=2)

    density = np.sum(np.abs(psi) ** 2, axis=0)
    common = potential + scalar_potential + params.g * density
    psi[0] *= np.exp(-0.5j * dt * (common + zeeman_up) / params.hbar)
    psi[1] *= np.exp(-0.5j * dt * (common + zeeman_down) / params.hbar)
    return psi


def evolve_imaginary(
    psi: np.ndarray,
    grid: Grid2D,
    params: GPParameters,
    energy_function,
    dtau: float = 3e-3,
    max_steps: int = 100_000,
    energy_tolerance: float = 1e-7,
    check_every: int = 100,
    potential: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Evolve in imaginary time until the energy change is below a tolerance."""
    energies = []
    checked_steps = []
    for step in range(max_steps):
        psi = split_step_imaginary(psi, grid, params, dtau, potential=potential)
        if step % check_every == 0:
            energy = energy_function(psi, grid, params, potential)
            energies.append(energy)
            checked_steps.append(step)
            if len(energies) > 1 and abs(energies[-1] - energies[-2]) < energy_tolerance:
                return energy, psi, np.asarray(energies), np.asarray(checked_steps)
    return energies[-1], psi, np.asarray(energies), np.asarray(checked_steps)

