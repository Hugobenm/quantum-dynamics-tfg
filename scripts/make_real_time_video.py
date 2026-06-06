"""Create a real-time density animation from a saved stationary state.

The typical use case is to load a state relaxed at E=0 and then switch on an
electric field during real-time propagation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gp2d_split_step import GPParameters, make_grid, normalize, split_step_real_time, x_wall_potential
from src.io_utils import load_npz
from src.plotting import animate_density


def _get_float(data: dict, *keys: str, default: float) -> float:
    for key in keys:
        if key in data:
            return float(data[key])
    return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time split-step evolution from a saved spinor state.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "data" / "example_states" / "ground_state_split_step_B1_g50_E0_Lx0.5.npz",
        help="Input .npz state. By default, a stationary E=0 state is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "videos" / "evolucion_tiempo_real.gif",
        help="Output animation path. Supported extensions depend on Matplotlib writers, e.g. .gif or .mp4.",
    )
    parser.add_argument("--g", type=float, default=None, help="Interaction strength used during real-time evolution.")
    parser.add_argument("--B", type=float, default=None, help="Magnetic field used during real-time evolution.")
    parser.add_argument(
        "--E",
        type=float,
        default=None,
        help="Electric field used during real-time evolution.",
    )
    parser.add_argument("--V0", type=float, default=None, help="Soft-wall potential height.")
    parser.add_argument("--wall-fraction-x", type=float, default=None, help="Soft-wall position as a fraction of Lx.")
    parser.add_argument("--wall-steepness", type=float, default=None, help="Soft-wall steepness.")
    parser.add_argument("--q", type=float, default=None, help="Effective charge.")
    parser.add_argument("--m", type=float, default=None, help="Effective mass.")
    parser.add_argument("--hbar", type=float, default=None, help="Reduced Planck constant in code units.")
    parser.add_argument("--Lx", type=float, default=None, help="Half-width of the x domain.")
    parser.add_argument("--Ly", type=float, default=None, help="Half-width of the y domain.")
    parser.add_argument("--N", type=int, default=None, help="Number of grid points per direction.")
    parser.add_argument("--dt", type=float, default=2e-3, help="Real-time step.")
    parser.add_argument("--steps", type=int, default=1500, help="Number of real-time steps.")
    parser.add_argument("--save-every", type=int, default=20, help="Store one animation frame every this many steps.")
    parser.add_argument("--interval", type=int, default=45, help="Frame interval in milliseconds.")
    parser.add_argument(
        "--component",
        choices=["total", "up", "down"],
        default="total",
        help="Density shown in the animation.",
    )
    parser.add_argument(
        "--renormalize",
        action="store_true",
        help="Renormalize after every real-time step. Usually unnecessary, but useful for long diagnostic runs.",
    )
    return parser


def density_frame(psi: np.ndarray, component: str) -> np.ndarray:
    if component == "up":
        return np.abs(psi[0]) ** 2
    if component == "down":
        return np.abs(psi[1]) ** 2
    return np.sum(np.abs(psi) ** 2, axis=0)


def main() -> None:
    args = build_parser().parse_args()

    state_path = args.state if args.state.is_absolute() else ROOT / args.state
    output = args.output if args.output.is_absolute() else ROOT / args.output

    data = load_npz(state_path)
    psi = np.asarray(data["psi"], dtype=complex)

    grid = make_grid(
        args.Lx if args.Lx is not None else _get_float(data, "Lx", default=5.0),
        args.Ly if args.Ly is not None else _get_float(data, "Ly", default=5.0),
        args.N if args.N is not None else int(data.get("N", 128)),
    )

    params = GPParameters(
        g=args.g if args.g is not None else _get_float(data, "g", default=50.0),
        B=args.B if args.B is not None else _get_float(data, "B", default=1.0),
        E=args.E if args.E is not None else _get_float(data, "E", default=0.0),
        q=args.q if args.q is not None else _get_float(data, "q", default=1.0),
        m=args.m if args.m is not None else _get_float(data, "m", default=1.0),
        hbar=args.hbar if args.hbar is not None else _get_float(data, "hbar", default=1.0),
        V0=args.V0 if args.V0 is not None else _get_float(data, "V0", "V_0", default=100.0),
        wall_fraction_x=(
            args.wall_fraction_x
            if args.wall_fraction_x is not None
            else _get_float(data, "wall_fraction_x", default=0.5)
        ),
        wall_steepness=(
            args.wall_steepness
            if args.wall_steepness is not None
            else _get_float(data, "wall_steepness", default=9.0)
        ),
    )
    potential = x_wall_potential(grid, params)

    frames = []
    for step in range(args.steps + 1):
        if step % args.save_every == 0:
            frames.append(density_frame(psi, args.component))
        psi = split_step_real_time(psi, grid, params, args.dt, potential=potential)
        if args.renormalize:
            psi = normalize(psi, grid)

    animation = animate_density(frames, grid, interval=args.interval)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, dpi=160)
    print(f"Loaded state: {state_path}")
    print(
        "Real-time parameters: "
        f"g={params.g}, B={params.B}, E={params.E}, V0={params.V0}, "
        f"wall_fraction_x={params.wall_fraction_x}"
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
