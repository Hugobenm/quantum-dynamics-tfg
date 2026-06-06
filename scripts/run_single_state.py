"""Run one custom imaginary-time evolution.

Examples
--------
No-vortex Gaussian seed:

    python scripts/run_single_state.py --g 50 --B 1 --E 0 --output data/example_states/my_state.npz

Continue from a saved vortex state:

    python scripts/run_single_state.py --g 50 --B 1 --E 0.25 \
        --seed file --seed-file data/example_states/ground_state_split_step_B1_g50_E0_Lx0.5.npz \
        --output data/example_states/vortex_E025.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.run_utils import run_single_imaginary_time_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "example_states" / "custom_state.npz")

    parser.add_argument("--Lx", type=float, default=5.0)
    parser.add_argument("--Ly", type=float, default=5.0)
    parser.add_argument("--N", type=int, default=128)

    parser.add_argument("--g", type=float, default=100.0)
    parser.add_argument("--B", type=float, default=1.0)
    parser.add_argument("--E", type=float, default=0.0)
    parser.add_argument("--V0", type=float, default=100.0)
    parser.add_argument("--wall-fraction-x", type=float, default=0.8)

    parser.add_argument("--dtau", type=float, default=3e-3)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--energy-tolerance", type=float, default=1e-7)
    parser.add_argument("--check-every", type=int, default=100)

    parser.add_argument("--seed", choices=["gaussian", "noise", "file"], default="gaussian")
    parser.add_argument("--seed-file", type=Path, default=None)
    parser.add_argument("--spin-down", action="store_true", help="Populate the down component in Gaussian/noise seeds.")
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--noise-amplitude", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=43)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_single_imaginary_time_state(
        args.output,
        Lx=args.Lx,
        Ly=args.Ly,
        N=args.N,
        g=args.g,
        B=args.B,
        E=args.E,
        V0=args.V0,
        wall_fraction_x=args.wall_fraction_x,
        dtau=args.dtau,
        max_steps=args.max_steps,
        energy_tolerance=args.energy_tolerance,
        check_every=args.check_every,
        seed=args.seed,
        seed_file=args.seed_file,
        spin_down=args.spin_down,
        sigma=args.sigma,
        noise_amplitude=args.noise_amplitude,
        random_seed=args.random_seed,
    )

    print(f"Saved: {result['output']}")
    print(f"Energy: {result['energy']:.12g}")
    print(f"Chemical potential: {result['mu']:.12g}")
    print(f"Norm: {result['norm']:.12g}")
    print(f"Last |dE|: {result['last_delta_energy']:.3e}")
    print(f"Final checked step: {result['checked_steps'][-1]}")


if __name__ == "__main__":
    main()

