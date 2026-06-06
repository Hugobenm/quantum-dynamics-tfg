# Data Files

This directory contains small example states used by the notebooks.

The sweep files used in the critical-branch notebook are stored in `results/sweeps/`. They are included as reproducible examples, but in a larger project these files would typically be regenerated from `scripts/run_g_sweep.py` and `scripts/run_B_sweep.py` rather than tracked in Git.

Expected `.npz` state fields:

- `psi`: complex spinor array with shape `(2, Ny, Nx)`.
- `x`, `y`: one-dimensional grid arrays.
- `Lx`, `Ly`, `N`: grid metadata.
- `g`, `B`, `E`: dimensionless simulation parameters.
- Optional diagnostics: `energy`, `mu`, `dtau`, `energy_tolerance`.

