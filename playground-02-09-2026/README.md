# jax_hf_variational (playground bootstrap)

This folder bootstraps a standalone `jax_hf_variational` package from
`playground-02-06-2026/variational_hf.py` without a runtime dependency on
`jax_hf`.

## Layout

- `jax_hf_variational/variational_hf.py`: variational finite-T HF solver.
- `jax_hf_variational/utils.py`: local low-level utilities (`hermitize`,
  Fermi-Dirac, FFT self-energy, chemical-potential solve).
- `examples/repro_contimod_10_graphene_bilayer_linecuts_variational_sectortrack_plotly.py`: variational 1D density linecuts with sector labels.
- `examples/blg_phase_diagram_variational/repro_contimod_10_graphene_bilayer_phase_diagram_variational_sectortrack.py`: variational 2D (D, n) phase-diagram scaffold via sheet interpolation.

## Notes

- `VariationalHF` is **exchange-only by default** (`V_hartree_q=None` disables Hartree). To enable Hartree, pass an explicit `V_hartree_q` kernel when constructing the solver.

## Quick check

From this directory:

```bash
pip install -e .
python -c "import jax_hf_variational as v; print(v.__version__)"
```
