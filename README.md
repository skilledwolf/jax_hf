# jax_hf has moved

The JAX Hartree-Fock solvers now live in the **`hartreefock`** package,
together with the native C++ engine that used to be `cpp_hf`:
<https://github.com/skilledwolf/cpp_hf>.

```bash
pip install "hartreefock[jax] @ git+https://github.com/skilledwolf/cpp_hf"
```

```python
import hartreefock as hf

kernel = hf.HartreeFockKernel(weights, h, Vq, T=0.1)
result = hf.solve(kernel, P0, n_electrons, backend="jax")      # was jax_hf.solve
scf = hf.solve_scf(kernel, P0, n_electrons, backend="jax")     # was jax_hf.solve_scf
```

The JAX-only tools (RDIIS / Anderson SCF, stability analysis, checked
minimization) are in `hartreefock.backends.jax`. Results use the field names
`density`, `fock`, `mu` and `n_iter`. The released `jax-hf` wheels on PyPI
stay available for old scripts; the full history of this repository is in
its git log.
