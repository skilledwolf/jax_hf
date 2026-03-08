import time

import jax
import jax.numpy as jnp


nk1, nk2, nb = 128, 128, 4
key = jax.random.PRNGKey(0)

# Case 1: (nk1, nk2, nb, nb), FFT on first two axes
x1 = jax.random.normal(key, (nk1, nk2, nb, nb)) + 1j * jax.random.normal(
    key, (nk1, nk2, nb, nb)
)


@jax.jit
def fft_outer(x):
    return jnp.fft.fftn(x, axes=(0, 1))


# Case 2: (nb, nb, nk1, nk2), FFT on last two axes
x2 = jnp.transpose(x1, (2, 3, 0, 1))


@jax.jit
def fft_inner(x):
    return jnp.fft.fftn(x, axes=(2, 3))


# Warmup
fft_outer(x1).block_until_ready()
fft_inner(x2).block_until_ready()

# Benchmark
n_runs = 1000

start = time.time()
for _ in range(n_runs):
    fft_outer(x1).block_until_ready()
t1 = time.time() - start

start = time.time()
for _ in range(n_runs):
    fft_inner(x2).block_until_ready()
t2 = time.time() - start

print(f"Time for (nk1, nk2, nb, nb) [outer axes]: {t1:.4f} s")
print(f"Time for (nb, nb, nk1, nk2) [inner axes]: {t2:.4f} s")
