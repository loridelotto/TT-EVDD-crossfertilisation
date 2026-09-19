## State convention

A state on n qubits is a numpy array of `2**n` complex128 amplitudes with norm 1.

The amplitude of the bit string `x_0 x_1 ... x_{n-1}` is stored at index

```
i = sum_k  x_k * 2**(n-1-k)
```

so site 0 is the most significant bit: the top of the decision diagram, and the first site of the MPS.

In Python:

```python
import numpy as np

psi = np.array([1, 0, 0, 1], dtype=complex)   # Bell state, n = 2
psi /= np.linalg.norm(psi)

np.save("bell.npy", psi)
psi = np.load("bell.npy")
```