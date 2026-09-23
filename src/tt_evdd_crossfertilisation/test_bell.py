# src/tt_evdd_crossfertilisation/demo_bell.py
"""Bell state, end to end: build, check, draw."""

import numpy as np

from .evdd import from_state, size #,to_vector
from .draw_dd import to_tikz 


psi = np.array([1, 0, 0, 1], dtype=complex)
psi /= np.linalg.norm(psi)

dd, root = from_state(psi)

print("nodes      :", size(dd))
print("root edge  :", root)
#print("round trip :", np.allclose(to_vector(dd, root), psi))

for i in sorted(dd["level"]):
    print(f"  node {i}  level {dd['level'][i]}  "
          f"0-> {dd['edges_0'][i]}   1-> {dd['edges_1'][i]}")

print()
print(to_tikz(dd, root))
