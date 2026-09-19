# Truncation and Variable Orders across Tensor Networks and Decision Diagrams

## *Introduction*
Tensor networks and tractable circuits are two data structures for the same objects:
high-dimensional functions with real or complex weights. Physicists use matrix product states
(tensor trains) and tree tensor networks; the verification and AI communities use decision diagrams
and other circuits studied in knowledge compilation. Recent work [a] proves that these formalisms coincide, up to translations with polynomial (often linear) overhead: tensor trains correspond exactly
to non-deterministic edge-valued decision diagrams (EVDDs), and tree tensor networks to
structured-decomposable circuits with edge weights. Because the translations are constructive,
techniques developed on one side become available on the other — but several such transfers are
still open. Tensor networks are routinely compressed by truncating small singular values, yet what
this truncation means on the decision diagram side, and how it relates to approximate diagram
algorithms that come with rigorous guarantees, is not known. Conversely, decision diagrams enjoy
canonical forms and a mature toolbox of variable reordering heuristics, whereas the tensor train
community resorts to singular value decomposition and hand-tuned contraction orders.
## *Project Goal*
The student will take one or two of these open cross-fertilisation problems and settle them
constructively. A first track is approximation: implement singular-value truncation on the decision
diagram side via the translation of [a], characterise which reduction rule it corresponds to, and
measure the resulting size/accuracy trade-off against existing approximate decision diagram
methods on quantum state benchmarks. A second track is ordering: port decision diagram
reordering heuristics such as sifting to the choice of variable order or vtree of a tensor train or tree
tensor network, and measure the effect on bond dimension [b]. Either track combines a proof
obligation (what exactly does the transferred operation compute?) with an implementation and an
experimental evaluation.
[a] https://arxiv.org/abs/2605.00106
[b] https://epubs.siam.org/doi/10.1137/23M161286X
