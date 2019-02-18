# SparseRegression
This repository contains the code for the experiments of the paper Sparse regression: Scalable algorithms and empirical performance (preprint available on arXiv)

Folders 'package/' and 'oa/' contain a wrapper function to perform the cross-validation of the subgradient and the cutting-plane algorithm respectively (available as seperate packages, namely [SubsetSelection](https://github.com/jeanpauphilet/SubsetSelection.jl) and [SubsetSelectionCIO](https://github.com/jeanpauphilet/SubsetSelectionCIO.jl) ).

All codes follow the same structure (see 'regression/synthetic_problem.jl' for an example).
