# SparseRegression
This repository contains skeleton code for the experiments of the paper Sparse regression: Scalable algorithms and empirical performance (preprint available on arXiv).

We perform experiments in regression ('regression/') as well as classification ('classification/') settings. The experiment files vary on the nature of the design matrix - a Toeplitz matrix (default case) which satisfies mutual incoherence condition, a "hard" case not satisfying MIC and real-world data (available for download [here](http://cancergenome.nih.gov)) - and on whether the sparsity parameter is cross-validated or not. Within each file, different correlation, signal-to-noise ratios and problem sizes are tested.

Each file in 'regression/' and 'classification/' follows the same structure. We advise starting by 'regression/synthetic_problem.jl' for an example. The preamble defines the experimental setup and the methods to use. Six methods can be compared (Lasso, ENet, MCP, SCAD, [SubsetSelection](https://github.com/jeanpauphilet/SubsetSelection.jl) and [SubsetSelectionCIO](https://github.com/jeanpauphilet/SubsetSelectionCIO.jl)). The script should be provided with a non-negative integer corresponding to the experiment through the command
```julia
julia synthetic_problem.jl 0
```
The number provided will define the problem size and simulation number (each experiment is performed 10 times and results are averaged over simulations). The script reads this integer, take the corresponding parameter ('n', 'p', 'k_true', 'snr', 'ρ'), generates data, normalizes it and then successively computes the desired estimators. All results are saved into method-specific csv files. Analysis of the results are then done seperately.

Folders 'package/' and 'oa/' contain a wrapper function to perform the cross-validation of the subgradient and the cutting-plane algorithm respectively . In practice, 'oa/saddle_cio_cv.jl' computes the solution of both algorithms over a range of sparsity parameters (other regularization parameters being fixed).
