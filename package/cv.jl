using SubsetSelection, SubsetSelectionCIO

##############################################
##SOLVER FOR CROSS-VALIDATING k/λ
##############################################
"""
CV pipeline for the subgradient method
"""
function ss_cv_constraint(ℓ::LossFunction, kRange, Γmax, Y, X;
    holdout = .3,
    anticycling = false, averaging = true, gradUp = 10, γ = 1/sqrt(size(X,1)), maxIter = 150, δ = 1e-3 )

  Path = DataFrame(param=Float64[], indices=Array[], w=Array[],
            α=Array[], b=Float64[], γ=Float64[], Δt=Float64[], iter=Integer[], error=Float64[])

  n, p = size(X)

  train, val = split_data(n, holdout) #Split data into train and validation sets
  indInit = SubsetSelection.ind_init(Constraint(kRange[1]), p);
  αInit = SubsetSelection.alpha_init(ℓ, Y[train]) #Initialization

  for c in 1:length(kRange)
    k = kRange[c]
    γ0 = 1.*p / k / (maximum(sum(X[train,:].^2,2))*n)
    factor = 1.
    stop = false;

    for inner_epoch in 1:Γmax
      if !stop
        γ = factor*γ0

        tic()
        result = subsetSelection(ℓ, Constraint(k), Y[train], X[train, :],
            indInit = indInit, αInit=αInit,
            anticycling = anticycling, averaging = averaging, gradUp = gradUp, γ = γ, maxIter = maxIter, δ = δ)
        Δt_saddle=toc()
        w = result.w
        # w = SubsetSelectionCIO.recover_primal(ℓ, Y[train], X[train, result.indices], γ)

        valError = error(ℓ, Y[val], X[val, :], result.indices, w)
        push!(Path, [k, result.indices, w, result.α, result.b,
                  γ, Δt_saddle, result.iter, valError])

        αInit = result.α[:]
        indInit = result.indices[:]

        g_old = norm(grad_primal(ℓ, Y[train], X[train, result.indices], zeros(length(result.indices)), 2*γ))
        g_new = norm(grad_primal(ℓ, Y[train], X[train, result.indices], w, 2*γ))
        stop = (g_new / g_old < 1e-2)

        factor *= 2
      end
    end
  end

  return Path
end
