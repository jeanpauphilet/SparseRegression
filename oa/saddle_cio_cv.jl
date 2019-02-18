using JuMP, CPLEX
using DataFrames, SubsetSelectionCIO, SubsetSelection

"""
CV pipeline for the discrete cutting plane approach
Uses the subgradient algorithm as a warm-start
Computes the estimator for decreasing values of k, reusing the same model from
  one value to another
"""
function oa_formulation_cv(ℓ::LossFunction, Y, X, kRange, γ;
        maxIter=200, ΔT_max=60, verbose=false, Gap=0e-3)

  n,p = size(X)

  kRange=unique(kRange) #Remove duplicates
  sort!(kRange, rev=true) #Sort in descending order

  saddle_path = DataFrame(k=Int[], gamma=Float64[],
            indices=Vector{Int64}[], w=Array{Float64}[],
            value=Float64[], time=Float64[], iter=Real[])

  cio_path = DataFrame(k=Int[], gamma=Float64[],
            indices=Vector{Int64}[], w=Array{Float64}[],
            value=Float64[], gap=Float64[], time=Float64[], cuts=Real[])

  k = kRange[1]

  #Warmstart
  tic()
  result = subsetSelection(ℓ, Constraint(k), Y, X, γ = γ, maxIter=maxIter, noImprov_threshold=maxIter)
  Δt_saddle=toc()
  s0 = zeros(p); s0[result.indices]=1
  c0, ∇c0 = SubsetSelectionCIO.inner_op(ℓ, Y, X, s0, γ)
  push!(saddle_path, [k, γ, result.indices, result.w, c0, Δt_saddle, maxIter])


  miop = Model(solver=CplexSolver(CPX_PARAM_EPGAP=Gap,
                CPX_PARAM_TILIM=ΔT_max, CPX_PARAM_SCRIND=1*verbose))

  # Optimization variables
  @variable(miop, s[j=1:p], Bin, start=s0[j])
  @variable(miop, t>=0, start=c0)

  # Objective
  @objective(miop, Min, t)

  # Constraints
  @constraint(miop, cardcon, sum(s)<=k)

  cutCount=1;
  @constraint(miop, t>= c0 + dot(∇c0, s-s0))

  # Outer approximation method for Convex Integer Optimization (CIO)
  function outer_approximation(cb)
    cutCount += 1
    c, ∇c = SubsetSelectionCIO.inner_op(ℓ, Y, X, getvalue(s), γ)
    @lazyconstraint(cb, t>=c + dot(∇c, s-getvalue(s)))
  end
  addlazycallback(miop, outer_approximation)

  t0 = time()
  status = solve(miop)
  Δt = getsolvetime(miop)
  indices = find(s->s>0.5, getvalue(s))
  w = SubsetSelection.recover_primal(ℓ, Y, X[:, indices], γ)

  push!(cio_path, [k, γ, indices, w,
          getobjectivevalue(miop),
          (status == :Optimal) ? 0. : 1 - JuMP.getobjbound(miop) /  getobjectivevalue(miop),
          Δt, cutCount])

  for k in kRange[2:end]
      println("Solve for k=", k)
      #Warmstart
      tic()
      result = subsetSelection(ℓ, Constraint(k), Y, X, γ = γ, maxIter = maxIter)
      Δt_saddle=toc()

      s0 = zeros(p); s0[result.indices]=1
      c0, ∇c0 = SubsetSelectionCIO.inner_op(ℓ, Y, X, s0, γ)
      push!(saddle_path, [k, γ, result.indices, result.w, c0, Δt_saddle, maxIter])

      @constraint(miop, sum(s) <= k)
      cutCount+=1;
      @constraint(miop, t>= c0 + dot(∇c0, s-s0))

      tic()
      status = solve(miop)
      Δt = toc()

      indices = find(s->s>0.5, getvalue(s))
      w = SubsetSelection.recover_primal(ℓ, Y, X[:, indices], γ)

      push!(cio_path, [k, γ, indices, w,
              getobjectivevalue(miop),
              (status == :Optimal) ? 0. : 1 - JuMP.getobjbound(miop) /  getobjectivevalue(miop),
              Δt, cutCount])
  end

  return saddle_path, cio_path
end
