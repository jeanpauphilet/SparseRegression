# Pkg.update()

using DataFrames, CSV

using GLMNet, Distributions, Base.Test
using SubsetSelection
include("../data_construction.jl")
include("../oa/oa_cv.jl")

using RCall
# R"install.packages(\"ncvreg\", repos=\"https://cloud.r-project.org\")"
# R"install.packages(\"glmnet\", repos=\"https://cloud.r-project.org\")"

R"library(ncvreg)"
R"library(glmnet)"

EXPERIMENTS = DataFrame(prefix=["LL", "LM", "LH", "HL", "HM", "HH"],
                        snr = [√6., √1., √0.05, √6, √1., √0.05],
                        ρ = [.2, .2, .2, .7, .7, .7])

ℓ = SubsetSelection.OLS()

do_lasso = true; do_enet = true
do_saddle = true; do_cio = true
do_mcp = true; do_scad = true

ΔTmax = 60.

for ARG in ARGS
    array_num = parse(Int, ARG)

    iter_run = array_num % 10
    n_index = div(array_num, 10) + 1

    sense = mod(iter_run,2)

    for experiment in 1:3

        p = (experiment % 3 == 1) ? 20000 : ((experiment % 3 == 2) ? 10000 : 2000)
        k = (experiment % 3 == 1) ? 100 : ((experiment % 3 == 2) ? 50 : 10)

        nRange = (experiment % 3 == 1) ? collect(500:300:4700) : ((experiment % 3 == 2) ? collect(500:400:6500) : collect(500:700:13800))
        if sense > .5
            nRange = (experiment % 3 == 1) ? collect(4700:-300:500) : ((experiment % 3 == 2) ? collect(6500:-400:500) : collect(13800:-700:500))
        end

        if n_index <= length(nRange)
            prefix=string("HardFix_", EXPERIMENTS[:prefix][experiment], "_")
            snr = EXPERIMENTS[:snr][experiment]
            ρ = EXPERIMENTS[:ρ][experiment]


            MCP = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], rho= Real[], k=Int[],
                            lambda=Real[], gamma=Real[], time=Real[], A=Real[], MSE=Real[])
            SCAD = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], rho= Real[], k=Int[],
                            lambda=Real[], gamma=Real[], time=Real[], A=Real[], MSE=Real[])

            LASSO = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], rho= Real[], k=Int[],
                            lambda=[], time=Real[], A=Real[], MSE=Real[])
            ENET = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], rho= Real[], k=Int[],
                            lambda=[], alpha=Real[], time=Real[], A=Real[], MSE=Real[])

            SADDLE = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], rho= Real[], k=Int[],
                            gamma=Real[], time=Real[], cuts=Real[], A=Real[], MSE=Real[])
            CIO = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], rho= Real[], k=Int[],
                            gamma=Real[], time=Real[], Gap=Real[], cuts=Real[], A=Real[], MSE=Real[])


            for n in [nRange[n_index]]
                println("***********************")
                println("BUILDING DATA")
                println("***********************")
                @time X, Y, indices_true, w_true = hard_data_construction(ℓ, n+400, k, p, snr, ρ)

                normalization = sqrt.(sum(X.^2,1))/sqrt.(size(X,1))
                X = X./normalization

                train = 1:n
                val = (n+1):(n+400)

                println("***********************")
                println("Samples = ", n)
                println("Features = ", p)
                println("Sparsity = ", k)
                println("SNR = ", snr)
                println("ρ = ", ρ)
                println("***********************")

                x= X[train,:]; y = Y[train];

                println("***********************")
                println("R-GLMNET")
                println("***********************")

                tic()
                R"glmnet($x, $y, family=\"gaussian\")"
                Δt_0 = toc()

                println("***********************")
                println("R-MCP")
                println("***********************")
                if do_mcp
                    for gamma in logspace(log10(2), log10(4.5), 7)
                        tic()
                        R"mcp = ncvreg($x, $y, family=\"gaussian\", penalty=\"MCP\", gamma=$gamma, alpha=1)"
                        Δt_mcp = toc()
                        @rget mcp

                        col = size(mcp[:beta], 2)
                        while length(find(s->abs(s)>1e-8, mcp[:beta][2:end, col]))>k && col>=1
                            col = col - 1
                        end
                        λ = mcp[:lambda][col]
                        γ = mcp[:gamma]

                        k_mcp = length(find(s->abs(s)>1e-8, mcp[:beta][2:end, col]))
                        indices_mcp = find(s->abs(s)>1e-8, mcp[:beta][2:end, col])
                        w_mcp = mcp[:beta][2:end, col]; w_mcp = w_mcp[indices_mcp]

                        accuracy_mcp = length(intersect(indices_true, indices_mcp))/k*100
                        mse_mcp = error(ℓ, Y[val], X[val,:], indices_mcp, w_mcp)

                        push!(MCP, [iter_run, n, p, snr, ρ, k_mcp, λ, γ, Δt_mcp/Δt_0, accuracy_mcp, mse_mcp])
                    end
                    filename = string(prefix, "MCP", array_num, ".csv")
                    CSV.write(filename, MCP)
                end

                println("***********************")
                println("R-SCAD")
                println("***********************")
                if do_scad
                    for gamma in logspace(log10(3), log10(4.5), 7)
                        tic()
                        R"scad = ncvreg($x, $y, family=\"gaussian\", penalty=\"SCAD\", gamma=$gamma, alpha=1)"
                        Δt_scad = toc()
                        @rget scad

                        col = size(scad[:beta], 2)
                        while length(find(s->abs(s)>1e-8, scad[:beta][2:end, col]))>k && col>=1
                            col = col - 1
                        end
                        λ = scad[:lambda][col]
                        γ = scad[:gamma]

                        k_scad=length(find(s->abs(s)>1e-8, scad[:beta][2:end, col]))
                        indices_scad = find(s->abs(s)>1e-8, scad[:beta][2:end, col])
                        w_scad = scad[:beta][2:end, col]; w_scad = w_scad[indices_scad]

                        accuracy_scad = length(intersect(indices_true, indices_scad))/k*100
                        mse_scad = error(ℓ, Y[val], X[val,:],indices_scad, w_scad)

                        push!(SCAD, [iter_run, n, p, snr, ρ, k_scad, λ, γ, Δt_scad/Δt_0, accuracy_scad, mse_scad])
                    end
                    filename = string(prefix, "Scad", array_num, ".csv")
                    CSV.write(filename, SCAD)
                end

                println("***********************")
                println("Julia-LASSO")
                println("***********************")

                tic()
                l1 = glmnet(X[train,:], vec(y), intercept=false)
                Δt_lasso = toc()

                if do_lasso
                    col = size(l1.betas, 2)
                    while length(find(s->abs(s)>1e-8, l1.betas[:, col]))>k && col>=1
                        col = col - 1
                    end
                    λ = l1.lambda[col]

                    indices_lasso = find(s->abs(s)>1e-8, l1.betas[:, col])
                    w_lasso = l1.betas[:, col][indices_lasso]

                    accuracy_lasso = length(intersect(indices_true, indices_lasso))/k*100
                    mse_lasso = error(ℓ, Y[val], X[val,:], indices_lasso, w_lasso)

                    push!(LASSO, [iter_run, n, p, snr, ρ, k, λ, Δt_lasso, accuracy_lasso, mse_lasso])
                    filename = string(prefix, "Lasso", array_num, ".csv")
                    CSV.write(filename, LASSO)
                end
                println("***********************")
                println("Julia-ENET")
                println("***********************")
                if do_enet
                    for α in 0.1:.1:1
                        tic()
                        enet = glmnet(X[train,:], vec(y), intercept=false, alpha=α)
                        Δt_enet = toc()

                        col = size(enet.betas, 2)
                        while length(find(s->abs(s)>1e-8, enet.betas[:, col]))>k && col>=1
                            col = col - 1
                        end

                        λ = enet.lambda[col]
                        indices_enet = find(s->abs(s)>1e-8, enet.betas[:, col])
                        w_enet = enet.betas[:, col][indices_enet]

                        accuracy_enet = length(intersect(indices_true, indices_enet))/k*100
                        mse_enet = error(ℓ, Y[val], X[val,:], indices_enet, w_enet)
                        push!(ENET, [iter_run, n, p, snr, ρ, k, λ, α, Δt_enet/Δt_lasso, accuracy_enet, mse_enet])
                        filename = string(prefix, "Enet", array_num, ".csv")
                        CSV.write(filename, ENET)
                    end
                end

                println("***********************")
                println("Julia - SADDLE POINT RELAXATION and CUTTING PLANES")
                println("***********************")
                if do_saddle||do_cio
                  # Regularization
                  # subsetSelection(ℓ, Constraint(10), Y[1:10], X[1:10,:], maxIter=10)
                  γ0 = .5*p /k /(maximum(sum(X[train,:].^2,2))*n)
                  factor = 1.

                  stop = !(do_cio||do_saddle)
                  for inner_epoch in 1:20
                      if !stop
                          # γ = GammaRange[inner_epoch]/sqrt.(n)*k/log(p-k)
                          γ = factor*γ0
                          println("***********************")
                          println("Sparsity k = ", k)
                          println("Regularization γ = ", γ)
                          println("***********************")

                          indicesRelax=find(x-> x< k/size(X,2), rand(size(X,2)))
                          if do_saddle
                              tic()
                              result_saddle = subsetSelection(ℓ, Constraint(k), Y[train], X[train,:], γ=γ, maxIter=200)
                              Δt_saddle = toc()
                              indices_saddle = result_saddle.indices[:]
                              w_saddle = result_saddle.w
                              # w_saddle = SubsetSelectionCIO.recover_primal(ℓ, Y[train], X[train, indices_saddle], γ)

                              accuracy_saddle = length(intersect(indices_true, indices_saddle))/k*100
                              mse_saddle = error(ℓ, Y[val], X[val,:], indices_saddle, w_saddle)
                              push!(SADDLE, [iter_run, n, p, snr, ρ, k,
                                  γ, Δt_saddle/Δt_lasso, result_saddle.iter, accuracy_saddle, mse_saddle])
                              filename = string(prefix, "Saddle", array_num, ".csv")
                              CSV.write(filename, SADDLE)

                              indicesRelax=result_saddle.indices[:]

                              g_old = norm(grad_primal(ℓ, Y[train], X[train, indices_saddle], zeros(length(indices_saddle)), 2*γ))
                              g_new = norm(grad_primal(ℓ, Y[train], X[train, indices_saddle], w_saddle, 2*γ))
                              stop = (g_new / g_old < 1e-2)
                          end
                          if do_cio
                              indices_cio, w_cio, Δt_cio, status, Gap, cutCount = oa_formulation(ℓ, Y[train], X[train,:], k, γ,
                                    indices0=indicesRelax, ΔT_max=ΔTmax)

                              accuracy_cio = length(intersect(indices_true, indices_cio))/k*100
                              mse_cio = error(ℓ, Y[val], X[val,:], indices_cio, w_cio)
                              push!(CIO, [iter_run, n, p, snr, ρ, k,
                                  γ, Δt_cio/Δt_lasso, Gap, cutCount, accuracy_cio, mse_cio])
                              filename = string(prefix, "CIO", array_num, ".csv")
                              CSV.write(filename, CIO)

                              g_old = norm(grad_primal(ℓ, Y[train], X[train, indices_cio], zeros(length(indices_cio)), 2*γ))
                              g_new = norm(grad_primal(ℓ, Y[train], X[train, indices_cio], w_cio, 2*γ))
                              stop = (g_new / g_old < 1e-2)
                          end
                          factor *=2
                      end
                  end
                end
            end
        end
    end
end
