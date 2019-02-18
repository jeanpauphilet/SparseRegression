# Pkg.update()
using DataFrames, CSV

##Import data set
filenameX = string("../cancer_data/X_test.txt")
Xtest = readdlm(filenameX, '\t')

filenameY = string("../cancer_data/Y_test.txt")
Ytest = vec(readdlm(filenameY, '\t'))

##Import data set
filenameX = string("../cancer_data/X_train.txt")
X = readdlm(filenameX, '\t')

filenameY = string("../cancer_data/Y_train.txt")
Y = vec(readdlm(filenameY, '\t'))


#Split into test, val and train
sizeVal = floor(Int, 0.30*size(Y, 1))
sizeTrain = size(Y, 1) - sizeVal

train = 1:sizeTrain
val = (sizeTrain+1):(sizeTrain+sizeVal)

##Normalize data
meanX = sum(X, 1)./size(X,1)
X[:,:] .-= meanX
stdX = sqrt.(sum(X.^2, 1)./size(X,1))
X[:,:] ./= stdX
X = [X ones(size(X,1))] #Add Bias

Xtest[:,:] .-= meanX
Xtest[:,:] ./= stdX
Xtest = [Xtest ones(size(Xtest,1))]

println("***********************")
println("Data Description")
println("***********************")
println("Samples = ", size(X,1))
println("Features = ", size(X,2))
println("***********************")

using GLMNet, Distributions, Base.Test
using SubsetSelection
include("../cv.jl")

using RCall
R"library(ncvreg)"
R"library(glmnet)"

ℓ = SubsetSelection.L1SVM()
ΔTmax = 180.

do_lasso = false; do_enet = false
do_saddle = true; do_cio = true
do_mcp = false; do_scad = false

for ARG in ARGS
    iter_run = parse(Int, ARG)

    LASSO = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], rho= Real[], k=Int[],
                    time=Real[], TF=Real[], FF=Real[], vMSE=Real[], tMSE=Real[])
    ENET = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], rho= Real[], k=Int[],
                    alpha=Real[], time=Real[], TF=Real[], FF=Real[], vMSE=Real[], tMSE=Real[])
    SADDLE = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], rho= Real[], k=Int[],
                    gamma=Real[], time=Real[], TF=Real[], FF=Real[], vMSE=Real[], tMSE=Real[])
    CIO = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], rho= Real[], k=Int[],
                    gamma=Real[], time=Real[], TF=Real[], FF=Real[], vMSE=Real[], tMSE=Real[])

    MCP = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], rho= Real[], k=Int[],
                    time=Real[], TF=Real[], FF=Real[], vMSE=Real[], tMSE=Real[])
    SCAD = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], rho= Real[], k=Int[],
                    time=Real[], TF=Real[], FF=Real[], vMSE=Real[], tMSE=Real[])


    #Remelange des donnees d'apprentissage (train+val)
    n,p = size(X)
    melange = randperm(n)
    X = X[melange, :]; Y = Y[melange]

            println("***********************")
            println("Julia - LASSO")
            println("***********************")
                Y_transf = [(Y[train].<0) (Y[train].>0)]

                tic()
                l1 = glmnet(X[train, :], convert(Matrix{Float64}, Y_transf), GLMNet.Binomial(), intercept=false)
                Δt_lasso = toc()

            if do_lasso
                for col in 1:size(l1.betas, 2)
                  indices_lasso = find(s->abs(s)>1e-8, l1.betas[:, col])

                  k_lasso = length(indices_lasso)
                  w_lasso = l1.betas[:, col][indices_lasso]

                  vmse_lasso = error(ℓ, Y[val], X[val,:], indices_lasso, w_lasso)
                  tmse_lasso = error(ℓ, Ytest, Xtest, indices_lasso, w_lasso)

                  push!(LASSO, [iter_run, n, p, 0., 0., k_lasso,
                            Δt_lasso, 0., 0., vmse_lasso, tmse_lasso])
                end

                filename = string("LassoCancer", iter_run, ".csv")
                CSV.write(filename, LASSO)
            end

            println("***********************")
            println("ENET")
            println("***********************")
            if do_enet
                for α in 0.1:.1:1
                    tic()
                    enet = glmnet(X[train, :], convert(Matrix{Float64}, Y_transf), GLMNet.Binomial(), intercept=false, alpha=α)
                    Δt_enet = toc()

                    for col in 1:size(enet.betas, 2)
                      indices_enet = find(s->abs(s)>1e-8, enet.betas[:, col])

                      k_enet = length(indices_enet)
                      w_enet = enet.betas[:, col][indices_enet]

                      vmse_enet = error(ℓ, Y[val], X[val,:], indices_enet, w_enet)
                      tmse_enet = error(ℓ, Ytest, Xtest, indices_enet, w_enet)

                      push!(ENET, [iter_run, n, p, 0., 0., k_enet,
                                α, Δt_enet/Δt_lasso, 0., 0., vmse_enet, tmse_enet])
                    end
                end

                filename = string("EnetCancer", iter_run, ".csv")
                CSV.write(filename, ENET)
            end

            x= X[train,:]; y = 1*(Y[train].>0);

            println("***********************")
            println("R - GLMNET")
            println("***********************")
                tic()
                R"glmnet($x, $y, family=\"binomial\")"
                Δt_0 = toc()

            println("***********************")
            println("R - MCP")
            println("***********************")
            if do_mcp
                tic()
                R"mcp = ncvreg($x, $y, family=\"binomial\", penalty=\"MCP\", alpha=1, lambda.min=.001)"
                Δt_mcp = toc()
                @rget mcp

                for col in  1:size(mcp[:beta], 2)
                    k_mcp = length(find(s->abs(s)>1e-8, mcp[:beta][2:end, col]))
                    indices_mcp = find(s->abs(s)>1e-8, mcp[:beta][2:end, col])
                    w_mcp = mcp[:beta][2:end, col]; w_mcp = w_mcp[indices_mcp]

                    vmse_mcp = error(ℓ, Y[val], X[val,:], indices_mcp, w_mcp)
                    tmse_mcp = error(ℓ, Ytest, Xtest, indices_mcp, w_mcp)

                    push!(MCP, [iter_run, n, p, 0., 0., k_mcp,
                            Δt_mcp/Δt_0, 0., 0., vmse_mcp, tmse_mcp])
                end

                filename = string("MCPCancer", iter_run, ".csv")
                CSV.write(filename, MCP)
            end

            println("***********************")
            println("R - SCAD")
            println("***********************")
            if do_scad
                tic()
                R"scad = ncvreg($x, $y, family=\"binomial\", penalty=\"SCAD\", alpha=1, lambda.min=.001)"
                Δt_scad = toc()
                @rget scad

                for col in 1:size(scad[:beta], 2)
                    k_scad=length(find(s->abs(s)>1e-8, scad[:beta][2:end, col]))
                    indices_scad = find(s->abs(s)>1e-8, scad[:beta][2:end, col])
                    w_scad = scad[:beta][2:end, col]; w_scad = w_scad[indices_scad]

                    vmse_scad = error(ℓ, Y[val], X[val,:], indices_scad, w_scad)
                    tmse_scad = error(ℓ, Ytest, Xtest, indices_scad, w_scad)

                    push!(SCAD, [iter_run, n, p, 0., 0., k_scad,
                        Δt_scad/Δt_0, 0., 0., vmse_scad, tmse_scad])
                end
                filename = string("ScadCancer", iter_run, ".csv")
                CSV.write(filename, SCAD)
            end

            println("***********************")
            println("Julia - SADDLE POINT RELAXATION")
            println("***********************")
            if do_saddle||do_cio
                # Regularization
                subsetSelection(ℓ, Constraint(10), Y[1:10], X[1:10,:],maxIter=10)
                kRange = collect(20:5:170)

                for c in 1:length(kRange)
                    k = kRange[c]

                    indices_saddle = [];
                    stop = !do_cio

                    γ = 1. / (maximum(sum(X[train,:].^2,2))*n)

                    for inner_epoch in 1:20
                        tic()
                        result = subsetSelection(ℓ, Constraint(k), Y[train], X[train, :],
                            # indInit = indInit, αInit=αInit,
                            anticycling = false, averaging = true, γ = γ, maxIter = 100)
                        Δt_saddle=toc()

                        indices_saddle = result.indices
                        k_saddle = length(indices_saddle)

                        w_saddle = recover_primal(ℓ, Y[train], X[train, indices_saddle], γ)

                        vmse_saddle = error(ℓ, Y[val], X[val,:], indices_saddle, w_saddle)
                        tmse_saddle= error(ℓ, Ytest, Xtest, indices_saddle, w_saddle)
                        println(Ytest)
                        push!(SADDLE, [iter_run, n, p, 0., 0., k, γ,
                                  Δt_saddle/Δt_lasso, 0., 0., vmse_saddle, tmse_saddle])

                        filename = string("SaddleCancer", iter_run, ".csv")
                        CSV.write(filename, SADDLE)

                        g_old = norm(grad_primal(ℓ, Y[train], X[train, :], zeros(p), 2*γ))/p
                        g_new = norm(grad_primal(ℓ, Y[train], X[train, indices_saddle], w_saddle, 2*γ))/k
                        stop = (g_new / g_old < 1e-2)

                        if do_cio
                            println("***********************")
                            println("Julia - CIO")
                            println("***********************")
                            indices_cio, w_cio, Δt_cio, status, Gap, cutCount = oa_formulation(ℓ, Y[train], X[train, :], k, γ,
                                                                indices0=indices_saddle, ΔT_max=ΔTmax)

                            vmse_cio = error(ℓ, Y[val], X[val,:], indices_cio, w_cio)
                            tmse_cio = error(ℓ, Ytest, Xtest, indices_cio, w_cio)

                            push!(CIO, [iter_run, n, p, 0., 0., k, γ,
                                    Δt_cio/Δt_lasso, 0., 0., vmse_cio, tmse_cio])

                            filename = string("CIOCancer", iter_run, ".csv")
                            CSV.write(filename, CIO)

                            g_old = norm(grad_primal(ℓ, Y[train], X[train, :], zeros(p), 2*γ))/p
                            g_new = norm(grad_primal(ℓ, Y[train], X[train, indices_cio], w_cio, 2*γ))/k
                            stop = (g_new / g_old < 1e-2)
                        end

                        γ *= 2
                    end
                end
            end
end
