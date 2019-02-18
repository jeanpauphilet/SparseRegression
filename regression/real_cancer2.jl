using DataFrames, CSV

##Import data set
filenameX = string("../cancer_data/X_test.txt")
Xtest = readdlm(filenameX, '\t')

##Import data set
filenameX = string("../cancer_data/X_train.txt")
X = readdlm(filenameX, '\t')

X = [X; Xtest]

##Normalize data
meanX = sum(X, 1)./size(X,1)
X[:,:] .-= meanX
stdX = sqrt.(sum(X.^2, 1)./size(X,1))
X[:,:] ./= stdX
X = [X ones(size(X,1))] #Add Bias

n,p = size(X)

#Save 20% for test and val
sizeVal = round(Int, n*.2)
test = (n-sizeVal+1):n
val = (n-2*sizeVal+1):(n-sizeVal)
train = 1:(n-2*sizeVal)

println("***********************")
println("Data Description")
println("***********************")
println("Samples = ", n)
println("Features = ", p)
println("***********************")

using GLMNet, Distributions, Base.Test
using SubsetSelection
include("../oa/saddle_cio_cv.jl")
include("../cv.jl")

using RCall
R"library(ncvreg)"
R"library(glmnet)"

include("../data_construction.jl")

ℓ = SubsetSelection.OLS()
ΔTmax = 60.

do_lasso = false; do_enet = false
do_saddle = false; do_cio = false
do_mcp = true; do_scad = true

# snr = √4
prefix = "CancerReg2"
k_true = 50

snrRange = sqrt.(logspace(log10(6),log10(.05),10))

for ARG in ARGS
    array_num = parse(Int, ARG)

    iter_run = array_num % 10
    snr_index = div(array_num, 10) + 1
    snr = snrRange[snr_index]

    #Melange des donnees
    X = X[randperm(n), :]


    indices_true, w_true = generate_estimator(p, k_true)
    Y = generate_output(ℓ, noisy_signal(X, indices_true, w_true, snr))

    # Regularization
    kRange = collect(20:5:170)


    LASSO = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], k=Int[],
                    lambda=Real[], time=Real[], TF=Real[], FF=Real[], vMSE=Real[], tMSE=Real[])
    ENET = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], k=Int[],
                    lambda=Real[], alpha=Real[], time=Real[], TF=Real[], FF=Real[], vMSE=Real[], tMSE=Real[])
    SADDLE = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], k=Int[],
                    gamma=Real[], time=Real[], TF=Real[], FF=Real[], vMSE=Real[], tMSE=Real[])
    CIO = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], k=Int[],
                    gamma=Real[], time=Real[], TF=Real[], FF=Real[], vMSE=Real[], tMSE=Real[])

    MCP = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], k=Int[],
                    lambda=Real[], gamma=Real[], time=Real[], TF=Real[], FF=Real[], vMSE=Real[], tMSE=Real[])
    SCAD = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], k=Int[],
                    lambda=Real[], gamma=Real[], time=Real[], TF=Real[], FF=Real[], vMSE=Real[], tMSE=Real[])


    println("***********************")
    println("Julia - LASSO")
    println("***********************")
    x= X[train,:]; y = Y[train];

    tic()
    l1 = glmnet(x, vec(y), intercept=false)
    Δt_lasso = toc()

    if do_lasso
        for k in kRange
          l1 = glmnet(X[train, :], vec(Y[train]), dfmax=k+1, intercept=false)
          col = size(l1.betas, 2)
          while length(find(s->abs(s)>1e-8, l1.betas[:, col]))>k && col>=1
              col -= 1
          end

          indices_lasso = find(s->abs(s)>1e-8, l1.betas[:, col])
          k_lasso1 = length(indices_lasso)
          w_lasso = l1.betas[:, col][indices_lasso]
          λ = l1.lambda[col]

          TF_lasso = length(intersect(indices_true, indices_lasso))
          FF_lasso = length(setdiff(indices_lasso, indices_true))
          vmse_lasso = error(ℓ, Y[val], X[val,:], indices_lasso, w_lasso)
          tmse_lasso = error(ℓ, Y[test], X[test,:], indices_lasso, w_lasso)

          if k_lasso1<k
            col += 1
            indices_lasso = find(s->abs(s)>1e-8, l1.betas[:, col])
            k_lasso2 = length(indices_lasso)
            w_lasso = l1.betas[:, col][indices_lasso]

            TF_lasso *= (k-k_lasso1); TF_lasso += (k_lasso2-k)*length(intersect(indices_true, indices_lasso)); TF_lasso /=(k_lasso2-k_lasso1)
            FF_lasso *= (k-k_lasso1); FF_lasso += (k_lasso2-k)*length(setdiff(indices_lasso, indices_true)); FF_lasso /=(k_lasso2-k_lasso1)
            vmse_lasso *= (k-k_lasso1); vmse_lasso += (k_lasso2-k)*error(ℓ, Y[val], X[val,:], indices_lasso, w_lasso); vmse_lasso /=(k_lasso2-k_lasso1)
            tmse_lasso *= (k-k_lasso1); tmse_lasso += (k_lasso2-k)*error(ℓ, Y[test], X[test,:], indices_lasso, w_lasso); tmse_lasso /=(k_lasso2-k_lasso1)
          end

          push!(LASSO, [iter_run, n, p, snr, k,
                    λ, Δt_lasso, TF_lasso, FF_lasso, vmse_lasso, tmse_lasso])

          filename = string(prefix, "Lasso", array_num, ".csv")
          CSV.write(filename, LASSO)
        end
    end


    if do_enet
        println("***********************")
        println("ENET")
        println("***********************")

        for α in 0.1:.05:1
            for k in kRange
                tic()
                enet = glmnet(X[train, :], vec(Y[train]), dfmax=k+1,
                                                intercept=false, alpha=α)
                Δt_enet = toc()

                col = size(enet.betas, 2)
                while length(find(s->abs(s)>1e-8, enet.betas[:, col]))>k && col>=1
                    col -= 1
                end

                indices_enet = find(s->abs(s)>1e-8, enet.betas[:, col])
                k_enet1 = length(indices_enet)
                w_enet = enet.betas[:, col][indices_enet]
                λ = enet.lambda[col]

                TF_enet = length(intersect(indices_true, indices_enet))
                FF_enet = length(setdiff(indices_enet, indices_true))
                vmse_enet = error(ℓ, Y[val], X[val,:], indices_enet, w_enet)
                tmse_enet = error(ℓ, Y[test], X[test,:], indices_enet, w_enet)

                if k_enet1 < k
                    col += 1

                    indices_enet = find(s->abs(s)>1e-8, enet.betas[:, col])
                    k_enet2 = length(indices_enet)
                    w_enet = enet.betas[:, col][indices_enet]

                    TF_enet *= (k-k_enet1); TF_enet += (k_enet2-k)*length(intersect(indices_true, indices_enet)); TF_enet /=(k_enet2 - k_enet1)
                    FF_enet *= (k-k_enet1); FF_enet += (k_enet2-k)*length(setdiff(indices_enet, indices_true)); FF_enet /=(k_enet2 - k_enet1)
                    vmse_enet *= (k-k_enet1); vmse_enet += (k_enet2-k)*error(ℓ, Y[val], X[val,:], indices_enet, w_enet); vmse_enet /=(k_enet2 - k_enet1)
                    tmse_enet *= (k-k_enet1); tmse_enet += (k_enet2-k)*error(ℓ, Y[test], X[test,:], indices_enet, w_enet); tmse_enet /=(k_enet2 - k_enet1)
                end
                push!(ENET, [iter_run, n, p, snr^2, k,
                        λ, α, Δt_enet/Δt_lasso, TF_enet, FF_enet, vmse_enet, tmse_enet])

                filename = string(prefix, "Enet", array_num, ".csv")
                CSV.write(filename, ENET)
            end
        end
    end
    println("***********************")
    println("R - GLMNET")
    println("***********************")

    tic()
    R"glmnet($x, $y, family=\"gaussian\")"
    Δt_0 = toc()


    if do_mcp
        println("***********************")
        println("R - MCP")
        println("***********************")
        for gamma in logspace(log10(2), log10(4.5), 7)
            for k in kRange

                tic()
                R"mcp = ncvreg($x, $y, family=\"gaussian\", dfmax = $k+1,
                                penalty=\"MCP\", gamma=$gamma, alpha=1, lambda.min=.001, returnX=FALSE)"
                Δt_mcp = toc()
                @rget mcp

                col = size(mcp[:beta], 2)
                while length(find(s->abs(s)>1e-8, mcp[:beta][2:end, col])) > k && col >=1
                    col -= 1
                end

                λ = mcp[:lambda][col]
                γ = mcp[:gamma]

                indices_mcp = find(s->abs(s)>1e-8, mcp[:beta][2:end, col])
                k_mcp1 = length(indices_mcp)
                w_mcp = mcp[:beta][2:end, col]; w_mcp = w_mcp[indices_mcp]

                TF_mcp = length(intersect(indices_true, indices_mcp))
                FF_mcp = length(setdiff(indices_mcp, indices_true))
                vmse_mcp = error(ℓ, Y[val], X[val,:], indices_mcp, w_mcp)
                tmse_mcp = error(ℓ, Y[test], X[test,:], indices_mcp, w_mcp)

                if k_mcp1 < k
                    col += 1
                    # col = min.(col, size(mcp[:beta], 2))
                    indices_mcp = find(s->abs(s)>1e-8, mcp[:beta][2:end, col])
                    k_mcp2 = length(indices_mcp)

                    w_mcp = mcp[:beta][2:end, col]; w_mcp = w_mcp[indices_mcp]

                    TF_mcp *= (k-k_mcp1); TF_mcp += (k_mcp2-k)*length(intersect(indices_true, indices_mcp)); TF_mcp /=(k_mcp2-k_mcp1)
                    FF_mcp *= (k-k_mcp1); FF_mcp += (k_mcp2-k)*length(setdiff(indices_mcp, indices_true)); FF_mcp /=(k_mcp2-k_mcp1)
                    vmse_mcp *= (k-k_mcp1); vmse_mcp += (k_mcp2-k)*error(ℓ, Y[val], X[val,:], indices_mcp, w_mcp);vmse_mcp /=(k_mcp2-k_mcp1)
                    tmse_mcp *= (k-k_mcp1); tmse_mcp += (k_mcp2-k)*error(ℓ, Y[test], X[test,:], indices_mcp, w_mcp); tmse_mcp /=(k_mcp2-k_mcp1)
                end

                push!(MCP, [iter_run, n, p, snr^2, k,
                        λ, γ, Δt_mcp/Δt_0, TF_mcp, FF_mcp, vmse_mcp, tmse_mcp])
                filename = string(prefix, "MCP", array_num, ".csv")
                CSV.write(filename, MCP)
            end
        end
    end


    if do_scad
        println("***********************")
        println("R - SCAD")
        println("***********************")
        for gamma in logspace(log10(3), log10(4.5), 7)
            for k in kRange
                tic()
                R"scad = ncvreg($x, $y, family=\"gaussian\", dfmax = $k+1,
                                penalty=\"SCAD\", gamma=$gamma, alpha=1, lambda.min=.001, returnX=FALSE)"
                Δt_scad = toc()
                @rget scad

                col = size(scad[:beta], 2)
                while length(find(s->abs(s)>1e-8, scad[:beta][2:end, col])) > k && col > 1
                    col -= 1
                end

                λ = scad[:lambda][col]
                γ = scad[:gamma]

                indices_scad = find(s->abs(s)>1e-8, scad[:beta][2:end, col])
                k_scad1 = length(indices_scad)
                w_scad = scad[:beta][2:end, col]; w_scad = w_scad[indices_scad]

                TF_scad = length(intersect(indices_true, indices_scad))
                FF_scad = length(setdiff(indices_scad, indices_true))
                vmse_scad = error(ℓ, Y[val], X[val,:], indices_scad, w_scad)
                tmse_scad = error(ℓ, Y[test], X[test,:], indices_scad, w_scad)

                if k_scad1 < k
                    col += 1
                    # col = min.(col, size(scad[:beta], 2))

                    indices_scad = find(s->abs(s)>1e-8, scad[:beta][2:end, col])
                    k_scad2 = length(indices_scad)
                    w_scad = scad[:beta][2:end, col]; w_scad = w_scad[indices_scad]

                    TF_scad *= (k-k_scad1); TF_scad += (k_scad2-k)*length(intersect(indices_true, indices_scad)); TF_scad /= (k_scad2-k_scad1)
                    FF_scad *= (k-k_scad1); FF_scad += (k_scad2-k)*length(setdiff(indices_scad, indices_true)); FF_scad /= (k_scad2-k_scad1)
                    vmse_scad *= (k-k_scad1); vmse_scad += (k_scad2-k)*error(ℓ, Y[val], X[val,:], indices_scad, w_scad); vmse_scad /= (k_scad2-k_scad1)
                    tmse_scad *= (k-k_scad1); tmse_scad += (k_scad2-k)*error(ℓ, Y[test], X[test,:], indices_scad, w_scad); tmse_scad /= (k_scad2-k_scad1)
                end

                push!(SCAD, [iter_run, n, p, snr^2, k,
                    λ, γ, Δt_scad/Δt_0, TF_scad, FF_scad, vmse_scad, tmse_scad])
                filename = string(prefix, "Scad", array_num, ".csv")
                CSV.write(filename, SCAD)
            end
        end
    end


    if do_saddle||do_cio
        println("***********************")
        println("Julia - SADDLE POINT RELAXATION and CIO")
        println("***********************")

        # Regularization
        kRange = collect(20:5:170)

        k0 = kRange[end]
        γ0 = .5 *p /k0 /(maximum(sum(X[train,:].^2,2))*n)
        factor = 1
        stop = false

        for inner_epoch in 1:20
            γ = factor*γ0

            saddle_path, cio_path = oa_formulation_cv(ℓ, Y[train], X[train,:], kRange, γ,
                    maxIter=200, ΔT_max=ΔTmax, verbose=false, Gap=0e-3)

            for col in 1:size(saddle_path, 1)
                indices_saddle = saddle_path[col, :indices]
                w_saddle = saddle_path[col, :w]

                TF_saddle= length(intersect(indices_true, indices_saddle))/k_true*100
                FF_saddle= length(setdiff(indices_saddle, indices_true))/saddle_path[col,:k]*100
                vmse_saddle = error(ℓ, Y[val], X[val,:], indices_saddle, w_saddle)
                tmse_saddle= error(ℓ, Y[test], X[test,:], indices_saddle, w_saddle)

                push!(SADDLE, [iter_run, n, p, snr^2, saddle_path[col, :k],
                            γ, saddle_path[col, :time]/Δt_lasso,
                            TF_saddle, FF_saddle, vmse_saddle, tmse_saddle])

                g_old = norm(grad_primal(ℓ, Y[train], X[train, :], zeros(p), 2*γ))/p
                g_new = norm(grad_primal(ℓ, Y[train], X[train, indices_saddle], w_saddle, 2*γ))/saddle_path[col, :k]
                stop = stop || (g_new / g_old < 1e-2)
            end
            filename = string(prefix, "Saddle", array_num, ".csv")
            CSV.write(filename, SADDLE)


            for col in 1:size(cio_path,1)
                indices_cio = cio_path[col, :indices]
                w_cio = cio_path[col, :w]

                TF_cio= length(intersect(indices_true, indices_cio))/k_true*100
                FF_cio= length(setdiff(indices_cio, indices_true))/cio_path[col,:k]*100
                vmse_cio = error(ℓ, Y[val], X[val,:], indices_cio, w_cio)
                tmse_cio = error(ℓ, Y[test], X[test,:], indices_cio, w_cio)

                push!(CIO, [iter_run, n, p, snr^2, cio_path[col,:k],
                        γ, cio_path[col,:time]/Δt_lasso, TF_cio, FF_cio, vmse_cio, tmse_cio])

                g_old = norm(grad_primal(ℓ, Y[train], X[train, :], zeros(p), 2*γ))/p
                g_new = norm(grad_primal(ℓ, Y[train], X[train, indices_cio], w_cio, 2*γ))/ cio_path[col,:k]
                stop = stop || (g_new / g_old < 1e-2)
            end
            filename = string(prefix, "CIO", array_num, ".csv")
            CSV.write(filename, CIO)

            factor *= 2
            if stop
                break
            end
        end
    end
end
