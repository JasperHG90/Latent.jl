module GMM

#= 
Gaussian Mixture Models (GMMs) in Julia
Written by: Jasper Ginn
Date: 12/6/20
=#

using Random, Plots, Logging, LinearAlgebra, ProgressMeter, Distributions
#logger = global_logger(SimpleLogger(stdout, Logging.Error)) # Change to Logging.Debug for detailed info

#= Expectation-maximization algorithm =#

#= 
E-step (computing the probabilities of a point belonging to each cluster)
=#

function E_step(X::Array{Float64}, π::Array{Float64}, μ::Array{Float64}, Σ::Array{Float64})::Array{Float64}
    #=
    Perform the E-step of the expectation-maximization (EM) algorithm for a GMM 
    :param X: matrix of dimensions (n x m) that contains the input data
    :param π: array of dimensions (k x 1) that contains the mixing proportions
    :param μ: array of dimensions (k x m) that contains the mixture component means  
    :param Σ: tensor of dimensions (k x m x m) that contains the mixture component covariance matrices
    :return: matrix Γ of dimensions (n x k) that contains the probabilities of clusters for the objects
    :details: During the E-step, we solve for the posterior distribution of p(t_i =k | x_i, θ).
        Here, θ is a vector containing all parameters of the GMM.

        1. Given some variational distribution q(t) and parameter vector θ, we can define

           L(θ, q) = KL( q(t) || p(t | X, θ) ). The variational distribution that minimizes
           the KL distance is p(t | X, θ) = Γ.

        2. We can compute gamma as:
            
            (1) Γ = p(t = k | X, θ) = p(X | t = k) p(t = k) / [Σ p(X | t = k) p(t = k)]

           Where:

            (2) p(X | t = k) = N(X | μ_k, Σ_k)
            (3) p(t = k) = π_k
            
           Hence, we have:

            (4) Γ = [π_k * N(X | μ_k, Σ_k)] / [∑^K π_j * N(X | μ_j, Σ_j)]
    =#
    # Number of subjects / variables 
    N, M = size(X)
    # Number of clusters 
    K = size(π)[1]
    @debug "Dimensions are N=$N, M=$M, K=$K ..."
    # Init gamma 
    Γ = zeros((N, K))
    # For each cluster, compute the E-step 
    for k ∈ 1:K
        # Get mixing proportion, μ_k, Σ_k 
        π_k = π[k]
        μ_k = μ[k, :]
        Σ_k = Σ[:, :, k]
        # Emit log 
        @debug "Performing E-step for cluster $k with following parameters ..." π_k μ_k Σ_k
        # Draw pdf from multivariate normal 
        # Multiply the likelihood times the prior 
        posterior = MvNormal(μ_k, Σ_k |> 
                                    Hermitian |> 
                                    Matrix) |> 
                    x -> pdf(x, X') |> 
                    x -> π_k .* x
        # Add to gamma 
        Γ[:, k] = posterior
    end;
    # Normalize to make proper distribution
    Γ ./= sum(Γ, dims=2)
    # Return 
    return Γ
end;

#= 
M-step (computing the values of the Gaussian parameters)
=# 

function M_step(X::Array{Float64}, Γ::Array{Float64})::Tuple{Array{Float64}, Array{Float64}, Array{Float64}}
    #=
    Perform the M-step of the EM algorithm 
    :param X: matrix of dimensions (n x m) that contains the input data
    :param Γ: matrix Γ of dimensions (n x k) that contains the probabilities of clusters for the objects
    :return: a 3-tuple containing (I) (k x m) mean vector μ, (II) (k x 1) mixing proportions vector π, and 
             (III) (k x m x m) covariance tensor Σ
    :details: We want to maximize π_k, μ_k and Σ_k given the formula:

        (1) Q(X, θ) = ∑^N ∑^K p(t=k | X, θ) ln[p(x | t=k, θ) * p(t=k)] =
                      ∑^N ∑^K γ_{ik} ln[N(X; μ_k, Σ_k)] + ln[π_k]

    This optimization problem uses the probabilities computed during the E-step (γ_nk)
    and is subject to the following constraints:

        (a) ∑^K π_k = 1 --> g(π_k) = ∑^K π_k - 1
        (b) π_k ≥ 0 ∀π_k

    Constraint (b) can be ignored in this problem for the sake of simplicity, which leaves:
    
        (2) Z(X, θ) = Q(X, θ) - λ[g(π_k)] =
            ∑^N ∑^K {γ_{ik} (ln[N(X; μ_k, Σ_k)] + ln[π_k])} - λ[∑^K π_k - 1]

    And we want: 
    
        (3) arg max_{π_k, μ_k, Σ_k} Z(X, θ)

    - Mixing components
    
    To obtain the derivative of the mixing proportions π_k, we solve:

        (4) ∂Z/∂π_k ∑^N {γ_nk * ln[π_k]} - λ(π_k - 1) =
            ∑^N ∂Z/∂π_k {γ_nk * ln[π_k]} - ∂Z/∂π_k λ(π_k - 1) =
            ∑^N γ_nk / π_k - λ

    We set this to 0 and solve for π_k

        (5) ∑^N γ_nk / π_k - λ = 0 -->
            1/π_k ∑^N γ_nk = λ -->
            π_k = ∑^N γ_nk / λ

    Notice that, if we multiply both sides with ∑^K, we have:

        (c) ∑^K π_k = 1
        (d) ∑^N ∑^k γ_nk = N (because sum across probabilities for each person equals 1, 
                              and summing this across subjects yields N)
        (c) 1 = N / λ --> λ = N
    
    Hence, if we substitute these results in (5), we get:

        (6) π_k = ∑^N γ_nk / N

    - Means 

    The means for each cluster are given by:

        μ_k = ∑^N γ_nk x_n / ∑^N γ_nk

    - Covariances

    The covariances for each cluster are given by:

        Σ_k = ∑^N γ_nk (x_n - μ_k) (x_n - μ_k)' / ∑^N γ_nk
    =#
    # Record dimensions
    N, M = size(X)
    _, K = size(Γ)
    @debug "Dimensions are N=$N, M=$M, K=$K ..."
    # Allocate memory 
    μ = zeros((K, M))
    Σ = zeros((M, M, K))
    μ_k = zeros((1, M))
    Σ_k = zeros((M, M))
    # Compute the column sum of Γ
    Γ_colsum = sum(Γ, dims=1)
    # Compute new mixing probabilities
    π = Γ_colsum / N
    # For each cluster, compute values 
    for k ∈ 1:K
        @debug "Performing M-step for cluster $k ..."
        # Compute means 
        μ_k = sum(Γ[:,k] .* X, dims=1) ./ Γ_colsum[k]
        @debug "Computed M-step for μ ..."
        # Compute sigma
        X_centered = X .- μ_k 
        Σ_k = (Γ[:,k] .* X_centered)' * X_centered ./ Γ_colsum[k]
        @debug "Computed M-step for Σ ..."
        # Emit 
        @debug "Obtained the following parameter estimates on the M-step for cluster $k ..." round.(π[k], digits=2) round.(μ_k, digits=2) round.(Σ_k, digits=2)
        # Store 
        μ[k,:] = μ_k
        Σ[:,:,k] = Σ_k
    end;
    # Return 
    return π', μ, Σ
end;

#= 
Loss function
=#

function L(X::Array{Float64}, π::Array{Float64}, μ::Array{Float64}, Σ::Array{Float64}, Γ::Array{Float64})::Float64
    #=
    Compute the loss value given the current parameters 
    :param X: matrix of dimensions (n x m) that contains the input data
    :param π: array of dimensions (k x 1) that contains the mixing proportions
    :param μ: array of dimensions (k x m) that contains the mixture component means  
    :param Σ: tensor of dimensions (k x m x m) that contains the mixture component covariance matrices
    :param Γ: matrix of dimensions (n x k) that contains the probabilities of clusters for the objects
    :return: scalar, Loss value
    :details: Compute the loss for the likelihood of the data 

        (1) log[p(X)] = ∑^n log[∑^k π_k * N(x_n | μ_k, Σ_k)]

    =#
    # Get dimensions 
    N, M = size(X)
    _, K = size(Γ)
    @debug "Dimensions are N=$N, M=$M, K=$K ..."
    # Allocate 
    LL = zeros(N)
    # For each cluster 
    for k ∈ 1:K
        @debug "Computing log-likelihood for cluster $k ..."
        # Retrieve params for cluster k 
        μ_k = μ[k, :]
        Σ_k = Σ[:, :, k]
        π_k = π[k]
        # Compute likelihood 
        LL_k = MvNormal(μ_k, Σ_k |> 
                            Hermitian |> 
                            Matrix) |> 
                x -> pdf(x, X') |> 
                x -> π_k .* x
        # Add to running total 
        LL += LL_k
    end;
    # Apply logarithm and sum 
    LL = sum(log.(LL))
    @debug "Log-likelihood is $(round(LL, digits=3)) ..."
    # Return 
    return LL
end;

#= 
Initialization of parameters 
=#

function init(M::Int64, K::Int64)::Tuple{Array{Float64}, Array{Float64}, Array{Float64}}
    #=
    Initialize parameter vectors/matrices π_k, μ_k and Σ_k 
    :param N: Number of subjects
    :param M: Number of variables 
    :param K: Number of clusters
    :return: tuple containing π0, μ0 and Σ0
    =#
    # Allocate 
    π0 = zeros((K,))
    μ0 = rand(Float64, (K, M))
    Σ0 = zeros((M, M, K))
    @debug "Randomly initialized μ0 for all clusters..." μ0
    # For each cluster, init parameters 
    for k ∈ 1:K 
        # Draw π_k:
            # 1) from uniform distribution with a = 0 and b = sum(π0) if k < K
            # 2) As the remainder of 1 - sum(π)
        π0[k] = k < K ? Uniform(0, 1 - sum(π0)) |> rand : 1 - sum(π0)
        # Initialize Sigma 
        # NB: must be symmetric + positive semidefinite
        Σ0[:, :, k] = Matrix(I, (M, M)) 
        # Assert that covariance matrix is symmetric 
        @debug "Initialized π0 and Σ0 on cluster $k ..." π0[k] Σ0[:, :, k]
        @assert Σ0[:, :, k] == Σ0[:, :, k]' "Initial covariance matrix on cluster $k is not symmetric ..."
        # Assert that covariance matrix is positive semidefinite 
        @assert (eigen(Σ0[:, :, k]).values .> 0) |> all "Initial covariance matrix on cluster $k not positive semidefinite ..."
    end;
    @debug "π0 sums to $(sum(π0)) ..."
    # Assert that π0 sums to 1 
    @assert sum(π0) == 1 "Initial mixing proportions do not sum exactly to 1 ..."
    # Return 
    return π0, μ0, Σ0
end;

#=
Training loop for GMM using EM algorithm
=#

function train_GMM(X::Array{Float64}, K::Int64, ϵ::Float64=1e-3, maxiter::Int64=100, epochs::Int64=10)::Tuple{Array{Float64}, Array{Float64}, Array{Float64}, Float64, Array{Float64}}
    #=
    Implement the training loop for the GMM using EM
    :param X: matrix of dimensions (n x m) that contains the input data
    :param K: number of clusters 
    :param ϵ: delta value between current and last loss. If delta > ϵ the program will exit
    :param maxiter: maximum number of iterations in an epoch
    :param epochs: number of times we run the EM algorithm using different starting values
    :return: tuple containing:
        1. π_best (k x 1) 
        2. μ_best (k x m)
        3. Σ_best (m x m x k)
        4. loss_best 
        5. loss_history (maxiter x epochs)
    =#
    # Get dimensions 
    N, M = size(X)
    # Allocate
    π_best = zeros((K,))
    μ_best = zeros((K, M))
    Σ_best = zeros((K, M, M))
    Γ = zeros((N, K))
    Σ = zeros((M, M, K))
    μ = zeros((K, 1))
    loss_δ = 0.0
    # Best los 
    loss_best = NaN
    # Loss history
    LH = zeros((maxiter, epochs))
    # Run epochs 
    @showprogress "Fitting GMM ..." for epoch ∈ 1:epochs
        @debug "Started EM algorithm for epoch $epoch ..."
        loss_previous = NaN
        loss_current = 0.0
        # Initialize means etc. 
        π, μ, Σ = init(M, K)
        # Run iterations 
        for iter ∈ 1:maxiter
            @debug "Started iteration $iter on epoch $epoch ..."
            # Perform E-step 
            try
                Γ = E_step(X, π, μ, Σ)
            catch e
                @error "Encountered $e error while computing E-step ..."
                break
            end;
            @debug "Finished E-step ..."
            # Perform M-step 
            π, μ, Σ = M_step(X, Γ)
            @debug "Finished M-step ..."
            # Compute loss 
            try
                loss_current = L(X, π, μ, Σ, Γ)
            catch e
                @error "Encountered $e error while computing loss ..."
                break
            end;
            @debug "Computed log-likelihood ..."
            # If best loss NaN 
            if isnan(loss_best)
                LH[iter, epoch] = loss_best = loss_previous = loss_current
                π_best, μ_best, Σ_best = π, μ, Σ
                continue
            end;
            # If previous loss NaN
            if isnan(loss_previous)
                LH[iter, epoch] = loss_previous = loss_current
                continue
            end;
            # Assert that current loss is not smaller than previous loss 
            @assert loss_current > loss_previous "Loss is decreasing. There must be a bug in your code somewhere \\_(:/)_/ ..."
            # Store parameters if loss current beats loss best 
            if loss_current > loss_best
                loss_best = loss_current
                π_best, μ_best, Σ_best = π, μ, Σ
            end;
            # Check delta in losses 
            loss_δ = -1 * ((loss_current - loss_previous) / loss_previous) 
            if loss_δ ≤ ϵ
                @debug "Stopping condition reached. Exiting epoch $epoch ..."
                break
            end;
            # Set current loss as previous loss and add to history
            LH[iter, epoch] = loss_previous = loss_current
            @debug "Finished iteration $iter on epoch $epoch ..."
        end;
        @debug "Finished epoch $epoch ..."
    end;
    @debug "Finished program. Exiting now ..."
    # Return best values 
    return π_best, μ_best, Σ_best, loss_best, LH
end;

#= 
Wrapper functions / utility functions 
=#

"""
    clust(X, K, ϵ=1.e-3, maxiter=100, epochs=10)::Tuple{Array{Float64}, Array{Float64}}

Fit a Gaussian Mixture Model (GMM) on the input data X using EM estimation.

# Arguments
- `X::Array{Float64}`: 
- `K::Int64`: Number of clusters
- `ϵ::Float64=1e-3`: Value that governs the stopping condition. If the delta loss is smaller than this value, then quit the program. 
- `maxiter::Int64=100`: Maximum number of iterations the algorithm will execute in each epoch.
- `epochs::Int64=10`: Number of epochs the algorithm will run.

# Returns
A tuple containing 
- Cluster labels. This is an array of dimensions (N x 1)
- Loss history for each epoch. This is an array of dimensions (maxiter x epochs)

# Examples
```julia-repl

```
"""
function clust(X::Array{Float64}, K::Int64; ϵ::Float64=1e-3, maxiter::Int64=100, epochs::Int64=10)::Tuple{Array{Float64}, Array{Float64}}
    # Record dimensions
    N, M = size(X)
    # Run algorithm
    π1, μ1, Σ1, L1, history = train_GMM(X, K, ϵ, maxiter, epochs)
    # Compute labels using best values 
    lbls = [i for i in mapslices(argmax, E_step(X, π1, μ1, Σ1), dims=2)[:]]
    # Return history and labels
    return lbls, history
end;

function plot_history(history::Array{Float64})
    # Plot loss
    maxiter, epochs = size(history)
    hc = history[:,1]; 
    plot(hc[hc .< 0], legend = false)
    # For each epoch, plot 
    for epoch ∈ 1:(size(history)[2]-1)
        hc = history[:,epoch]
        plot!(hc[hc .< 0])
    end;
    hc = history[:,end]; 
    p = plot!(hc[hc .< 0], title = "Change in loss for $epochs epochs")
    return p
end;

"""
    simulate_gmm(K::Int64, N::Int64, μ::Array{Float64}, Σ::Array{Float64})::Tuple{Array{Float64}, Array{Float64}}

Simulate a dataset that is a mixture of K multivariate Gaussians

# Arguments 
- `K::Int64`: Number of clusters 
- `N::Array{Int64}`: Array of dimensions (1 x K) containing the number of data points for each cluster.
- `μ::Array{Float64}`: Array of dimensions (K x M) containing the M means for each cluster.
- `Σ::Array{Float64}`: Array of dimensions (M x M X K) containing the K M X M (co)variances. These matrices must be symmetric and positive semi-definite.

# Returns 
An array of dimensions (∑N x M) containing the data points.

# Examples
```julia-repl
julia>
```
"""
function simulate_GMM(K::Int64, N::Array{Int64}, μ::Array{Float64}, Σ::Array{Float64})::Tuple{Array{Float64}, Array{Float64}}
    # Assert dimensions 
    @assert K == size(Σ)[3] "You must pass a covariance matrix for each separate cluster ..."
    @assert K == size(μ)[1] "You must pass a mean array for each separate cluster ..."
    @assert K == size(N)[2] "You must pass a sample size (N) for each separate cluster ..."
    # Assert that dim(μ) and dim(Σ) are equal in the number of columns 
    @assert size(Σ)[2] == size(μ)[2]
    # Dimensions 
    N_total = sum(N)
    M = size(Σ)[1]
    # Open up result matrix 
    X = zeros((N_total, M))
    lbls = zeros((N_total,1))
    # For each cluster, generate data 
    start_idx = 1
    for k ∈ 1:K
        # Dims 
        N_k = N[k]
        μ_k = μ[k,:]
        Σ_k = Σ[:,:,k]
        # Create end idx 
        end_idx = k == 1 ? N_k : start_idx + N_k - 1
        # Draw points 
        X[start_idx:end_idx, :] = MvNormal(μ_k, Σ_k) |> 
            x -> rand(x, N_k)';
        # Labels 
        lbls[start_idx:end_idx, :] = fill(k, N_k)
        # Overwrite start values 
        if k < K
            start_idx = N_k + start_idx
        end;
    end;
    # Cat X and labels
    Xtmp = hcat(X, lbls);
    # Shuffle
    Xtmp = Xtmp[shuffle(1:end), :];
    # Return X and labels 
    return Xtmp[:, 1:M], Xtmp[:, end]
end;

## End module
end;