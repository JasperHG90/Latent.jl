#=
Bayesian Gaussian Mixture Model
=#

module BGMM

using Random, Logging, LinearAlgebra, ProgressMeter, Distributions, Plots
logger = global_logger(SimpleLogger(stdout, Logging.Info)) # Change to Logging.Debug for detailed info

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
function simulate(K::Int64, N::Array{Int64}, μ::Array{Float64}, Σ::Array{Float64})::Tuple{Array{Float64}, Array{Float64}}
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

# Sample means from multivariate normal distribution
function sample_posterior_mean(X, κ0, Τ0, Γ, Σ)
    #=
    Compute the posterior means for a multivariate normal distribution 
    :param X: input data of shape (N x M). 
    :param κ0: prior values for the mean κ1. Array of shape (M x K).
    :param Τ0: prior values for the covariance matrix Τ1. Array of shape (M x M x K).
    :param Γ: cluster assignments. Array of shape (N x 1).
    :param Σ: known covariance matrix. Array of shape (M x M x K)
    :return: Array of (M x K) containing the conditional means.
    :seealso: https://en.wikipedia.org/wiki/Conjugate_prior
    :details: The posterior means are sampled from a multivariate normal distribution. 
        The parameters p(μ_ret | X, π, Γ, Σ) are obtained by multiplying the likelihood of
        the data times the prior, which is also a normal distribution with prior hyperparameters
        μ0 and Σ0. 
    =#
    M = size(X)[2]
    K = size(Σ)[3]
    # Sampled means 
    μ_ret = zeros((M, K))
    # For each cluster, sample from multivariate normal
    for k ∈ 1:K
        # Compute sample size in cluster k 
        Nk = length(Γ[Γ.==k]) |>
            x -> convert(Int64, x)
        # Initialize means    
        ϵ = zeros((M))
        # If sample size is 0, then just initialize with 0
        # Else compute sample means in cluster k
        if Nk > 0
            for m ∈ 1:M  
                ϵ[m] = mean(X[Γ.==k,m])
            end;
        end;
        # Invert the prior covariance and covariance matrices 
        Τ0_inv = inv(Τ0[:,:,k])
        Σ_inv = inv(Σ[:,:,k])
        # Solve for posterior covariance 
        Τ1 = inv(Τ0_inv .+ (Nk .* Σ_inv)) |> 
            Hermitian |> 
            Matrix
        # Solve for posterior mean 
        κ1 = Τ1 * (Τ0_inv * κ0[:,k] + Nk .* Σ_inv * ϵ)
        @debug "Posterior means κ1: $κ1 ; Posterior covariance matrix Τ1: $Τ1 ..."
        # Sample means from multivariate normal 
        μ_ret[:, k] = MvNormal(κ1, Τ1) |>
            x -> rand(x, 1)
    end;
    # Return means 
    return μ_ret
end;

# Sample posterior covariance matrices from an inverse-wishart distribution 
function sample_posterior_covariance(X, ν0, Ψ0, Γ, μ)
    #=
    Sample a covariance matrix given a known mean μ
    :param X: input data of shape (N x M).
    :param ν0: prior hypothesized sample size. Array of shape (K x 1).
    :param Ψ0: prior hypothesized sums-of-squares matrix. Array of shape (M x M x K).
    :param Γ: cluster assignments. Array of shape (N x 1).
    :param μ: known means. Array of shape (M x K).
    :return: Array of shape (M x M x K) containing the sampled covariance matrices.
    :seealso: https://en.wikipedia.org/wiki/Conjugate_prior
    :details: The posterior covariance matrices are sampled from an inverse-wishart 
        distribution. The prior hyperparameters are ν0, which can be interpreted as the
        hypothesized number of subjects in each cluster and Ψ0, which can be interpreted
        as the hypothesized sums-of-squares matrix. 
    =#
    N, M = size(X)
    K = size(ν0)[1]
    Σ_out = zeros((M, M, K))
    # Sample for each cluster 
    for k ∈ 1:K
        @debug "Sampling covariance matrix for cluster $k ..."
        # Compute sample size in cluster k 
        Nk = length(Γ[Γ.==k]) |>
            x -> convert(Int64, x)
        # Compute posterior hyperparameters 
        ν1 = Nk + ν0[k]
        # Subtract group mean from observations in cluster K
        X_k = zeros(size(X[Γ.==k,:]))
        for m ∈ 1:M
            X_k[:,m] = X[Γ.==k,m] .- μ[m,k]
        end;
        # Compute posterior sums of squares 
        Ψ1 = Ψ0[:,:,k] .+ X_k' * X_k
        # Draw from inverse wishart 
        Σ_out[:,:,k] = InverseWishart(ν1, Ψ1) |>
            x -> rand(x, 1)[1]
    end;
    # Return 
    return Σ_out
end;

# Sample cluster assignments Γ from a multinomial 
function sample_cluster_assignments(X, ζ, μ, Σ)
    #=
    Sample cluster assignments for each point in X 
    :param X: input data of shape (N x M). 
    :param ζ: mixing proportions of shape (K x 1).
    :param μ: known means. Array of shape (M x K).
    :param Σ: known covariance matrix. Array of shape (M x M x K).
    :return: cluster assignments Γ. Array of size (N x 1).
    :seealso:
    :detais:
    =#
    N, M = size(X)
    K = size(ζ)[1]
    # Compute the likelihood for each point X given the cluster parameters
    Ω = zeros((N, K))
    for k ∈ 1:K
        Ω[:,k] = MvNormal(μ[:,k], Σ[:,:,k]) |>
            x -> pdf(x, X')
    end;
    # Normalize (to turn into probabilities)
    Ω ./= sum(Ω, dims=2)
    # Sample cluster assignment for each row 
    Γ1 = zeros(Int64,(N))
    for n ∈ 1:N
        Γ1[n] = Multinomial(1, Ω[n,:]) |>
            x -> rand(x, 1) |>
            x -> argmax(x)[1]
    end;
    # Return cluster assignments 
    return Γ1
end;

# Sample mixing proportions from dirichlet
function sample_mixing_proportions(Γ, α0, K)
    #=
    Sample mixing proportions from a dirichlet distribution 
    :param Γ:
    :param ζ0: hypothesized number of subjects in each category.
    :param K: number of clusters.
    :return:
    :seealso:
    :details:
    =#
    N = zeros((K))
    # Sample size by cluster
    for k ∈ 1:K
        N[k] = length(Γ.==k)
    end;
    # Sample from dirichlet 
    ζ1 = Dirichlet(N .+ α0) |> 
        x -> rand(x, 1)
    # Return 
    return ζ1
end;

K = 3
N = [100 90 35];
μ = [1.8 11.; 9.0 10.2 ; -.3 4.];
Σ = cat([5. .6; .6 3.2], [4.2 3; 3 3.6], [3 2.2 ; 2.2 3], dims=K);

# Simulate dataset 
X, Z = simulate(K, N, μ, Σ);



M = size(X)[2]

Γ = Z;
κ0 = zeros((size(X)[2], K))
EY = Matrix(I, M, M)
Τ0 = zeros((M, M, K))
Τ0[:,:,1] = EY ;Τ0[:,:,2] = EY;Τ0[:,:,3] = EY

ν0 = ones((K)) .+ 1
Ψ0 = Τ0 .+ 10
α0 = [1, 1, 1]
ζ = [0.3, 0.3, 0.4]
Σ = Τ0

# Initialize parameters 
function initialize_parameters(N, M, K)
    #= 
    Initialize parameters 
    =#
    # Initialize covariance matrix as identity
    Σ = zeros((M, M, K))
    # Initialize mixing proportions 
    ζ = zeros((K))
    for k ∈ 1:K
        ζ[k] = k < K ? Uniform(.1, .9 - sum(ζ)) |> rand : 1 - sum(ζ)
        Σ[:,:,k] = Matrix(I, M, M)
    end;
    # Shuffle zeta 
    shuffle!(ζ)
    # Initialize mean 
    μ = rand(Float64, (M, K))
    # Initialize cluster assignments 
    d = Multinomial(1, ζ) |>
        x -> rand(x, N) 
    Γ = zeros(Int64,(N))
    for i ∈ 1:N
        Γ[i] = argmax(d[:,i])
    end;
    # Return initial values 
    return μ, Σ, ζ, Γ
end;

# Gibbs sampling routine 
function gibbs_sampler(X, K, α0, κ0, Τ0, ν0, Ψ0; iterations = 2000, burnin = 1000)
    #= 
    Gibbs sampling routine for GMM 
    =#
    N, M = size(X);
    # Initialize parameters 
    μ, Σ, ζ, Γ = initialize_parameters(N, M, K);
    # Bookkeeping
    μ_history = zeros((iterations, M, K));
    Σ_history = zeros((iterations, M, M, K));
    ζ_history = zeros((iterations, K));
    Γ_history = zeros(Int64, (iterations, N));
    # Store initial values 
    μ_history[1, :, :] = μ;
    Σ_history[1, :, :, :] = Σ;
    ζ_history[1, :] = ζ;
    Γ_history[1, :] = Γ;
    # For each iteration, run the sampler
    for i ∈ 2:iterations
        @debug "μ at iteration $i is $μ ..."
        # Sample new mixing proportions 
        ζ_next = sample_mixing_proportions(Γ, α0, K)
        # Sample means 
        μ_next = sample_posterior_mean(X, κ0, Τ0, Γ, Σ)
        # Sample covariances
        Σ_next = sample_posterior_covariance(X, ν0, Ψ0, Γ, μ)
        # Sample cluster assignments
        Γ_next = sample_cluster_assignments(X, ζ, μ, Σ)
        # Store 
        μ_history[i, :, :] = μ_next
        Σ_history[i, :, :, :] = Σ_next
        ζ_history[i, :] = ζ_next
        Γ_history[i, :] = Γ_next
        # Assign new values 
        ζ[:] = ζ_next[:]
        μ[:,:] = μ_next[:,:]
        Σ[:,:,:] = Σ_next[:,:,:]
        Γ[:,:] = Γ_next[:,:]
    end;
    # Return
    return μ_history, Σ_history, ζ_history, Γ_history
end;

history = gibbs_sampler(X, K, α0, κ0, Τ0, ν0, Ψ0; iterations=2000);

plot(history[1][:,1,1])

### End module
end;
