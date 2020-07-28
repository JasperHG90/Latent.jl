#=
Bayesian Gaussian Mixture Model
=#

module GMM

using Random, Logging, LinearAlgebra, ProgressMeter, Distributions

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

K = 3
N = [100 90 35];
μ = [1.8 11.; 9.0 10.2 ; -.3 4.];
Σ = cat([5. .6; .6 3.2], [4.2 3; 3 3.6], [3 2.2 ; 2.2 3], dims=K);

# Simulate dataset 
X, Z = simulate(K, N, μ, Σ);

# Sample means from multivariate normal distribution
function sample_posterior_mean(X, π, μ0, Γ, Σ)
    #=
    Compute the posterior means for a multivariate normal distribution 
    =#
    M = size(X)[2]
    K = size(Σ)[3]
    out = zeros((M, K))
    # For each cluster, sample from multivariate normal
    for k ∈ 1:K
        # Compute sample size in cluster k 
        Nk = sum(Z[Z.==k])
        # If sample size is 0, then just initialize with 0
        if(Nk == 0)
            ϵ = zeros((K))
        else
            
        end;
    end;
end;

### End module
end;

X = randn(100);
Y = 3 .* randn(100) .+ 2;
S = X .>= 0; 
X_greater_0 = X[S]