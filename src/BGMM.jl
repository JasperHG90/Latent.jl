#=
Bayesian Gaussian Mixture Model
# TODO: hierarchical priors
# TODO: bayesian updating
=#

module BGMM

using Random, Logging, LinearAlgebra, ProgressMeter, Distributions, Plots
logger = global_logger(SimpleLogger(stdout, Logging.Info)) # Change to Logging.Debug for detailed info

# Sample means from multivariate normal distribution
function sample_posterior_mean(X::Array{Float64}, κ0::Array{Float64}, Τ0::Array{Float64}, Γ::Array{Int64}, Σ::Array{Float64})::Array{Float64}
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
        κ1 = Τ1 * (Τ0_inv * κ0[:,k] .+ Nk .* Σ_inv * ϵ)
        @debug "Posterior means κ1: $κ1 ; Posterior covariance matrix Τ1: $Τ1 ..."
        # Sample means from multivariate normal 
        μ_ret[:, k] = MvNormal(κ1, Τ1) |>
            x -> rand(x, 1)
    end;
    # Return means 
    return μ_ret
end;

# Sample posterior covariance matrices from an inverse-wishart distribution 
function sample_posterior_covariance(X::Array{Float64}, ν0::Array{Int64}, Ψ0::Array{Float64}, Γ::Array{Int64}, μ::Array{Float64})::Array{Float64}
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
        X_k = X[Γ .== k, :]
        for m ∈ 1:M
            X_k[:,m] = X_k[:,m] .- μ[m,k]
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

# Assign to clusters
function cluster_assignments(X::Array{Float64}, ζ::Array{Float64}, μ::Array{Float64}, Σ::Array{Float64})::Array{Int64}
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
    #Γ1 = zeros(Int64,(N))
    Γ2 = zeros(Int64,(N))
    for n ∈ 1:N
        Γ2[n] = argmax(Ω[n,:])
        #Γ1[n] = wsample(1:K, Ω[n,:], 1)[1]
    end;
    #pdiff = sum(Γ1 .!= Γ2) / N
    #@info "Cluster assignments: agreement is $(1-pdiff) ..."
    # Return cluster assignments
    # SOmething really weird happens here 
    # When using ML, the estimation is super. But when using 
    # sampling, it is consistently off by 10%.
    Γ2
end;

# Sample mixing proportions from dirichlet
function sample_mixing_proportions(Γ::Array{Int64}, α0::Array{Int64}, K::Int64)::Array{Float64}
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
        N[k] = length(Γ[Γ.==k])
    end;
    # Sample from dirichlet 
    ζ1 = Dirichlet(N .+ α0) |> 
        x -> rand(x, 1)
    # Return 
    return ζ1
end;

# Initialize parameters 
function initialize_parameters(N::Int64, M::Int64, K::Int64; ϵ = 0.1)::Tuple{Array{Float64}, Array{Float64}, Array{Float64}, Array{Int64}}
    #= 
    Initialize parameters 
    =#
    # Initialize covariance matrix as identity
    Σ = zeros((M, M, K))
    # Initialize mixing proportions 
    ζ = zeros((K))
    for k ∈ 1:K
        ζ[k] = k < K ? Uniform(.1, .9 - sum(ζ)) |> rand : 1 - sum(ζ)
        A = rand(Float64, (M,M))
        Σ[:,:,k] = ϵ * I + A' * A
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
function gibbs_sampler(X::Array{Float64}, K::Int64, α0::Array{Int64}, κ0::Array{Float64}, Τ0::Array{Float64}, ν0::Array{Int64}, Ψ0::Array{Float64}; iterations = 2000, chains = 2, burnin = 1000)::Tuple{Array{Float64}, Array{Float64}, Array{Float64}}
    #= 
    Gibbs sampling routine for GMM 
    =#
    N, M = size(X);
    # Bookkeeping
    μ_history = zeros((chains, iterations, M, K));
    Σ_history = zeros((chains, iterations, M, M, K));
    ζ_history = zeros((chains, iterations, K));
    Γ_history = zeros(Int64, (chains, iterations, N));
    # For each iteration, run the sampler
    for c in 1:chains
        println("Sampling chain $c/$chains ...")
        # Initial values
        μ, Σ, ζ, Γ = initialize_parameters(N, M, K);
        # Store initial values 
        μ_history[c, 1, :, :] = μ;
        Σ_history[c, 1, :, :, :] = Σ;
        ζ_history[c, 1, :] = ζ;
        # Sample for each iteration
        @showprogress "Sampling ..." for i ∈ 2:iterations
            @debug "μ at iteration $i is $μ ..."
            # Sample new mixing proportions 
            ζ = sample_mixing_proportions(Γ, α0, K)
            # Sample means 
            μ = sample_posterior_mean(X, κ0, Τ0, Γ, Σ)
            # Sample covariances
            Σ = sample_posterior_covariance(X, ν0, Ψ0, Γ, μ)
            # Obtain cluster assignments
            Γ = cluster_assignments(X, ζ, μ, Σ)
            # Store 
            μ_history[c, i, :, :] = μ
            Σ_history[c, i, :, :, :] = Σ 
            ζ_history[c, i, :] = ζ
        end;
    end;
    # Return
    return μ_history, Σ_history, ζ_history
end;

### End module
end;
