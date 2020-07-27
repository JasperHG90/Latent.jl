module HMM

#= 
Implementation of a Hidden Markov Model (HMM) with a Gaussian emission distribution

Code in part adapted from:

    Zucchini, W., MacDonald, I. L., & Langrock, R. (2017). Hidden Markov models for 
    time series: an introduction using R. CRC press.

TODO: forward // backward probabilities -- use log s.t. multiplication does not become excessively small
TODO: simulating data --> ensure that are using the right definitions (i.e. gamma is homogeneous)
TODO: add multinomial distribution 
TODO: add forecasting function (p.246)
=#

using Random, Plots, Logging, LinearAlgebra, ProgressMeter, Distributions
using Logging;
logger = global_logger(SimpleLogger(stdout, Logging.Debug)) # Change to Logging.Debug for detailed info

"""
    simulate_HMM(M::Int64, T::Int64, Γ::Array{Float64}, δ::Array{Float64}, μ::Array{Float64}, σ::Array{Float64})::Tuple{Array{Float64}, Array{Float64}}

Simulate a dataset of a mixture of M gaussians for T timesteps. This dataset will follow a Markov process.

# Arguments 
- `M::Int64`: Number of latent states.
- `T::Array{Int64}`: Length of the observed sequence of data.
- `Γ::Array{Float64}`: Transition probability matrix. Array of dimensions (M x M).
- `δ::Array{Float64}`: Initial transition probabilities. Array of dimensions (M x 1).
- `μ::Array{Float64}`: Array of dimensions (M x 1) containing the means for each latent state.
- `σ::Array{Float64}`: Array of dimensions (M x 1) containing the variances for each latent state.

# Returns 
An array of dimensions (∑N x M) containing the data points.

# Examples
```julia-repl
julia>
```
"""
function simulate_HMM(M::Int64, T::Int64, Γ::Array{Float64}, δ::Array{Float64}, μ::Array{Float64}, σ::Array{Float64})::Tuple{Array{Float64}, Array{Int64}}
    # Assert dimensions 
    @assert M == size(σ)[1] "Number of variances not equal to the number of latent states ..."
    @assert M == size(μ)[1] "Number of means not equal to the number of latent states ..."
    @assert M == size(Γ)[1] == size(Γ)[2] "Transition probability matrix should be of dimensions M x M ..."
    @assert M == size(δ)[1] "Initial transition probability array should be of dimensions M x 1"
    # TODO: TPM may not contain zero entries
    # Populate X and Z 
    X = zeros((T))
    Z = zeros(Int64, (T))
    # Draw from latent states
    Z[1] = Multinomial(1, δ) |>
        x -> rand(x, 1) |>
        x -> argmax(x)[1]
    # Draw first X-value 
    X[1] = Normal(μ[Z[1]], σ[Z[1]]) |>
            x -> rand(x, 1)[1]
    for t ∈ 2:T 
        # Draw from distribution
        Z[t] = Γ[Z[t-1],:] |>
            x -> Multinomial(1, x) |>
            x -> rand(x, 1) |>
            x -> argmax(x)[1]
        # Draw X-value 
        X[t,1] = Normal(μ[Z[t,1]], σ[Z[t,1]]) |>
            x -> rand(x, 1)[1]
    end;
    # Return X and Z 
    return X, Z
end;

Random.seed!(425234);
M = 2
T = 500
Γ = [0.8 0.2 ; 0.35 0.65]
δ = [(1/2) ; (1/2)]
μ = [-1.0 ; 4.0]
σ = [1.3 ; 0.8]

#inv(Γ'Γ) * Γ' * ones((M))

X, Z = simulate_HMM(M, T, Γ, δ, μ, σ);
# X is bimodal
histogram(X, bins=15)

Ψ = normalized_pdf(X, μ, σ)
a = forward_algorithm(Γ ,Ψ)
b = backward_algorithm(Γ, Ψ)

# Utility function to retrieve the normalized probability density of X given parameters on Z
function normalized_pdf(X::Array{Float64}, μ::Array{Float64}, σ::Array{Float64})::Array{Float64}
    #=
    Compute the probabilities of X for each component distribution given its parameters 
    :param X: univariate input data. Array of size T x 1
    :param μ: component distribution means. Array of size M x 1
    :param σ: component distribution standard deviations. Array of size M x 1
    :return: Ψ, which is an array of size T x M, where Ψ[:,m] contains the likelihood of X 
             given the parameters of component distribution m 
    =#
    # Get dimensions 
    T = size(X)[1]
    M = size(μ)[1]
    # Results matrix 
    Ψ = zeros((T, M));
    # Compute the likelihood of X given parameters on Z
    for m ∈ 1:M
        Ψ[:,m] = Normal(μ[m], σ[m]) |>
            dist -> pdf.(dist, X)
    end;
    # Normalize across rows s.t. each row sums to unity 
    Ψ ./= sum(Ψ, dims=2)
    # Return psi 
    return Ψ
end;

# Forward algorithm 
# This function uses logged probabilities to reduce the chance of underflow issues 
function forward_algorithm(Γ::Array{Float64}, Ψ::Array{Float64})::Array{Float64}
    #=
    Compute forward step of the forward-backward algorithm
    :param Ψ: Likelihood of input data X given the M component distributions. Array of size T x M. See also 
               normalized_pdf()
    :param Γ: transition probabily matrix of size m x m, where m is the number of hidden states 
    :param Ψ: emission distribution probabilities. In the case of gaussian emission distributions, this is:

                p(X | Zk) = N(X | μk, Σk) ∀ k

              We assume that we know these and that this matrix Ψ ∈ R^{m x T}

    :param δ: initial distribution matrix of size 1 x m
    :return: matrix Α ∈ R^{T x m} with logged probabilities computed in the forward step
    =#
    # Shapes
    T = size(Ψ)[1] 
    M = size(Γ)[1]
    # Compute the initial distribution 
    δ = (Matrix(I, M, M) .- Γ .+ 1) \ ones(M)
    # Set up matrix α
    α = zeros((T, M))
    # Forward step using the initial distribution 
    κ = δ .* Ψ[1, :]
    ω = sum(κ)[1]
    η = log(ω)
    # Normalize κ by ω (turning it into a valid distribution)
    κ /= ω
    # Add to alpha 
    α[1, :] = log.(κ) .+ η
    # Repeat for 2:T
    for t ∈ 2:T 
        # Forward step 
        κ = (κ' * Γ)' .* Ψ[t,:]
        # Normalize 
        ω = sum(κ)[1]
        κ /= ω
        # Add to total 
        η += log(ω)
        # Set alpha 
        α[t,:] = log.(κ) .+ η
    end;
    # Return alpha 
    return α
end;

# Backward algorithm 
function backward_algorithm(Γ::Array{Float64}, Ψ::Array{Float64})::Array{Float64}
    #=
    Compute the backward step of the forward-backward algorithm
    :param Ψ: Likelihood of input data X given the M component distributions. Array of size T x M. See also 
               normalized_pdf()
    :param Γ: transition probabily matrix of size m x m, where m is the number of hidden states 
    :param Ψ: emission distribution probabilities.
    :return: matrix Β ∈ R^{T x m} with logged probabilities computed in the backward step
    =#
    # Shapes
    T = size(Ψ)[1] 
    M = size(Γ)[1]
    # Set up matrix α
    β = zeros((T, M))
    # setting ln(beta(T)) = ln(1) = [0] * M
    β[T,:] = zeros((1, M))
    # Compute backward probabilities at T 
    κ = repeat([1/M], outer=[M])
    η = log(M)
    # Loop from T-1 to 1
    for t ∈ reverse(1:(T-1))
        κ = Γ * (Ψ[t+1,:] .* κ)
        # Add to beta 
        β[t,:] = @. η + log(κ)
        # Sum across kappa and normalize 
        ω = sum(κ)[1]
        κ /= ω
        η += log(ω)
    end;
    # Return beta 
    return β
end;

# Baum-welch algorithm
# Modified to reduce the chance of underflow errors.
function baum_welch(Γ::Array{Float64}, X::Array{Float64}, μ::Array{Float64}, σ::Array{Float64}; iterations = 150, tol=1e-6)::Tuple{Array{Float64}, Array{Float64}, Array{Float64}, Float64, Float64, Float64}
    #=
    Use the Baum-Welch algorithm to estimate the TPM and the emission distribution parameters
    :param Γ: initial transition probabily matrix of size m x m, where m is the number of hidden states 
    :param X: observed data. Sequence of datapoints of length T (T x 1).    
    :param μ: intial mean vector of length M x 1.
    :param σ: initial vector containing standard deviations. This array is of length M x 1.
    :return: Tuple containing updated Γ, μ, and σ as well as the log-likelihood, the AIC and the BIC
    =#
    # Get dimensions 
    M = size(Γ)[1]
    T = size(X)[1]
    # For each iteration, estimate parameters 
    for n ∈ 1:iterations 
        # Compute the likelihood of the data given the parameters 
        #  Ψ ∈ R^{T, M}
        Ψ = normalized_pdf(X, μ, σ)
        # Compute forward & backward probabilities 
        α = forward_algorithm(Γ, Ψ)
        β = backward_algorithm(Γ, Ψ)
        # Compute the log-likelihood of the data 
        s1 = α[T, :]
        s2 = s1[argmax(s1)]
        LL = s2 + log(sum(exp.(s1 .- s2)))
        
    end;
end;

# Forward algorithm
function forward_algorithm_old(Γ::Array{Float64}, Ψ::Array{Float64}, δ::Array{Float64})::Array{Float64}
    #=
    Compute forward step of the forward-backward algorithm
    :param X: observed data. Sequence of datapoints of length T (1 x T)
    :param Γ: transition probabily matrix of size m x m, where m is the number of hidden states 
    :param Ψ: emission distribution probabilities. In the case of gaussian emission distributions, this is:

                p(X | Zk) = N(X | μk, Σk) ∀ k

              We assume that we know these and that this matrix Ψ ∈ R^{m x T}

    -- TODO: solve this later :param Ψ: emission distribution parameters. Matrix of size (m x n(θ)), where n(θ) equals the number of parameters.
    :param δ: initial distribution matrix of size 1 x m
    :return: matrix Α ∈ R^{T x m} with probabilities computed in the forward step
    =#
    # Initialize alpha
    T = size(Ψ)[1]
    M = size(Γ)[1]
    Α = zeros((T, M))
    # Populate first alpha (initial distribution)
    Α[1, :] = δ .* Ψ[1, :]
    # For each step in 2 : T, populate alpha 
    for t ∈ 2:T
        for m ∈ 1:M
            Α[t, m] = (Α[t-1, :]' * Γ[:, m]) .* Ψ[t, m]
        end;
    end;
    # return
    return Α
end;

# Backward algorithm
function backward_algorithm_old(Γ::Array{Float64}, Ψ::Array{Float64}, δ::Array{Float64})::Array{Float64}
    #=
    Compute the backward step of the forward-backward algorithm
    :param X: observed data. Sequence of datapoints of length T (1 x T)
    :param Γ: transition probabily matrix of size m x m, where m is the number of hidden states 
    :param Ψ: emission distribution probabilities.
    :return: matrix Β ∈ R^{T x m} with probabilities computed in the forward step
    =#
    # Initialize Β
    T = size(Ψ)[1]
    M = size(Γ)[1]
    Β = zeros((T, M))
    # Set B_T to 1
    Β[T,:] = ones((1, M))
    # Loop from T-1 to 1
    for t ∈ reverse(1:(T-1))
        for m ∈ 1:M
            Β[t, m] = (Β[t+1, :] .* Ψ[t+1, m])' * Γ[m,:]
        end;
    end;
    # Return 
    return Β
end;

# Utility function that computes log-likelihood given x, mu, sigma and transition probability
function log_likelihood(ψ::Float64, γ::Float64)::Float64
    #= 
    Compute the log-likelihood of X given the parameters 
    =#
    return log(ψ) + log(γ)
end;

# Viterbi algorithm
# Used for decoding the most likely set of state sequences across T
function viterbi_algorithm(Ψ::Array{Float64}, Γ::Array{Float64}, δ::Array{Float64})::Array{Int64}
    #=
    Compute the maximizing latent state sequence
    =#
    # Dimensions 
    T = size(Ψ)[1]
    M = size(Γ)[1]
    # State sequences
    sequences = zeros(Int64, (T-1, M))
    # Viterbi probabilities 
    Ω = zeros((T, M))
    # First time step 
    Ω[1,:] = [log_likelihood(Ψ[1,m], δ[m]) for m ∈ 1:M]
    @debug "Forward pass (computing set of likely sequences) ...)"
    # Loop through time steps 
    for t ∈ 2:T
        @debug "Computing sequences at time step $t ..."
        # For each state, conditioning on the last state, do ...
        for m ∈ 1:M
            # Compute for each state the probability of ending up at state m ...
            prob = zeros((M))
            for j ∈ 1:M
                prob[j] = Ω[t-1, j] + log_likelihood(Ψ[t, m], Γ[j, m])
            end;
            # Get argmax
            sequences[t-1, m] = argmax(prob)
            Ω[t, m] = prob[sequences[t-1,m]]
        end;
    end;
    @debug "Finished forward pass ..."
    states = zeros(Int64, (T))
    states[T] = argmax(Ω[T, :])
    # Backward pass 
    for t ∈ reverse(1:(T-1))
        states[t] = sequences[t, states[t+1]]
    end;
    # Return the state sequence, log-likelihood 
    return states
end;

# Baum-Welch algorithm (EM estimation)
function baum_welch(Γ::Array{Float64}, X::Array{Float64}, δ::Array{Float64}; iterations = 150)
    #=
    Perform EM estimation for the HMM parameters
    =#
    # Shapes 
    M, _ = size(Γ)
    T = size(X)[1]
    # Initialize parameters 
    # For each iteration 
    for i ∈ 1:iterations
        # Compute PDF on X 
        Ψ = normalized_pdf(X, μ, σ)
        # Compute forward & backward probabilities
        α = forward_algorithm(Γ, Ψ, δ)
        β = backward_algorithm(Γ, Ψ, δ)
        # expectation step (latent variable optimization)
        Λ = zeros((M, M, T-1))
        for t ∈ 1:(T-1)
            d = ((α[t, :]' * Γ) .* Ψ[t+1, :]) * β[t+1, :]'
            for m ∈ 1:M
                n = α[t, m] .* Γ[m, :] * Ψ[t+1,:]' .* β[t+1,:]
                Λ[m, :, t] = n / d
            end;
        
        end;
    end;
end;

M = 2
T = 40

Γ = [0.3 0.7 ; 0.6 0.4]
δ = [(1/2) ; (1/2)]
μ = [-1.0 ; 4.0]
σ = [1.3 ; 0.8]

X, Z = simulate_HMM(M, T, Γ, δ, μ, σ);

Γ = rand(Float64, (M,M))
Γ ./ sum(Γ, dims=2)
δ = rand(Float64, (M))
δ ./ sum(δ)  
μ = rand(Float64, (M))
σ = ones(M)

# Initialize parameters 
#  .... TODO
# For each iteration 
iterations = 20
for i ∈ 1:iterations 
    @show μ
    @show σ
    @show Γ
    @debug "Processing iteration $i ..."
    # Compute PDF on X 
    Ψ = normalized_pdf(X, μ, σ)
    # Compute forward & backward probabilities
    α = forward_algorithm(Γ, Ψ, δ)
    β = backward_algorithm(Γ, Ψ, δ)
    # expectation step (latent variable optimization)
    Λ = zeros((M, M, T-1))
    for t ∈ 1:(T-1)
        # This is correct
        d = (α[t,:]' * Γ .* Ψ[t+1, :]') * β[t+1, :]
        for m ∈ 1:M
            n = α[t, m] .* Γ[m, :] .* Ψ[t+1, :] .* β[t+1,:]
            Λ[m, :, t] = n ./ d
        end;
    end;
    # Take sum across columns in the tpm 
    ζ = sum(Λ, dims=2) |>
        x -> dropdims(x; dims=2)
    # Update Gamma
    Γ[:,:] = sum(Λ, dims=3) ./ sum(ζ, dims=2) |>
        x -> dropdims(x; dims=3)[:,:]
    # Add T'th dimensions
    ζ = sum(Λ[:,:,T-2], dims=1) |>
        x -> hcat(ζ, x')
    # Compute mixing proportions
    # (This is the same as in regular EM)
    mp = sum(ζ, dims=2) ./ sum(ζ)
    ζ_cs = sum(ζ, dims=2)
    # Update mean / variance estimates 
    for m ∈ 1:M
        # Compute mean for cluster m 
        μ_m = sum(ζ[m,:] .* X, dims=1)[1] / ζ_cs[m]
        # Center X by mu_m
        X_centered = X .- μ_m 
        # Compute variance 
        σ_m = (ζ[m,:] .* X_centered)' * X_centered ./ ζ_cs[m]
        # Update parameters 
        μ[m] = μ_m 
        σ[m] = σ_m[1]
    end;
end;
# Viterbi algorithm 
v = viterbi_algorithm(normalized_pdf(X, μ, σ),
                      Γ, δ)




# This is supposed to be a 3 x 1 array (row m of the TPM at time t)
for m ∈ 1:M
    n = α[t, m] .* Γ[m, :] .* Ψ[t+1, :] .* β[t+1,:]
    Λ[m, :, t] = n ./ d
end;

d = ((α[t, :]' * Γ) .* Ψ[t+1, :]') * β[t+1, :]
m = 1
n = α[t, m] .* Γ[m, :] .* Ψ[t+1,:]' .* β[t+1,:]

α[t,:]' * Γ .* Ψ[t+1, :]'


Ψ = normalized_pdf(X, μ, σ)

forward_algorithm(X, Γ, Ψ, δ)
backward_algorithm(X, Γ, Ψ, δ)

states, LL = viterbi_algorithm(Ψ, Γ, δ);
# Not all states correct
sum(states .== Z)
# Which ones?
findall(x->x==1, states .!= Z)

# Value of wrong state
idx = 123
# Value of observed 
X[idx]
# Value of latent 
Z[idx]
# Value of predicted latent 
states[idx]
# Value of probabilities for each state
Ψ[idx, :]
# Transition probability given previous state 
states[idx-1]
Γ[states[idx-1], :]
Γ[states[idx-1], states[idx]]

### End module
end;