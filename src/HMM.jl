module HMM

#= 
Implementation of a Hidden Markov Model (HMM) with a Gaussian emission distribution
TODO: forward // backward probabilities -- use log s.t. multiplication does not become excessively small
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
    X = zeros((T, 1))
    Z = zeros(Int64, (T, 1))
    # Draw from latent states
    Z[1,1] = Multinomial(1, δ) |>
        x -> rand(x, 1) |>
        x -> argmax(x)[1]
    # Draw first X-value 
    X[1,1] = Normal(μ[Z[1,1]], σ[Z[1,1]]) |>
            x -> rand(x, 1)[1]
    for t ∈ 2:T 
        # Multiply the initial distribution by the t'th power of Γ and draw latent state
        Z[t,1] = δ' * Base.power_by_squaring(Γ, t) |>
            x -> Multinomial(1, x') |>
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
M = 3
T = 500
Γ = [0.2 0.3 0.5 ; 0.3 0.3 0.4; 0.8 0.1 0.1]
δ = [(1/3) ; (1/3); (1/3)]
μ = [1.0 ; 6.0 ; -5.0]
σ = [1.3 ; 0.8; 0.9]

X, Z = simulate_HMM(M, T, Γ, δ, μ, σ);
# X is bimodal
histogram(X, bins=20)

# Utility function to retrieve the normalized probability density of X given parameters on Z
function normalized_pdf(X::Array{Float64}, μ::Array{Float64}, σ::Array{Float64})::Array{Float64}
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
function forward_algorithm(X::Array{Float64}, Γ::Array{Float64}, Ψ::Array{Float64}, δ::Array{Float64})::Array{Float64}
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
    T = size(X)[1]
    M = size(Γ)[1]
    Α = zeros((T, M))
    # Populate first alpha (initial distribution)
    Α[1, :] = δ .* Ψ[1, :]
    # For each step in 2 : T, populate alpha 
    for t ∈ 2:T
        for m ∈ 1:M
            Α[t, m] = Α[t-1, :]' * Γ[m, :] .* Ψ[t, m]
        end;
    end;
    # return
    return Α
end;

# Backward algorithm
function backward_algorithm(X::Array{Float64}, Γ::Array{Float64}, Ψ::Array{Float64}, δ::Array{Float64})::Array{Float64}
    #=
    Compute the backward step of the forward-backward algorithm
    :param X: observed data. Sequence of datapoints of length T (1 x T)
    :param Γ: transition probabily matrix of size m x m, where m is the number of hidden states 
    :param Ψ: emission distribution probabilities.
    :return: matrix Β ∈ R^{T x m} with probabilities computed in the forward step
    =#
    # Initialize Β
    T = size(X)[1]
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
function viterbi_algorithm(Ψ::Array{Float64}, Γ::Array{Float64}, δ::Array{Float64})::Tuple{Array{Int64}, Float64}
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
    LL = Ω[T, states[T]]
    # Backward pass 
    for t ∈ reverse(1:(T-1))
        states[t] = sequences[t, states[t+1]]
        LL += Ω[t, states[t]]
    end;
    # Return the state sequence, log-likelihood 
    return states, LL
end;

Ψ = normalized_pdf(X, μ, σ)

forward_algorithm(X, Γ, Ψ, δ)
backward_algorithm(X, Γ, Ψ, δ)

states = viterbi_algorithm(Ψ, Γ, δ);
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