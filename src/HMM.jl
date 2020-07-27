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
    @debug "Computing likelihood of X ..."
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
    @debug "Finished computing likelihood of X ..."
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
    @debug "Computing forward probabilities ..."
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
    @debug "Completed forward step ..."
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
    @debug "Computing backward step ...."
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
    @debug "Completed backward step ..."
    return β
end;

# Baum-welch algorithm
# Modified to reduce the chance of underflow errors.
function baum_welch(Γ::Array{Float64}, X::Array{Float64}, μ::Array{Float64}, σ::Array{Float64}; iterations = 150, tol=1e-20)::Tuple{Array{Float64}, Array{Float64}, Array{Float64}, Float64, Float64, Float64}
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
    @showprogress "Fitting HMM ..." for n ∈ 1:iterations 
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
        @debug "Log-likelihood of the forward-backward probabilities is $LL ..."
        # For each state, compute next values 
        Γ_next = zeros(size(Γ))
        μ_next = zeros(size(μ))
        σ_next = zeros(size(σ))
        for i ∈ 1:M
            for j ∈ 1:M
                inside = @. α[1:(T-1), i] + Ψ[2:T,j] + β[2:T, j] - LL
                Γ_next[i,j] = Γ[i,j] * sum(exp.(inside))
            end;
            # Update mean, sd 
            p_i = @. exp(α[:,i] + β[:, i] - LL)
            μ_next[i] = sum(p_i .* X) / sum(p_i)
            X_centered = X .- μ_next[i]
            σ_next[i] = sqrt((p_i .* X_centered)' * X_centered / sum(p_i))
        end;
        # Normalize new TPM 
        Γ_next ./= sum(Γ_next, dims=1)'
        # Compute measures of fit 
        params = M*M+M-1
        AIC = 2*(LL-params)
        BIC = -2*LL*params*log(T)
        # Compute stopping criterion 
        # (sum of difference in parameter change)
        ϵ = (abs.(μ.-μ_next) |> sum, abs.(σ.-σ_next) |> sum, abs.(Γ.-Γ_next) |> sum) |>
            sum
        @debug "Epsilon has value $ϵ ..."
        # Set new parameters to old ones 
        Γ = Γ_next 
        μ = μ_next
        σ = σ_next
        # If less than tolerance, break 
        if ϵ < tol
            @debug "Stopping condition reached after $n iterations. Exiting now ..."
            return Γ, μ, σ, LL, AIC, BIC
        end;
    end;
    # Return best parameters and fit statistics
    return Γ, μ, σ, LL, AIC, BIC
end;

# Viterbi algorithm
# Used for decoding the most likely set of state sequences across T
function viterbi_algorithm(Γ::Array{Float64}, Ψ::Array{Float64})::Array{Int64}
    #=
    Compute the maximizing latent state sequence
    =#
    # Dimensions 
    T = size(Ψ)[1]
    M = size(Γ)[1]
    # Compute the initial distribution 
    δ = (Matrix(I, M, M) .- Γ .+ 1) \ ones(M)
    # State sequences
    sequences = zeros(Int64, (T-1, M))
    # Viterbi probabilities 
    Ω = zeros((T, M))
    # First time step 
    Ω[1,:] = [log(Ψ[1,m]) + log(δ[m]) for m ∈ 1:M]
    @debug "Forward pass (computing set of likely sequences) ...)"
    # Loop through time steps 
    for t ∈ 2:T
        @debug "Computing sequences at time step $t ..."
        # For each state, conditioning on the last state, do ...
        for m ∈ 1:M
            # Compute for each state the probability of ending up at state m ...
            prob = zeros((M))
            for j ∈ 1:M
                prob[j] = Ω[t-1, j] + log(Ψ[t, m]) + log(Γ[j, m])
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

# Initialize parameters 
function initialize_parameters(M::Int64, T::Int64)::Tuple{Array{Float64},Array{Float64}, Array{Float64}}
    #=
    Initialize the TPM and parameters of the distributions
    =#
    # Gamma 
    Γ = ones((M,M)) ./= M
    μ = Uniform(0, 1) |>
        x -> rand(x, M)
    # Standard deviations
    σ = ones((M))
    # Return 
    return Γ, μ, σ
end;

# Generate dataset
Random.seed!(425234);
M = 2
T = 500
Γ = [0.8 0.2 ; 0.35 0.65]
μ = [-2.0 ; 4.0]
σ = [1.3 ; 0.8]
X, Z = simulate_HMM(M, T, Γ, δ, μ, σ);
# X is bimodal
histogram(X, bins=15)

# Make initial parameters 
Γ0, μ0, σ0 = initialize_parameters(M, T)

# Fit HMM
Γ1, μ1, σ1, LL, AIC, BIC = baum_welch(Γ0, X, μ0, σ0; iterations=100, tol=1e-6)

# Predicted state sequence 
Ψ = normalized_pdf(X, μ, σ)
S = viterbi_algorithm(Γ, Ψ)

# Accuracy
sum(Z .== S) / length(Z)

### End module
end;