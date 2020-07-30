module BHMM

using Random, Plots, Logging, LinearAlgebra, ProgressMeter, Distributions
using Logging;

include("src/HMM.jl")

logger = global_logger(SimpleLogger(stdout, Logging.Info)); # Change to Logging.Debug for detailed info
#close(io)

# Generate dataset
Random.seed!(425234);
M = 3
T = 800
Γ = [0.7 0.12 0.18 ; 0.17 0.6 0.23 ; 0.32 0.38 0.3]
μ = [-6.0 ; 0; 6]
σ = [0.1 ; 2.0; 1.4]
X, Z = Main.HMM.simulate(M, T, Γ, μ, σ);

# Forward probabilities 
function forward_probabilities(Γ::Array{Float64}, Ψ::Array{Float64})::Array{Float64}
    #=
    Compute forward probabilities
    =#
    M, _ = size(Γ)
    T = size(X)[1]
    # Allocate 
    A = zeros((T, M))
    # Compute the initial distribution 
    δ = (Matrix(I, M, M) .- Γ .+ 1) \ ones(M)
    α = δ .* Ψ[1,:]
    A[1,:] = α /= sum(α)
    for t ∈ 1:T
        ω = (α' * Γ)' .* Ψ[t,:]
        A[t,:] = ω / sum(ω)
    end;
    # Return 
    return A
end;

function sample_states(Γ::Array{Float64}, Ψ::Array{Float64})::Tuple{Array{Float64}, Array{Int64}}
    #=
    Sample states given the data and TPM 
    =#
    # Compute forward probabilities
    A = forward_probabilities(Γ, Ψ)
    M, _ = size(Γ)
    # Sampled states 
    λ = zeros(Int64, (T))
    # Unnormalized TPM 
    UΓ = zeros(Int64, (M, M))
    # At time T 
    # Sample from the rows 
    λ[T] = Multinomial(1, A[T,:]) |>
        x -> rand(x, 1) |>
        x -> argmax(x)[1]
    for t ∈ reverse(1:T-1)
        # If t < T, then multiply times the transition probability
        αt = A[t,:] .* Γ[:,λ[t+1]]
        αt /= sum(αt)
        # Sample from the rows 
        λ[t] = Multinomial(1, αt) |>
            x -> rand(x, 1) |>
            x -> argmax(x)[1]
        # Add to unnormalized TPM 
        UΓ[λ[t], λ[t+1]] += 1
    end;
    # Return state & tpm
    return λ, UΓ

end;

# Compute likelihood of X 
Ψ = Main.HMM.normalized_pdf(X, μ, σ);
λ, UG = sample_states(Γ, Ψ);

mapslices(x -> argmax(x), α, dims=[2])

### End module
end;