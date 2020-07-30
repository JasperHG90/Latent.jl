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

# Sample state sequence and unnormalized TPM 
function sample_states(Γ::Array{Float64}, Ψ::Array{Float64})::Tuple{Array{Int64}, Array{Int64}}
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
    #λ[T] = Multinomial(1, A[T,:]) |>
        #x -> rand(x, 1) |>
        #x -> argmax(x)[1]
    λ[T] = argmax(A[T,:])
    for t ∈ reverse(1:T-1)
        # If t < T, then multiply times the transition probability
        αt = A[t,:] .* Γ[:,λ[t+1]]
        αt /= sum(αt)
        # Sample from the rows 
        #λ[t] = Multinomial(1, αt) |>
            #x -> rand(x, 1) |>
            #x -> argmax(x)[1]
        λ[t] = argmax(αt)
        # Add to unnormalized TPM 
        UΓ[λ[t], λ[t+1]] += 1
    end;
    # Return state & tpm
    return λ, UΓ

end;

# Sample TPM from dirichlet
function sample_TPM(UΓ::Array{Int64}, α0::Array{Int64})::Array{Float64}
    #=
    From the counts in the TPM, sample new transition probabilities
    =#
    # Shapes
    M, _ = size(UΓ)
    # Allocate
    NΓ = zeros((M,M))
    # Draw from dirichlet
    for m ∈ 1:M
        # Sample from dirichlet 
        NΓ[m,:] = Dirichlet(UΓ[m,:] .+ α0) |> 
            x -> rand(x, 1)
    end;
    # Return 
    return NΓ
end;

# Sample means from normal
function sample_means(X::Array{Float64}, λ::Array{Int64}, M::Int64, μ0::Array{Float64}, τ0::Array{Float64}, σ::Array{Float64})::Array{Float64}
    #= 
    Sample means from univariate normal 
    =#
    μ_out = zeros((M))
    # For each state, compute means 
    for m ∈ 1:M
        # Compute sample size 
        Nm = sum(λ .== m)
        # Sample mean 
        if Nm == 0.0
            xbk = 0.0
        else
            xbk = mean(X[λ .== m, :], dims=1)[1]
        end;
        # Posterior precision 
        τ1 = (Nm / σ[m]) + τ0[m]
        # Posterior mean 
        μ1 = (μ0[m] * τ0[m] + (xbk * Nm * (1/σ[m]))) / τ1
        # Sample 
        μ_out[m] = Normal(μ1, sqrt(1/τ1)) |> rand
    end;
    # Return 
    return μ_out
end;

# Compute likelihood of X 
Ψ = Main.HMM.normalized_pdf(X, μ, σ);
_, UΓ = sample_states(Γ, Ψ);
# Sample new TPM from dirichlet distribution
NΓ = sample_TPM(UΓ, α0)

# Gibbs sampling routine while keeping mu and sigma fixed 
# History 
# Iterations 
niter = 15000;
Γ_history = zeros((M, M, niter));
μ_history = zeros((M, niter));
# Set priors 
μ0 = zeros(M)
τ0 = ones(M)
α0 = ones(Int64, M)
let
    # Initialize 
    Γ, μ, _ = Main.HMM.initialize_parameters(M);
    #μ = [-1.0 ; 0; 1]
    # Save 
    Γ_history[:,:,1] = Γ
    μ_history[:,1] = μ
    @showprogress "Sampling ..." for n ∈ 2:niter 
        # Compute normalized likelihood 
        Ψ = Main.HMM.normalized_pdf(X, μ, σ)
        # Sample state sequence and unnormalized TPm
        λ, UΓ = sample_states(Γ, Ψ)
        # Sample new TPM from dirichlet distribution
        NΓ = sample_TPM(UΓ, α0)
        # Sample new mean 
        Nμ = sample_means(X, λ, M, μ0, τ0, σ)
        # Save in history 
        Γ_history[:,:,n] = NΓ
        μ_history[:,n] = Nμ
        # Set to new Γ
        Γ = NΓ
        μ = Nμ
    end;
end;

mapslices(x -> argmax(x), α, dims=[2])

plot(μ_history')

### End module
end;