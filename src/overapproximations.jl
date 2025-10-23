using JuMP
using Gurobi
using Polyhedra

include("utils.jl")

function linear_overapproximation(f::Function, domain::HRepresentation; n_samples::Int=0, dispersion::Real=0, Lipschitz::Real=0, H_W::Union{HRepresentation,Nothing}=nothing)
    # returns A and W such that ∀ x ∈ domain: f(x) \in A*x ⊕ W, with W = {w | H_W*w ≤ h_W }. The objective sum(h_W) is minimized.

    dim_x = fulldim(domain)

    if (n_samples == 0 && dispersion == 0) || (n_samples != 0 && dispersion != 0)
        error("One and only one of the keyword arguments 'n_samples' and 'dispersion' must be provided.")
    end
    if H_W !== nothing && Lipschitz > 0
        error("The keyword arguments 'H_W' and 'Lipschitz' cannot be used together.")
    end

    if n_samples > 0
        samples_x = grid_polytope(domain, n_samples=n_samples) # Vector{Vector{Float64}}
    elseif dispersion > 0
        samples_x = grid_polytope(domain, dispersion=dispersion) # Vector{Vector{Float64}}
    end
    n_samples = length(samples_x)
    println("n_samples=$n_samples used to compute the linear over-approximation")

    samples_f = hcat([f(sample_x) for sample_x in samples_x]...) # dim_f x n_samples

    samples_x = hcat(samples_x...) # dim_x x n_samples

    dim_f = size(samples_f, 1)

    if H_W === nothing
        H_W = axis_aligned_H(dim_f)
    end

    model = Model(Gurobi.Optimizer)

    @variable(model, A[1:dim_f, 1:dim_x])
    @variable(model, h_W[1:size(H_W,1)])

    @constraint(model, H_W*( samples_f - A*samples_x ) .<= h_W)

    @objective(model, Min, sum(h_W))

    optimize!(model)

    A = value.(A)
    h_W = value.(h_W)

    if Lipschitz > 0 && dispersion > 0
        # Lipschite constant of the error err(x) = f(x)-Ax is upper bounded by Lip_f + ||A||_{∞→∞}
        Lipschitz_error = Lipschitz + opnorm(A, Inf)
        @info "Enlarging W by Lipschitz_error * dispersion = $(Lipschitz_error * dispersion)"
        # Enforce Lipschitz constraint on W
        h_W = h_W .+ Lipschitz_error * dispersion
    end
    W = hrep(H_W, h_W)

    return A, W
end

function linear_overapproximation(f::Function, domains::Vector{T}; n_samples::Int=0, dispersion::Real=0, Lipschitz::Real=0, H_W::Union{HRepresentation,Nothing}=nothing) where T<:HRepresentation
    # f: function with k arguments
    # domains: vector of k polytopes (H-representation) for each argument
    # returns A_list and W such that ∀ (x_1,...x_k) ∈ product(domains): f(x_1,...x_k) ∈ A_list[1]*x_1 + ... + A_list[k]*x_k ⊕ W

    k = length(domains)
    dims_x = [fulldim(dom) for dom in domains]
    dim_total = sum(dims_x)

    # Cartesian product of domains
    domain_product = *(domains...)  # must return HRepresentation of product

    # Wrap f into a single-argument function that takes the concatenated vector
    function f_wrapped(x_concat::Vector{Float64})
        # split the concatenated vector into pieces
        args = Vector{Vector{Float64}}(undef, k)
        idx = 1
        for i in 1:k
            args[i] = x_concat[idx:idx+dims_x[i]-1]
            idx += dims_x[i]
        end
        return f(args...)  # call original f with multiple args
    end

    A_big, W = linear_overapproximation(f_wrapped, domain_product; n_samples=n_samples, dispersion=dispersion, Lipschitz=Lipschitz, H_W=H_W)

    # De-concatenate A_big into [A1, A2, ...]
    A_list = Vector{Matrix{Float64}}(undef, k)
    col_start = 1
    for i in 1:k
        A_list[i] = A_big[:, col_start:col_start + dims_x[i] - 1]
        col_start += dims_x[i]
    end

    return A_list, W

end