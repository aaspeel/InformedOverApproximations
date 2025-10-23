using JuMP
using LinearAlgebra
using Polyhedra

include("LinearSystems.jl")
using .LinearSystems

import Base: ^

function lowerBlockTriangularVariable(model, m, n, n_blocks)
    # Returns a (m,n)-lower-block triangular matrix with T blocks of size (m,n).
    Phi = Matrix{Union{Float64,VariableRef}}(undef, m*n_blocks, n*n_blocks)
    Phi .= 0.0
    for i in 1:n_blocks
        for j in 1:i
            Phi[(i-1)*m+1:i*m,(j-1)*n+1:j*n] = @variable(model, [1:m,1:n])
        end
    end
    return Phi
end

function block_downshift(n::Int, T::Int)
    # Returns a matrix of size (nT,nT) with (n,n)-identity matrices on its first subdiagonal, and zeros elsewhere.
    # Shift matrix in block space (T×T) with 1's on subdiagonal
    S = diagm(-1 => ones(T-1))
    # Kronecker product with identity block of size n
    return kron(S, I(n))
end

function addSLPConstraint(model, A, B, T)
    # x(t+1)=Ax(t)+Bu(t)+w(t) for t=0,…,T-1
    (n_x, n_u) = size(B)

    calA = kron(I(T+1), A)
    calB = kron(I(T+1), B)
    downshiftOperator = block_downshift(n_x,T+1)

    ϕ_x = lowerBlockTriangularVariable(model, n_x, n_x, T+1)
    ϕ_u = lowerBlockTriangularVariable(model, n_u, n_x, T+1)

    # Add SLP constraint
    @constraint(model, [I-downshiftOperator*calA  -downshiftOperator*calB] * [ϕ_x; ϕ_u] .== I)

    return ϕ_x, ϕ_u
end

function addPolytopeInclusionConstraint(model, H_in, h_in, H_out, h_out)
    # Add constraint P_in ⊆ P_out, where P_j = {x | H_j*x ≤ h_j} for j∈{in,out}, using Farkas' lemma.
    (k_in, dim1) = size(H_in)
    (k_out, dim2) = size(H_out)
    dim1 == dim2 || throw(ArgumentError("The dimensions of the constraints must be equal."))

    Λ = @variable(model, [1:k_out, 1:k_in])
    @constraint(model, Λ .>= 0)
    @constraint(model, Λ*H_in .== H_out)
    @constraint(model, Λ*h_in .<= h_out)

    return Λ
end

function ^(p::HRepresentation, n::Int)
    # Cartesian power
    # TODO: code is not very efficient. Could be improved.
    res = p
    for i in 2:n
        res = res * p
    end
    return res
end

function safetyProblem(A, B, domain_x, domain_u, domain_w, domain_x0, T; contractConst=Inf, terminal_set=nothing)
    
    model = Model()
    ϕ_x, ϕ_u = addSLPConstraint(model, A, B, T)

    if isnothing(terminal_set)
        terminal_set = domain_x
    end
    prod_domain_x = domain_x^T * terminal_set # domain_x^(T+1)
    prod_domain_u = domain_u^(T+1)

    prod_noise = domain_x0 * domain_w^T

    Λ_x = addPolytopeInclusionConstraint(model, prod_noise.A, prod_noise.b, prod_domain_x.A*ϕ_x, prod_domain_x.b)
    Λ_u = addPolytopeInclusionConstraint(model, prod_noise.A, prod_noise.b, prod_domain_u.A*ϕ_u, prod_domain_u.b)

    if isfinite(contractConst)
        (dim_x, dim_u) = size(B)
        iseven(dim_x) || throw(ArgumentError("dim_x augmented system must be even."))
        dim_x_concrete = div(dim_x, 2) # dimenion of the non-augmented system
        for t in 0:T
            mat = ϕ_u[ t*dim_u+1:(t+1)*dim_u, t*dim_x+dim_x_concrete+1:(t+1)*dim_x]
            @assert size(mat) == (dim_u, dim_x - dim_x_concrete)
            #   || mat ||_∞→∞         ≤ contractivityConstant
            # ⟺ max_i ∑_j |mat[i,j]| ≤ contractivityConstant
            # ⟺ ∀i, ∑_j |mat[i,j]|   ≤ contractivityConstant
            # ⟺ ∀i, ||mat[i,:]||_1   ≤ contractivityConstant
            for i in 1:dim_u
                @constraint(model, [contractConst; mat[i,:].+0.0] in MOI.NormOneCone(dim_x_concrete+1) ) # +0.0 triggers a conversion to AffExpr
            end
        end
    end

    display(model)

    return model, ϕ_x, ϕ_u, Λ_x, Λ_u
end

function safetyProblem(sys::LinearSystem, domain_x0, T; contractConst=Inf, terminal_set=nothing)
    iszero(sys.c) || sys.D == I || throw(ArgumentError("sys.c=0 and sys.D=I required."))
    model, ϕ_x, ϕ_u, Λ_x, Λ_u = safetyProblem(sys.A, sys.B, sys.domain_x, sys.domain_u, hrep(sys.domain_w), domain_x0, T; contractConst=contractConst, terminal_set=terminal_set)
    return model, ϕ_x, ϕ_u, Λ_x, Λ_u
end

function get_policy(ϕ_x, ϕ_u, n_x, n_u)
    # returns a function describing the policy u = Kx with K=ϕ_u*inv(ϕ_x)
    # It works as u_t = policy(t, list_x), where list_x = [x_0, x_1, ..., x_t].
    (Tn_x,Tn_x2) = size(ϕ_x)
    (Tn_u,Tn_x3) = size(ϕ_u)
    Tn_x == Tn_x2 == Tn_x3 || throw(ArgumentError("The dimensions of ϕ_u and ϕ_x must match."))
    T = div(Tn_x, n_x)
    T2 = div(Tn_u, n_u)
    T == T2 || throw(ArgumentError("The dimensions of ϕ_u and ϕ_x must match."))

    K = ϕ_u*inv(ϕ_x)
    function policy(t, list_x)
        0 ≤ t ≤ T || throw(ArgumentError("t must be in [0,T] with T=$T."))
        length(list_x)==t+1 || throw(ArgumentError("Length of list_x must be t+1."))
        #length(list_x[1])==n_x || throw(ArgumentError("Length of list_x[1] must be n_x."))
        u = zeros(n_u)
        for τ in 0:t
            u += K[t*n_u+1:(t+1)*n_u, τ*n_x+1:(τ+1)*n_x]*list_x[τ+1]
        end
        return u
    end
    return policy
end

function concretize_augmented_policy(augmentedPolicy::Function, f::Function, sys::LinearSystem)
    # Returns a function that computes the trajectory of the augmented system
    # given an initial state x0 and a policy.
    n_x = sys.dim_x
    n_u = sys.dim_u
    n_w = sys.dim_w

    function concrete_policy(t, list_x, list_w)
        length(list_x)==t+1 || throw(ArgumentError("Length of list_x must be t+1=$(t+1)."))
        length(list_w)==t || throw(ArgumentError("Length of list_w must be t=$t (currently $(length(list_w)))."))
        length(list_x[1])==n_x || throw(ArgumentError("Length of list_x[1] must be n_x."))

        model = Model(Gurobi.Optimizer)
        @variable(model, u[1:n_u])
        @constraint(model, u ∈ sys.domain_u)

        next_w = f(list_x[t+1], u) - sys.A * list_x[t+1] - sys.B * u - sys.c
        list_w_tmp = vcat(list_w, [next_w])
        list_augmented_state = [vcat(list_x[i], list_w_tmp[i]) for i in eachindex(list_x)]

        @constraint(model, u .== augmentedPolicy(t, list_augmented_state)) # fixed point equation
        optimize!(model)
        if termination_status(model) != MOI.OPTIMAL
            error("The optimization did not finish successfully.")
        end

        return value.(u), value.(next_w)
    end
    return concrete_policy
end

function concretize_augmented_policy_contractivity(augmentedPolicy::Function, f::Function, sys::LinearSystem)
    n_x = sys.dim_x
    n_u = sys.dim_u
    n_w = sys.dim_w

    # Find an itinial u0 ∈ domain_u to initialize the fixed-point iteration
    model = Model(Gurobi.Optimizer)
    @variable(model, u0[1:n_u])
    @constraint(model, u0 ∈ sys.domain_u)
    optimize!(model)
    if termination_status(model) != MOI.OPTIMAL
        error("The optimization did not finish successfully.")
    end
    u0 = value.(u0)

    function concrete_policy(t, list_x, list_w; u0=u0, max_iter=1000, tol=1e-5)
        length(list_x)==t+1 || throw(ArgumentError("Length of list_x must be t+1=$(t+1)."))
        length(list_w)==t || throw(ArgumentError("Length of list_w must be t=$t (currently $(length(list_w)))."))
        length(list_x[1])==n_x || throw(ArgumentError("Length of list_x[1] must be n_x."))
        length(u0)==n_u || throw(ArgumentError("Length of u must be n_u."))

        # construct fixed-point operator
        function fixed_point_operator(u)
            next_w = f(list_x[t+1], u) - sys.A * list_x[t+1] - sys.B * u - sys.c
            list_w_tmp = vcat(list_w, [next_w])
            list_augmented_state = [vcat(list_x[i], list_w_tmp[i]) for i in eachindex(list_x)]
            return augmentedPolicy(t, list_augmented_state)
        end

        # iterate over the fixed-point operator
        for iter = 1:max_iter
            u1 = fixed_point_operator(u0)
            residual = norm(u1 - u0)
            println("Iter $iter: residual = $residual, u1 = $u1")
            if residual < tol
                sys.domain_u.A*u1 ≤ sys.domain_u.b .+ 10*tol || throw(ArgumentError("The computed input is not in the input constraint set.")) # u1 ∈ sys.domain_u
                return u1, f(list_x[t+1], u1) - sys.A * list_x[t+1] - sys.B * u1 - sys.c
            end
            u0 = u1
        end
        error("Max iteration reached. The fixed-point iteration did not converge.")
    end
    return concrete_policy
end

function simulate_trajectory(x0, f::Function, policy::Function, T::Int)
    list_x = [x0]
    list_u = []
    for t in 0:T-1
        u = policy(t, list_x)
        x0 = f(x0, u)
        push!(list_x, x0)
        push!(list_u, u)
    end
    return hcat(list_x...), hcat(list_u...) # dim_x x T, dim_u x T-1
end

function simulate_concrete_trajectory(x0, f::Function, concrete_policy::Function, T::Int)
    # Simulates the trajectory of the system sys with initial state x0 and time horizon T.
    # If minimizeInput is true, it minimizes the input at each step.

    list_x = [x0]
    list_w = Vector{Float64}[]
    list_u = Vector{Float64}[]
    for t in 0:T-1
        println("t = $t")
        u, w = concrete_policy(t, list_x, list_w)
        x0 = f(x0, u)
        push!(list_x, x0)
        push!(list_w, w)
        push!(list_u, u)
    end
    return hcat(list_x...), hcat(list_u...) # dim_x x T+1, dim_u x T
end

function augmentedSystem(sys::LinearSystem)
    if sys.dim_x != sys.dim_w
        @warn "sys.dim_x must be equal to sys.dim_w."
    end

    A = [sys.A sys.D; zeros(sys.dim_w, sys.dim_x+sys.dim_w)]
    B = [sys.B; zeros(sys.dim_w, sys.dim_u)]
    c = [sys.c; zeros(sys.dim_w)]
    D = Matrix{Float64}(I(sys.dim_x+sys.dim_w))
    
    domain_x = sys.domain_x * sys.domain_w
    domain_w = originPolyhedron(sys.dim_x) * sys.domain_w
    
    return LinearSystem(A, B, c, D, domain_x, sys.domain_u, domain_w)
end

function originPolyhedron(n::Int)
    # Returns the polyhedron {0}⊆R^n

    # { x | I*x ≤ 0 } ∩ { x | -I*x ≤ 0 }
    return polyhedron(intersect( hrep(Matrix{Float64}(I(n)),zeros(n)) , hrep(-Matrix{Float64}(I(n)),zeros(n)) ))
end


