module LinearSystems
export LinearSystem

using Polyhedra
using LinearAlgebra

import Base: display # function Base.display must be explicitly imported to be extended to new types

struct LinearSystem
    # x^+ = A x + B u + c + D w, with x ∈ domain_x, u ∈ domain_u, w ∈ domain_w.
    A::Matrix{Float64}
    B::Matrix{Float64}
    c::Vector{Float64} # default is 0
    D::Matrix{Float64} # default is I

    domain_x::HRepresentation # Hrep
    domain_u::HRepresentation # Hrep
    domain_w::Polyhedron # we need both the HRepresentation and the VRepresentation

    dim_x::Int64
    dim_u::Int64
    dim_w::Int64
    
    # e(x,u) or f(x,u) ?
end

function LinearSystem(A, B, c, D, domain_x, domain_u, domain_w)
    (dim_x, dim_u) = size(B)

    if isnothing(c)
        c = zeros(dim_x)
    end
    if isnothing(D)
        D = Matrix(I, dim_x, dim_x)
    end

    (_,dim_w) = size(D)

    fulldim(domain_x) == dim_x || throw(ArgumentError("Dimension mismatch: dim_x != fulldim(domain_x)"))
    fulldim(domain_u) == dim_u || throw(ArgumentError("dimension mismatch: dim_u != fulldim(domain_u)"))
    fulldim(domain_w) == dim_w || throw(ArgumentError("dimension mismatch: dim_w != fulldim(domain_w)"))
    size(A) == (dim_x, dim_x) || throw(ArgumentError("dimension mismatch: size(A) != (dim_x, dim_x)"))
    size(c) == (dim_x,) || throw(ArgumentError("dimension mismatch: size(c) != (dim_x,)"))
    size(D) == (dim_x, dim_w) || throw(ArgumentError("dimension mismatch: size(D) != (dim_x, dim_w)"))

    return LinearSystem(A, B, c, D, domain_x, domain_u, domain_w, dim_x, dim_u, dim_w)
end

function LinearSystem(A, B, domain_x, domain_u, domain_w)
    return LinearSystem(A, B, nothing, nothing, domain_x, domain_u, domain_w)
end

function LinearSystem(A, B, c::Vector{Float64}, domain_x, domain_u, domain_w)
    return LinearSystem(A, B, c, nothing, domain_x, domain_u, domain_w)
end

function LinearSystem(A, B, D::Matrix{Float64}, domain_x, domain_u, domain_w)
    return LinearSystem(A, B, nothing, D, domain_x, domain_u, domain_w)
end

function display(sys::LinearSystem)
    println("A:")
    show(stdout, "text/plain",sys.A)
    println()

    println("B:")
    show(stdout, "text/plain",sys.B)
    println()

    println("c:")
    show(stdout, "text/plain",sys.c)
    println()

    println("D:")
    show(stdout, "text/plain",sys.D)
    println()

    println("domain_x with dim_x=$(sys.dim_x):")
    show(stdout, "text/plain",sys.domain_x)
    println()

    println("domain_u with dim_u=$(sys.dim_u):")
    show(stdout, "text/plain",sys.domain_u)
    println()

    println("domain_w with dim_w=$(sys.dim_w):")
    show(stdout, "text/plain",sys.domain_w)
    println()

end

end