using Polyhedra

function hrep_full_space(dim::Int)
    # return R^dim in HRepresentation
    return hrep( Array{Float64}(undef, 0, dim) , Array{Float64}(undef, 0) ) # R^{dim}, i.e., polytope without any constraint
end

function axis_aligned_H(n::Int)
    #= return the H-representation of a n-dimensional axis aligned hyperbox centered around the origin, using the form
    # H = [1  0 ... 0;
           -1 0 ... 0;
           0  1 0 ... 0;
           0 -1 0 ... 0]
    =#
    M = zeros(Int, 2n, n)
    for i in 1:n
        M[2i - 1, i] = 1
        M[2i, i] = -1
    end
    return M
end

function range_dispersion(;start::Real, stop::Real, dispersion::Real)
    start <= stop || error("start must be less than or equal to stop")
    dispersion >= 0 || error("dispersion must be non-negative")

    new_start = start + dispersion
    new_stop = stop - dispersion
    if new_start >= new_stop # Interval smaller than 2*dispersion, return the middle point.
        middle = (start + stop) / 2
        return range(middle, middle)
    end

    L = new_stop - new_start
    step = 2 * dispersion
    # minimum number of steps needed so spacing ≤ step
    n = ceil(Int, L / step)
    
    # generate n+1 evenly spaced points
    return range(new_start, new_stop; length = n+1)
end

function grid_polytope(pol::HRepresentation; n_steps::Int=0, step::Real=0.0, n_samples::Int=0, dispersion::Real=0.0)
    # Returns a list of points sampled from a grid inside a polytope. The polytope must be an axis-aligned box.

    if sum([n_steps != 0, step != 0, n_samples != 0, dispersion != 0]) != 1
        error("One and only one of the keyword arguments 'n_steps', 'step', 'n_samples', and 'dispersion' can be non zero. We received: n_steps=$(n_steps), step=$(step), n_samples=$(n_samples), dispersion=$(dispersion).")
    end

    n_dims = size(pol.A, 2)

    @assert pol.A == axis_aligned_H(n_dims) "The polytope must be an axis-aligned box in canonical form."

    bounds = reshape(pol.b, 2, n_dims)
    lower_bounds = -bounds[2, :]
    upper_bounds = bounds[1, :]

    # If n_samples is provided, compute n_steps in each dimension
    if n_samples != 0
        n_steps = ceil(Int64, n_samples^(1/n_dims) ) # number of steps in each dimension
        if n_steps^n_dims != n_samples
            @warn("n_steps=n_samples^(1/dimension) is not an integer. The number of samples has been ceiled and will be $(n_steps^n_dims).")
        end
    end

    if dispersion != 0
        ranges = [range_dispersion(start=lower_bounds[i], stop=upper_bounds[i], dispersion=dispersion) for i in 1:n_dims] # This calls the custom range function defined above.
    elseif step != 0
        ranges = [range(start=lower_bounds[i], stop=upper_bounds[i], step=step) for i in 1:n_dims]
    elseif n_steps != 0 # This should always be true here.
        ranges = [range(start=lower_bounds[i], stop=upper_bounds[i], length=n_steps) for i in 1:n_dims]
    else
        @assert false "This should never happen."
    end

    # Generate grid points as a list of vectors
    grid_points = []
    for point in Iterators.product(ranges...)
        push!(grid_points, collect(point))
    end

    return grid_points
end