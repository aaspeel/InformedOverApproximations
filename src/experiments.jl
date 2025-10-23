using Gurobi
using Plots
using LaTeXStrings

include("systemLevelSynthesis.jl")
include("overapproximations.jl")

markersize = 3
linewidth = 3
fontsize = 10

function inputAffineExperiment()
    ### PROBLEM DESCRIPTION

    α=0.5
    β=0.75
    γ=2.0
    f_cont(x,u) = [x[2] + α*x[1]^2; β*x[1]^2 + γ*x[2]^3 + cos(x[2])*u[1]]
    lipschitz_f_cont_dispersion = [1+2*α; 2+2*β+3*γ*0.5^2] # Note that this depends on domain_x and domain_u.

    dt = 0.1
    f(x, u) = x + dt*f_cont(x,u)

    H_domain_x = [1 0; -1 0; 0 1; 0 -1]
    h_domain_x = [1; 0; 0.5; 0.5]
    domain_x = hrep(H_domain_x, h_domain_x)

    H_domain_u = [1; -1;;] # must be a matrix, e.g., [1; -1;;]
    h_domain_u = [1; 1]
    domain_u = hrep(H_domain_u, h_domain_u)

    h_domain_x0 = [0.1; -0.1; 0.0; 0.0]
    domain_x0 = hrep(H_domain_x, h_domain_x0)

    h_terminal_set_informed = [1.0; -0.81; 0.5; 0.5]
    h_terminal_set_uninformed = [1.0; -0.62; 0.5; 0.5]
    # (α,β,γ) = (0.5,0.25,0.5) → 0.94 and 0.87 → Δ=0.07
    # (α,β,γ) = (0.5,0.25,1.0) → 0.94 and 0.86 → Δ=0.08
    # (α,β,γ) = (0.5,0.25,2.0) → 0.93 and 0.82 → Δ=0.11
    # (α,β,γ) = (0.5,0.5,2.0) → 0.88 and 0.74 → Δ=0.14
    # (α,β,γ) = (0.5,0.75,2.0) → 0.81 and 0.62 → Δ=0.19


    terminal_set_informed = hrep(H_domain_x, h_terminal_set_informed)
    terminal_set_uninformed = hrep(H_domain_x, h_terminal_set_uninformed)

    ### COMPUTE LINEAR OVER-APPROXIMATION
    dispersion = 0.03
    list_matrices, domain_w = linear_overapproximation(f, [domain_x, domain_u]; dispersion=dispersion)
    A = list_matrices[1]
    B = list_matrices[2]

    # Inflate W to have guarantees based on Lipschitz constants and dispersion.
    # f(x,u) = x + dt*f_cont(x,u)
    # e(x,u) = f(x,u) - A*x - B*u
    #        = dt*f_cont(x,u) - (A-I)*x - B*u
    # Lipschitz constant of e(x,u) w.r.t. (x,u) is ≤ dt*lipschitz_f_cont_dispersion + || [(A-I) B][i,:] ||_{∞→∞}
    for (i,r) in enumerate(eachrow( [(A-I) B] ))
        lipschitz_f_i = dt*lipschitz_f_cont_dispersion[i] + norm(r, 1) # the 1-norm of the vector r is the induced ∞→∞ norm of the matrix [r].
        domain_w.b[2*i-1] += lipschitz_f_i * dispersion
        domain_w.b[2*i]   += lipschitz_f_i * dispersion
        @info "Inflation of w_$i: $(lipschitz_f_i * dispersion)"
    end
    println("domain_w after inflation:")
    display(domain_w)

    sys = LinearSystem(A, B, domain_x, domain_u, polyhedron(domain_w))
    
    ### COMPUTE INFORMED POLICY
    T_inf = 30 # 40 ≤ T < 
    augmented_sys = augmentedSystem(sys)
    model, ϕ_x, ϕ_u, _, _ = safetyProblem(augmented_sys, domain_x0*domain_w, T_inf; terminal_set=terminal_set_informed*domain_w)

    set_optimizer(model, Gurobi.Optimizer)
    optimize!(model)
    solution_summary(model)
    if termination_status(model) != MOI.OPTIMAL
        error("INFORMED POLICY SYNTHESIS - The optimization did not finish successfully.")
    end

    augmented_policy = get_policy(value.(ϕ_x), value.(ϕ_u), sys.dim_x+sys.dim_w, sys.dim_u)
    policy_inf = concretize_augmented_policy(augmented_policy, f, sys)
    
    ### COMPUTE UNINFORMED POLICY
    T_uninf = T_inf # ? ≤ T < 40
    model, ϕ_x, ϕ_u, _, _ = safetyProblem(sys, domain_x0, T_uninf; terminal_set=terminal_set_uninformed )

    set_optimizer(model, Gurobi.Optimizer)
    optimize!(model)
    solution_summary(model)
    if termination_status(model) != MOI.OPTIMAL
        error("UNINFORMED POLICY SYNTHESIS - The optimization did not finish successfully.")
    end

    policy_uninf = get_policy(value.(ϕ_x), value.(ϕ_u), sys.dim_x, sys.dim_u)

    ### PLOTS

    # plot sets
    fig = Plots.plot(polyhedron(domain_x), alpha=0.1, label="Domain")
    #Plots.plot!(fig, polyhedron(domain_x0), color=:blue, alpha=0.3, label="Initial set")
    Plots.plot!(fig, polyhedron(terminal_set_uninformed), color=RGB(0.8,0.5,0.5), alpha=1, label="Terminal set (uninformed)")
    Plots.plot!(fig, polyhedron(terminal_set_informed), color=RGB(0.5,0.8,0.5), alpha=1, label="Terminal set (informed)")
    xlabel!(fig, L"x_1")
    ylabel!(fig, L"x_2")
    Plots.plot!(fig, legend=:bottomleft,
        titlefontsize=fontsize,
        guidefontsize=fontsize,
        tickfontsize=fontsize,
        xguidefontsize=1.5*fontsize,
        yguidefontsize=1.5*fontsize,
        legendfontsize=fontsize)
    display(fig)

    # simulate and plot trajectories
    list_initial_conditions = [[0.1; 0.0]] #, [0.5; 0.5], [0.5; 1.0], [1.0; 0.5]]
    for (i, x0) in enumerate(list_initial_conditions)
        trajectory_x_inf, trajectory_u_inf = simulate_concrete_trajectory(x0, f, policy_inf, T_inf)
        trajectory_x_uninf, trajectory_u_uninf = simulate_trajectory(x0, f, policy_uninf, T_uninf)
        if i == 1
            Plots.plot!(fig, trajectory_x_uninf[1,:], trajectory_x_uninf[2,:], markersize=markersize, linewidth=linewidth, color=:red, markershape=:square, label="Uninformed trajectory")
            Plots.plot!(fig, trajectory_x_inf[1,:], trajectory_x_inf[2,:], markersize=markersize, linewidth=linewidth, color=:green, markershape=:circle, label="Informed trajectory")
        else # no label
            Plots.plot!(fig, trajectory_x_uninf[1,:], trajectory_x_uninf[2,:], markersize=markersize, linewidth=linewidth, color=:red, markershape=:square, label = false)
            Plots.plot!(fig, trajectory_x_inf[1,:], trajectory_x_inf[2,:], markersize=markersize, linewidth=linewidth, color=:green, markershape=:circle, label = false)
        end
    end
    Plots.scatter!(fig, [0.1], [0.0], markersize=8, color=:black, markershape=:star , label="Initial state")
    display(fig)
    savefig(fig, "inputAffineExperiment.pdf")

    return sys
end

sys = inputAffineExperiment()

function nonlinearExperiment()
    ### PROBLEM DESCRIPTION
    T = 35
    dt = 0.1
    g = 9.81
    L = g
    k = 2.0
    α = 0.5

    f_cont(x,u) = [x[2]; - g/L*sin(x[1]) + α*x[2]^2 + k*sin(u[1])]
    lipschitz_f_cont_dispersion = [1.0; (g/L+k+α*(2^2))] # Component-wise Lipschitz constant of f_cont w.r.t. (x,u). [1.0; g/L + k + α*|x[2]|^2]

    # Euler discretization
    f(x, u) = x + dt*f_cont(x,u)
    lipschitz_f_contraction = dt*k # Lipschitz constant of f(x,u) w.r.t. u

    H_domain_x = [1 0; -1 0; 0 1; 0 -1]
    h_domain_x = [π+π/12; π/12; 2; 2]
    domain_x = hrep(H_domain_x, h_domain_x)

    H_domain_u = [1; -1;;] # must be a matrix, e.g., [1; -1;;]
    h_domain_u = [π/2; π/2]
    domain_u = hrep(H_domain_u, h_domain_u)

    h_domain_x0 = [0.1; -0.1; 0.0; 0.0]
    domain_x0 = hrep(H_domain_x, h_domain_x0)

    h_terminal_set_informed = [π+π/12; -(3.1); 2.0; 2.0]
    h_terminal_set_uninformed = [π+π/12; -(2.7); 2.0; 2.0]
    # (g,L,k,α) = (9.81,9.81,2.0,0.5) → 2.5 and 2.0 → Δ=0.5 (T=30) (π/6 instead of π/12)
    # (g,L,k,α) = (9.81,9.81,2.0,0.5) → 3.1 and 2.7 → Δ=0.4 (T=35) (π/12)

    terminal_set_informed = hrep(H_domain_x, h_terminal_set_informed)
    terminal_set_uninformed = hrep(H_domain_x, h_terminal_set_uninformed)

    ### COMPUTE LINEAR OVER-APPROXIMATION
    dispersion = 0.03
    list_matrices, domain_w = linear_overapproximation(f, [domain_x, domain_u]; dispersion=dispersion)
    A = list_matrices[1]
    B = list_matrices[2]

    # Inflate W to have guarantees based on Lipschitz constants and dispersion.
    # f(x,u) = x + dt*f_cont(x,u)
    # e(x,u) = f(x,u) - A*x - B*u
    #        = dt*f_cont(x,u) - (A-I)*x - B*u
    # Lipschitz constant of e(x,u) w.r.t. (x,u) is ≤ dt*lipschitz_f_cont_dispersion + || [(A-I) B][i,:] ||_{∞→∞}
    for (i,r) in enumerate(eachrow( [(A-I) B] ))
        lipschitz_f_i = dt*lipschitz_f_cont_dispersion[i] + norm(r, 1) # the 1-norm of the vector r is the induced ∞→∞ norm of the matrix [r].
        domain_w.b[2*i-1] += lipschitz_f_i * dispersion
        domain_w.b[2*i]   += lipschitz_f_i * dispersion
        @info "Inflation of w_$i: $(lipschitz_f_i * dispersion)"
    end

    sys = LinearSystem(A, B, domain_x, domain_u, polyhedron(domain_w))
    println("The linearized system is:")
    display(sys)

    ### COMPUTE INFORMED POLICY
    augmented_sys = augmentedSystem(sys)

    contractConst = 1.0/(lipschitz_f_contraction + opnorm(B,Inf) + 1e-5) # +1e-5 to avoid numerical issues and have an overall Lipschitz constant < 1 (and not ≤ 1)
    println("\nUsing contractivity constant = $contractConst\n")

    model, ϕ_x, ϕ_u, _, _ = safetyProblem(augmented_sys, domain_x0*domain_w , T; contractConst=contractConst, terminal_set=terminal_set_informed*domain_w )

    set_optimizer(model, Gurobi.Optimizer)
    optimize!(model)
    solution_summary(model)
    if termination_status(model) != MOI.OPTIMAL
        error("INFORMED POLICY SYNTHESIS - The optimization did not finish successfully.")
    end

    augmented_policy = get_policy(value.(ϕ_x), value.(ϕ_u), sys.dim_x+sys.dim_w, sys.dim_u)
    policy_inf = concretize_augmented_policy_contractivity(augmented_policy, f, sys)

    ### COMPUTE UNINFORMED POLICY
    model, ϕ_x, ϕ_u, _, _ = safetyProblem(sys, domain_x0, T; terminal_set=terminal_set_uninformed)

    set_optimizer(model, Gurobi.Optimizer)
    optimize!(model)
    solution_summary(model)
    if termination_status(model) != MOI.OPTIMAL
        error("UNINFORMED POLICY SYNTHESIS - The optimization did not finish successfully.")
    end

    policy_uninf = get_policy(value.(ϕ_x), value.(ϕ_u), sys.dim_x, sys.dim_u)

    ### PLOTS
    # plot sets
    fig = Plots.plot(polyhedron(domain_x), alpha=0.1, label="Domain")
    #Plots.plot!(fig, polyhedron(domain_x0), color=:blue, alpha=0.3, label="Initial set")
    Plots.plot!(fig, polyhedron(terminal_set_uninformed), color=RGB(0.8,0.5,0.5), alpha=1.0, label="Terminal set (uninformed)")
    Plots.plot!(fig, polyhedron(terminal_set_informed), color=RGB(0.5,0.8,0.5), alpha=1.0, label="Terminal set (informed)")
    xlabel!(fig, L"x_1")
    ylabel!(fig, L"x_2")
    Plots.plot!(fig, legend=:bottomleft,
        titlefontsize=fontsize,
        guidefontsize=fontsize,
        tickfontsize=fontsize,
        xguidefontsize=1.5*fontsize,
        yguidefontsize=1.5*fontsize,
        legendfontsize=fontsize)
    display(fig)

    # simulate and plot trajectories
    list_initial_conditions = [[0.1; 0.0]] #, [0.5; 0.5], [0.5; 1.0], [1.0; 0.5]]
    for (i, x0) in enumerate(list_initial_conditions)
        trajectory_x_uninf, trajectory_u_uninf = simulate_trajectory(x0, f, policy_uninf, T)
        trajectory_x_inf, trajectory_u_inf = simulate_concrete_trajectory(x0, f, policy_inf, T)
        if i == 1
            Plots.plot!(fig, trajectory_x_uninf[1,:], trajectory_x_uninf[2,:], markersize=markersize, linewidth=linewidth, color=:red, markershape=:square, label="Uninformed trajectory")
            Plots.plot!(fig, trajectory_x_inf[1,:], trajectory_x_inf[2,:], markersize=markersize, linewidth=linewidth, color=:green, markershape=:circle, label="Informed trajectory")
        else # no label
            Plots.plot!(fig, trajectory_x_uninf[1,:], trajectory_x_uninf[2,:], markersize=markersize, linewidth=linewidth, color=:red, markershape=:square, label = false)
            Plots.plot!(fig, trajectory_x_inf[1,:], trajectory_x_inf[2,:], markersize=markersize, linewidth=linewidth, color=:green, markershape=:circle, label = false)
        end
    end
    Plots.scatter!(fig, [0.1], [0.0], markersize=8, color=:black, markershape=:star , label="Initial state")
    display(fig)
    savefig(fig, "nonlinearExperiment.pdf")

    return sys
end

sys = nonlinearExperiment()