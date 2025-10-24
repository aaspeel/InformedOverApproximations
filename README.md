# InformedOverApproximations


This repository contains all the code necessary to implement the experiments in

* Antoine Aspeel, Antoine Girard, and Thiago Alves Lima. **Title**.

The code uses Julia-1.10 and requires a Gurobi license. Free Gurobi licenses are available for academics. More information [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

Use the Julia REPL to activate the environment by running:

`import Pkg; Pkg.activate("Project.toml")`

Then, to reproduce the experiments, run:

`include("src/experiments.jl")`.

Other files:
* `LinearSystems.jl` contains a structure to represent linear systems.
* `overapproximations.jl` contains the functions to compute a linear over-approximation of a nonlinear function.
* `systemLevelSynthesis.jl` contains functions related to policy synthesis using system level synthesis. It also contains the methods to concretize an informed policy.
* `utils.jl` contains functions to sample a polytope. These are used in `overapproximations.jl`.
