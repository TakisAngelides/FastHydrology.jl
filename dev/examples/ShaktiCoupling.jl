#=
# [Shakti](@id Shakti-example)
Shakti is a transient model, it owns its grid and state and is solved at every time step. FastHydrology
only wraps it in `ShaktiHydroModel` and `TimeSimulation`, all the physics lives in
[Shakti.jl](https://github.com/TakisAngelides/Shakti.jl), see there for the model, its options and
real-data examples.

Both packages export `run!` and `step!`, so call them as `FastHydrology.run!`.
=#

using FastHydrology
using Shakti
using CairoMakie

# A small Shakti domain: sloped bed, ocean/land/other-basin boundaries, one moulin.
nx, ny = 10, 10
grid  = Shakti.Grid(nx, ny, 1e4, 1e4)
state = Shakti.State(grid)
p     = Shakti.ModelParameters(e_v = 0.0)
mi    = Shakti.ConstantMeltInput()
sl    = Shakti.RegularizedCoulombSlidingLaw(0.25)

mask = fill(Shakti.GROUNDED, nx, ny)
mask[end, :] .= Shakti.OCEAN
mask[1, :]   .= Shakti.LAND
mask[:, 1]   .= Shakti.OTHER_BASIN

A_visc = fill(5e-25, nx, ny)
zb     = repeat(reshape(-0.02 .* grid.x, nx, 1), 1, ny)  # sloped bed
zs     = zb .+ 1000.0
b      = fill(0.01, nx, ny)                              # initial gap height
G      = fill(0.06, nx, ny)
ub_x   = fill(1e-6, nx + 1, ny)
ub_y   = zeros(nx, ny + 1)
ieb    = zeros(nx, ny)
ieb[5, 5] = 3 / (grid.dx * grid.dy)                      # moulin
taub_x = zeros(nx + 1, ny)
taub_y = zeros(nx, ny + 1)

Shakti.set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

ls = Shakti.CholeskyDirectSolver(grid)
ps = Shakti.PicardSolver(500, 1e-6, ls, grid)

# Wrap it, 6 steps of 1 hour.
dt, tsteps = 3600.0, 6
shakti_sim = Shakti.Simulation(grid, state, tsteps, dt, p, "implicit", String[], mi, sl; ps = ps)
model      = ShaktiHydroModel(shakti_sim)
time_sim   = TimeSimulation(model)

# `FastHydrology.run!` runs all steps, `FastHydrology.step!` does one.
FastHydrology.run!(time_sim)

fig, ax, hm = heatmap(Array(shakti_sim.state.N); axis = (; title = "Effective pressure N [Pa]"))
Colorbar(fig[1, 2], hm)
fig
