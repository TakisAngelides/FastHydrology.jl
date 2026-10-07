#=
# [Coupling to Shakti.jl](@id ShaktiCoupling)
This is an example of how to use the [Shakti.jl](https://github.com/TakisAngelides/Shakti.jl)
subglacial hydrology solver (Sommers et al. 2018, https://gmd.copernicus.org/articles/11/2955/2018/) from
within FastHydrology.jl, via the `ShaktiHydroModel`/`TimeSimulation` combination provided by the
`FastHydrologyShaktiExt` package extension. Unlike `KazmierczakHydroModel`/`HABHydroModel` above,
Shakti is a genuinely time-evolving solver with its own grid, state, and Picard-iteration solve for
the hydraulic head at every timestep -- FastHydrology just wraps it and delegates to it, rather
than reimplementing any of its physics.

The extension activates automatically once both packages are loaded, so the only extra step
compared to the other examples is `using Shakti` alongside `using FastHydrology`.

Note: `FastHydrology` and `Shakti` both export a function called `run!` (and `step!`) -- unrelated
generic functions that happen to share a name. Loading both packages makes the bare name ambiguous,
so this example always calls `FastHydrology.run!`/`FastHydrology.step!` qualified.
=#

using FastHydrology
using Shakti
using CairoMakie

# ## A minimal Shakti simulation
#
# Shakti owns its own grid/state, independent of FastHydrology's `AbstractHydroGrid`/`HydroState`.
# Here we build a small, synthetic domain: a sloped bed, mixed grounded/ocean/land/other-basin
# boundary cells, and a single point-source moulin feeding meltwater in at one interior cell.

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
zb     = repeat(reshape(-0.02 .* grid.x, nx, 1), 1, ny) # sloped bed
zs     = zb .+ 1000.0
b      = fill(0.01, nx, ny) # initial gap height
G      = fill(0.06, nx, ny) # geothermal heat flux
ub_x   = fill(1e-6, nx + 1, ny)
ub_y   = zeros(nx, ny + 1)
ieb    = zeros(nx, ny)
ieb[5, 5] = 3 / (grid.dx * grid.dy) # a single moulin
taub_x = zeros(nx + 1, ny)
taub_y = zeros(nx, ny + 1)

Shakti.set_initial_conditions!(state, grid, p, mi, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

# A direct (Cholesky) solve backs the elliptic head scheme's Picard iteration.

ls = Shakti.CholeskyDirectSolver(grid)
ps = Shakti.PicardSolver(500, 1e-6, ls, grid)

# ## Wrapping it as a `ShaktiHydroModel`
#
# `Shakti.Simulation` bundles the grid, state, model parameters, and every scheme choice
# (head/gap/sliding-law/...) together -- `ShaktiHydroModel` just wraps that as an
# `AbstractHydroModel`, and `TimeSimulation` wraps the model, exactly like `SteadyStateSimulation`
# wraps `KazmierczakHydroModel`/`HABHydroModel` above.

dt = 3600.0 # 1 hour
tsteps = 6
shakti_sim = Shakti.Simulation(grid, state, tsteps, dt, p, "implicit", String[], mi, sl; ps = ps)

model    = ShaktiHydroModel(shakti_sim)
time_sim = TimeSimulation(model)

# ## Running the whole simulation
#
# `FastHydrology.run!` delegates straight to `Shakti.run!`, which advances all `tsteps` timesteps
# -- this is the same as calling `Shakti.run!(shakti_sim)` directly.

FastHydrology.run!(time_sim)

fig, ax, hm = heatmap(Array(shakti_sim.state.N); axis = (; title = "Effective pressure N [Pa]"))
Colorbar(fig[1, 2], hm)
fig
