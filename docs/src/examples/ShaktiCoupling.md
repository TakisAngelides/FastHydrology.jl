```@meta
EditURL = "ShaktiCoupling.jl"
```

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

````@example ShaktiCoupling
using FastHydrology
using Shakti
using CairoMakie
````

## A minimal Shakti simulation

Shakti owns its own grid/state, independent of FastHydrology's `AbstractHydroGrid`/`HydroState`.
Here we build a small, synthetic domain: a sloped bed, mixed grounded/ocean/land/other-basin
boundary cells, and a single point-source moulin feeding meltwater in at one interior cell.

````@example ShaktiCoupling
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
````

A direct (Cholesky) solve backs the elliptic head scheme's Picard iteration.

````@example ShaktiCoupling
ls = Shakti.CholeskyDirectSolver(grid)
ps = Shakti.PicardSolver(500, 1e-6, ls, grid)
````

## Wrapping it as a `ShaktiHydroModel`

`Shakti.Simulation` bundles the grid, state, model parameters, and every scheme choice
(head/gap/sliding-law/...) together -- `ShaktiHydroModel` just wraps that as an
`AbstractHydroModel`, and `TimeSimulation` wraps the model, exactly like `SteadyStateSimulation`
wraps `KazmierczakHydroModel`/`HABHydroModel` above.

````@example ShaktiCoupling
dt = 3600.0 # 1 hour
tsteps = 6
shakti_sim = Shakti.Simulation(grid, state, tsteps, dt, p, "implicit", String[], mi, sl; ps = ps)

model    = ShaktiHydroModel(shakti_sim)
time_sim = TimeSimulation(model)
````

## Running the whole simulation

`FastHydrology.run!` delegates straight to `Shakti.run!`, which advances all `tsteps` timesteps
-- this is the same as calling `Shakti.run!(shakti_sim)` directly.

````@example ShaktiCoupling
FastHydrology.run!(time_sim)

fig, ax, hm = heatmap(Array(shakti_sim.state.N); axis = (; title = "Effective pressure N [Pa]"))
Colorbar(fig[1, 2], hm)
fig
````

## Coupling to an ice flow model

A coupled ice-flow simulation typically needs to update the ice geometry every timestep and only
then advance the hydrology by one step, rather than handing the whole time loop over to
`Shakti.run!`. `FastHydrology.step!` is the entry point for that: it advances the wrapped
`Shakti.Simulation` by exactly one timestep (`Shakti.step!`, plus the matching `total_time[]`
update that `Shakti.run!`'s own loop would otherwise apply), and leaves everything else --
mutating geometry beforehand, reading fields back out afterward -- to the caller.

````@example ShaktiCoupling
for i in 1:3
    # Stand-in for an ice flow model lowering the surface by 0.5 m this step.
    shakti_sim.state.zs .-= 0.5

    FastHydrology.step!(time_sim)

    # `shakti_sim.state.N` now reflects the updated geometry, ready to feed into e.g. a sliding
    # law for the next ice flow step.
end

heatmap(Array(shakti_sim.state.N); axis = (; title = "Effective pressure N [Pa] after coupled steps"))
````

