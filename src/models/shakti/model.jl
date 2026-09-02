#################################
# Model: Shakti                 #
#################################


"""
$(TYPEDSIGNATURES)

Wraps a `Shakti.Simulation` (see the separate `Shakti.jl` package) so it can be run as an
`AbstractHydroModel`'s time evolution via `TimeSimulation`. Shakti's `Simulation` already owns its
own grid, state, and model parameters, so this wrapper holds nothing else -- FastHydrology's
`run!(::TimeSimulation)` for this model just delegates straight to `Shakti.run!`.

Defined unconditionally (no dependency on `Shakti` itself, since the field is generic over `S`), but
only usable once `Shakti` is loaded alongside `FastHydrology`: the `run!` method that dispatches on
this type is added by the `FastHydrologyShaktiExt` package extension (see `ext/FastHydrologyShaktiExt.jl`),
which Julia loads automatically once both packages are in scope (`using FastHydrology, Shakti`).

# Arguments

- `sim`: a `Shakti.Simulation` built the usual way (`Shakti.Simulation(grid, state, tsteps, dt, p, ...)`).
"""
struct ShaktiHydroModel{S} <: AbstractHydroModel
    sim::S
end
