"""
Package extension activated automatically once both `FastHydrology` and `Shakti` are loaded
(`using FastHydrology, Shakti`). Adds the `run!`/`step!` methods that let a `TimeSimulation`
wrapping a `ShaktiHydroModel` (see `src/model.jl`) actually run, by delegating to `Shakti.run!`/
`Shakti.step!` -- Shakti's own `run!` already handles the whole time loop, checkpointing, and
observer output, so there is nothing else to reimplement for that entry point. `step!` exists
separately for callers that need to drive the loop themselves, e.g. a coupled ice flow model that
updates geometry every timestep before/after advancing the hydrology by one step.
"""
module FastHydrologyShaktiExt

using FastHydrology
using Shakti

"""
    run!(sim::TimeSimulation{<:ShaktiHydroModel}; kwargs...)

Runs `sim` by delegating straight to `Shakti.run!(sim.model.sim; kwargs...)`. Any keyword accepted
by `Shakti.run!` (e.g. `checkpoint_every`, `checkpoint_path`, `restart_path`) can be passed through
here.
"""
function FastHydrology.run!(sim::FastHydrology.TimeSimulation{<:FastHydrology.ShaktiHydroModel}; kwargs...)
    Shakti.run!(sim.model.sim; kwargs...)
end

"""
    step!(sim::TimeSimulation{<:ShaktiHydroModel})

Advances `sim` by a single timestep: calls `Shakti.step!(sim.model.sim)` then advances
`sim.model.sim.total_time[]` by `sim.model.sim.dt`, mirroring one iteration of `Shakti.run!`'s own
loop (minus checkpointing/observer output, which are `run!`-level concerns). The `total_time[]`
advance matters even for constant-in-time melt input: Shakti's `step_h!` reads `sim.total_time[]`
for time-dependent forcing (e.g. `SeasonalMeltInput`), and only `run!`'s loop normally keeps it in
sync -- calling `Shakti.step!` directly without also bumping it would silently reuse the same
forcing time on every call.

Intended for a coupled driver (e.g. an ice flow model) that owns its own timestep loop: update ice
geometry into `sim.model.sim.state` (`zb`, `zs`, `H`, `mask`, ...) directly, call this, then read
back e.g. `sim.model.sim.state.N` for the next ice flow step.
"""
function FastHydrology.step!(sim::FastHydrology.TimeSimulation{<:FastHydrology.ShaktiHydroModel})
    shakti_sim = sim.model.sim
    Shakti.step!(shakti_sim)
    shakti_sim.total_time[] += shakti_sim.dt
    return nothing
end

end
