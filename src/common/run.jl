"""
$(TYPEDSIGNATURES)

A function to run a steady-state simulation.
"""
function run!(sim::SteadyStateSimulation)
    update_steady_state!(sim.model, sim.grid, sim.state)
end


"""
$(TYPEDSIGNATURES)

Advances a `TimeSimulation` by a single timestep, dispatching on its model type -- e.g. for
`ShaktiHydroModel`, the `FastHydrologyShaktiExt` package extension adds a method that calls
`Shakti.step!` directly (see `ext/FastHydrologyShaktiExt.jl`).

Exists as a stub here (no methods) so the name is owned by `FastHydrology` and can be extended by
package extensions, and so a coupled driver (e.g. an ice flow model updating geometry every
timestep) can call `step!` in its own loop instead of using `run!`, which owns the whole time loop
itself.
"""
function step! end
