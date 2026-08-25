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


#################################
# Model: Kazmierczak et al 2024 #
#################################


"""
$(TYPEDSIGNATURES)

Update the model and state variables for the Kazmierczak et al 2024 model steady-state calculation.
We specifically update first the distributed water flux q, then the effective pressure N, and then the
water layer thickness W -- W must come after N because two of its closures (`ConduitThickness`,
`ArealConduitThickness`; see `AbstractWaterThicknessAlgorithm` in model.jl) read `model.H`/`model.S_inf`,
which `update_N!` (via its own `update_H!`/`update_S_inf!` calls) is what keeps current. `state.W`
itself feeds back into nothing else in the model, so computing it last is safe.

When `model.sliding_law` is N-dependent (see water_flux.jl's `resolve_q!`), `update_q!` already
converges q and N together internally, so the explicit `update_N!`/`update_W!` calls below just
recompute the same converged values from the same final q -- redundant but harmless (both are pure
functions of q and the unchanged geometry/state), kept for a uniform call sequence across all
sliding-law choices.
"""
function update_steady_state!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)
    update_q!(model, grid, state)
    update_N!(model, grid, state)
    update_W!(model, grid, state)
end


####################################
# Model: Height above buoyancy (HAB) #
####################################


"""
$(TYPEDSIGNATURES)

Update the state variable of effective pressure according to Eq. (3) of the paper Kazmierczak et al 2022 (https://doi.org/10.5194/tc-16-4537-2022).
The model essentially assumes that ocean water infiltrates the ice sheet from the grounding line upwards to grounded ice regions where the bedrock
is below sea level.
"""
function update_steady_state!(model::HABHydroModel, grid::AbstractHydroGrid, state::HydroState)
    update_N!(model, grid, state)
end
