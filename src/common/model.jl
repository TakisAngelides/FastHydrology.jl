"""
$(TYPEDSIGNATURES)

An abstract type for the hydrology model to be simulated. The model can hold revelant constants and model-specific fields.
"""
abstract type AbstractHydroModel end


"""
$(TYPEDSIGNATURES)

Whether the model's effective pressure `N` responds to the host's basal sliding speed, so that a host
whose velocity solve depends on `N` should re-evaluate `N` inside that solve with [`N_from_ub!`](@ref)
rather than once per step from the previous velocity. A steady hydrology (no state of its own between
steps, `N` set by the current ice state) has to be: evaluated once per step, `N` and `u_b` alternate
between two states. A model that carries `N` as prognostic state (Shakti) or does not depend on `u_b`
(HAB) returns `false` and needs nothing.
"""
N_responds_to_ub(::AbstractHydroModel) = false
