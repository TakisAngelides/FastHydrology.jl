"""
$(TYPEDSIGNATURES)

Update the hydrostatic ice overburden pressure Po based on ice thickness h.
Defined for all models that carry a Po field and share the same rho_i, g constants.
"""
function update_Po!(model::AbstractHydroModel, grid::AbstractHydroGrid, state::HydroState)

    @. model.Po = max(model.rho_i * model.g * state.h, 1e5)
    fill_halo!(model.Po, grid)

    return nothing

end
