####################################
# Model: Height above buoyancy (HAB)
####################################


"""
$(TYPEDSIGNATURES)

Update the effective pressure N across the grid using a complementary error function
transition between geometric potential and far-field effective pressure.
"""
function update_N!(model::HABHydroModel, grid::AbstractHydroGrid, state::HydroState)

    update_Po!(model, grid, state)
    update_p_w!(model, grid, state)

    @. state.N = max(model.Po - model.p_w, model.N_min)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Update the water pressure p_w.
"""
function update_p_w!(model::HABHydroModel, grid::AbstractHydroGrid, state::HydroState)

    # Scalar coefficient precomputed once rather than re-evaluated per cell inside the broadcast.
    neg_coeff = -model.P_w * model.rho_sw * model.g
    @. model.p_w = neg_coeff * state.b
    overwrite_where!(grid, model.p_w, state.mask, !=(GROUNDED), state.h; scale = model.P_w * model.rho_i * model.g)
    overwrite_where!(grid, model.p_w, state.b, >=(0.0), 0.0)

    return nothing

end
