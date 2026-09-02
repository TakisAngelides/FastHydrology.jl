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

    @. state.N = max(model.Po - model.p_w, model.sigmat * model.Po)
    fill_halo!(state.N, grid)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Update the water pressure p_w.
"""
function update_p_w!(model::HABHydroModel, grid::AbstractHydroGrid, state::HydroState)

    # The scalar coefficient must be precomputed outside the broadcast: Oceananigans'
    # AbstractOperation conversion walks the whole broadcast tree structurally, and a
    # nested scalar-only sub-expression (here `-model.P_w`) breaks when embedded inside
    # a larger broadcast that also involves Fields.
    neg_coeff = -model.P_w * model.rho_sw * model.g
    @. model.p_w = neg_coeff * state.b
    overwrite_where!(grid, model.p_w, state.mask, ==(0.0), state.h; scale = model.P_w * model.rho_i * model.g)
    overwrite_where!(grid, model.p_w, state.b, >=(0.0), 0.0)
    fill_halo!(model.p_w, grid)

    return nothing

end
