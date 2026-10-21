#################################
# Model: Kazmierczak et al 2024 #
#################################


"""
$(TYPEDSIGNATURES)

Update the model and state variables for the Kazmierczak et al 2024 model steady-state calculation.
We specifically update first the distributed water flux q, then the effective pressure N, and then the
water layer thickness W -- W must come after N because its `ArealConduitThickness` closure (see
`AbstractWaterThicknessAlgorithm` in water_thickness.jl) reads `model.S_inf`, which `update_N!` (via its own
`update_S_inf!` call) is what keeps current. `state.W` itself feeds back into nothing else in the
model, so computing it last is safe.

When `update_q!` solves q and N together (an N-dependent `model.sliding_law` or sensible heat, see
`uses_qN_loop` and water_flux.jl's `resolve_q!`), its last sweep already ran `update_N!` on the final
q, so the `update_N!` here is skipped: it would recompute the same values (~40% of a solve).
"""
function update_steady_state!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)
    update_q!(model, grid, state)
    (uses_qN_loop(model) && model.max_qN_iters > 0) || update_N!(model, grid, state)
    update_W!(model, grid, state)
end
