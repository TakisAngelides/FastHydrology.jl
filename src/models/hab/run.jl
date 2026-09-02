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
