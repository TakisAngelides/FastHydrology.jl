#################################
# Model: Kazmierczak et al 2024 #
#################################


"""
$(TYPEDSIGNATURES)

Update the effective pressure N across the grid using a complementary error function
transition between geometric potential and far-field effective pressure.
"""
function update_N!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)

    update_Q!(model, grid) # volumetric water flux [m3 s-1] per conduit
    update_S_inf!(model, grid) # cross-sectional area of conduits
    update_H!(model, grid) # thickness of conduits
    update_Po!(model, grid, state) # ice overburden pressure rho*g*h
    update_N_inf!(model, grid) # far-field effective pressure

    field_data(state.N) .= max.(0.0, erf.((sqrt(pi) .* field_data(model.phi0)) ./ (2 .* field_data(model.N_inf))) .* field_data(model.N_inf))
    fill_halo!(state.N, grid)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Update the hydrostatic ice overburden pressure Po based on ice thickness h.
Defined for all models that carry a Po field and share the same rho_i, g constants.
"""
function update_Po!(model::AbstractHydroModel, grid::AbstractHydroGrid, state::HydroState)

    field_data(model.Po) .= max.(model.rho_i * model.g * field_data(state.h), 1e5)
    fill_halo!(model.Po, grid)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Update the conduit thickness H by calculating separate values for hard and soft beds,
then interpolating based on the bed heterogeneity indicator kappa.
"""
function update_H!(model::KazmierczakHydroModel, grid::AbstractHydroGrid)

    @. model.H_hard = sqrt(model.S_inf)
    field_data(model.H_soft) .= max.(0.0, model.H_0 .+ (sqrt.(field_data(model.S_inf)) ./ model.F_till .- model.H_0) .* exp.(-field_data(model.Q) / model.Q_c))
    @. model.H = (1 - model.kappa) * model.H_hard + model.kappa * model.H_soft
    fill_halo!(model.H, grid)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Update the far-field conduit cross-sectional area S_inf using the Manning or
Gauckler-Manning-Strickler flow law.
"""
function update_S_inf!(model::KazmierczakHydroModel, grid::AbstractHydroGrid)

    @. model.S_inf = model.K^(-1 / model.alpha) * model.abs_grad_phi0^((1 - model.beta) / model.alpha) * model.Q^(1 / model.alpha)
    fill_halo!(model.S_inf, grid)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Update the far-field effective pressure N_inf based on conduit geometry and
basal velocity, constrained by ice overburden pressure limits.
"""
function update_N_inf!(model::KazmierczakHydroModel, grid::AbstractHydroGrid)

    field_data(model.N_inf) .= min.(max.(
        ((field_data(model.H) ./ field_data(model.S_inf)).^2 .* ((model.rho_i .* model.L_w .* field_data(model.abs_v_b) .* model.h_b .+ field_data(model.Q) .* field_data(model.abs_grad_phi0)) # numerator
        ./ (2.0 .* model.n^(-model.n) .* model.rho_i .* model.L_w .* field_data(model.A_visc)))).^(1.0 / model.n), # denominator
        model.sigmat .* field_data(model.Po)), field_data(model.Po)) # min and max values of N_inf

    field_data(model.N_inf)[field_data(model.S_inf) .== 0.0] .= field_data(model.Po)[field_data(model.S_inf) .== 0.0]
    fill_halo!(model.N_inf, grid)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Update the volumetric water flux per conduit Q by scaling the distributed flux q
by the characteristic channel spacing l_c.
"""
function update_Q!(model::KazmierczakHydroModel, grid::AbstractHydroGrid)

    @. model.Q = model.q * model.l_c
    fill_halo!(model.Q, grid)

    return nothing

end


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

    field_data(state.N) .= max(field_data(model.Po) - field_data(model.p_w), model.sigmat * field_data(model.Po))
    fill_halo!(state.N, grid)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Update the water pressure p_w.
"""
function update_p_w!(model::HABHydroModel, grid::AbstractHydroGrid, state::HydroState)

    field_data(model.p_w) .= -model.P_w * model.rho_sw * model.g * field_data(state.b)
    field_data(model.p_w)[field_data(state.mask) .== 0.0] .= model.P_w * model.rho_i * model.g * field_data(state.h)[field_data(state.mask) .== 0.0]
    field_data(model.p_w)[field_data(state.b) .>= 0.0] .= 0.0
    fill_halo!(model.p_w, grid)

    return nothing

end