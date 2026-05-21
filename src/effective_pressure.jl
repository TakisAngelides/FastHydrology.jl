#################################
# Model: Kazmierczak et al 2024 #
#################################


"""
$(TYPEDSIGNATURES)

Update the effective pressure N across the grid using a complementary error function
transition between geometric potential and far-field effective pressure.
"""
function update_N!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)
    update_Q!(model, grid)           # volumetric water flux [m3 s-1] per conduit
    update_S_inf!(model, grid)       # cross-sectional area of conduits
    update_H!(model, grid)           # thickness of conduits
    update_Po!(model, grid, state)   # ice overburden pressure rho*g*h
    update_N_inf!(model, grid)       # far-field effective pressure
    @. state.N.data = max(0.0, erf((sqrt(pi) * model.phi0.data) / (2 * model.N_inf.data)) * model.N_inf.data)
    fill_halo!(state.N, grid)
    return nothing
end


"""
$(TYPEDSIGNATURES)

Update the hydrostatic ice overburden pressure Po based on ice thickness h.
Defined for all models that carry a Po field and share the same rho_i, g constants.
"""
function update_Po!(model::AbstractHydroModel, grid::AbstractHydroGrid, state::HydroState)
    @. model.Po.data = max(model.rho_i * model.g * state.h.data, 1e5)
    fill_halo!(model.Po, grid)
    return nothing
end


"""
$(TYPEDSIGNATURES)

Update the conduit thickness H by calculating separate values for hard and soft beds,
then interpolating based on the bed heterogeneity indicator kappa.
"""
function update_H!(model::KazmierczakHydroModel, grid::AbstractHydroGrid)
    @. model.H_hard      = sqrt(model.S_inf)
    @. model.H_soft.data = max(0.0, model.H_0 + (sqrt(model.S_inf.data) / model.F_till - model.H_0) * exp(-model.Q.data / model.Q_c))
    @. model.H           = (1 - model.kappa) * model.H_hard + model.kappa * model.H_soft
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
    @. model.N_inf.data = min(max(
        ((model.H.data / model.S_inf.data)^2 * (
            (model.rho_i * model.L_w * model.abs_v_b.data * model.h_b + model.Q.data * model.abs_grad_phi0.data) /
            (2.0 * model.n^(-model.n) * model.rho_i * model.L_w * model.A_visc.data)
        ))^(1.0 / model.n),
        model.sigmat * model.Po.data),
        model.Po.data)
    @. model.N_inf.data[model.S_inf.data .== 0.0] .= model.Po[model.S_inf.data .== 0.0]
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
# Model: Height above buoyancy (HAB) #
####################################


"""
$(TYPEDSIGNATURES)

Update the effective pressure N across the grid using a complementary error function
transition between geometric potential and far-field effective pressure.
"""
function update_N!(model::HABHydroModel, grid::AbstractHydroGrid, state::HydroState)
    update_Po!(model, grid, state)
    update_p_w!(model, grid, state)
    @. state.N.data = max(model.Po.data - model.p_w.data, model.sigmat * model.Po.data)
    fill_halo!(state.N, grid)
    return nothing
end


"""
$(TYPEDSIGNATURES)

Update the water pressure p_w.
"""
function update_p_w!(model::HABHydroModel, grid::AbstractHydroGrid, state::HydroState)
    @. model.p_w.data                           = -model.P_w * model.rho_sw * model.g * state.b.data
    @. model.p_w.data[state.mask.data .== 0.0]  =  model.P_w * model.rho_i  * model.g * state.h.data[state.mask.data .== 0.0]
    @. model.p_w.data[state.b.data .>= 0.0]     = 0.0
    fill_halo!(model.p_w, grid)
    return nothing
end
