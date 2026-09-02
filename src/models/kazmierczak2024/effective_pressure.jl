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

    # sqrt(pi) must be precomputed outside the broadcast; see update_p_w! for why.
    sqrt_pi = sqrt(pi)
    @. state.N = max(0.0, erf(sqrt_pi * model.phi0 / (2 * model.N_inf)) * model.N_inf)

    # N_inf == 0 makes the erf argument diverge (Inf, or 0/0 -> NaN when phi0 == 0 too --
    # e.g. flat, ungrounded cells where h == b == 0, so Po == 0). This used to be masked by
    # update_Po!'s old 1e5 floor, which kept N_inf away from exactly 0 in practice; it's
    # reachable on its own now that sigmat = 0.0 is the default (N_inf's lower clamp bound
    # sigmat*Po is 0 too). Physically, zero far-field effective pressure should give zero N,
    # so resolve the indeterminate form in favor of that limit rather than propagating NaN,
    # the same overwrite_where! pattern update_S_inf!/update_N_inf! already use for their
    # own 0/0 cells.
    overwrite_where!(grid, state.N, model.N_inf, ==(0.0), 0.0)
    fill_halo!(state.N, grid)

    return nothing

end

"""
$(TYPEDSIGNATURES)

The `Q_c` value `update_H!` uses in Eq. (9)'s exp(-Q/Q_c) soft-bed geometry blend, selected by
`model.drainage_mode` -- see `AbstractDrainageMode` for the physical justification of each case.
"""
effective_Q_c(::BothDrainage, Q_c) = Q_c
effective_Q_c(::EfficientOnly, Q_c::T) where {T} = zero(T)
effective_Q_c(::InefficientOnly, Q_c::T) where {T} = T(Inf)

"""
$(TYPEDSIGNATURES)

Update the conduit thickness H by calculating separate values for hard and soft beds,
then interpolating based on the bed heterogeneity indicator kappa.
"""
function update_H!(model::KazmierczakHydroModel, grid::AbstractHydroGrid)

    Q_c = effective_Q_c(model.drainage_mode, model.Q_c)

    @. model.H_hard = sqrt(model.S_inf)
    @. model.H_soft = max(0.0, model.H_0 + (sqrt(model.S_inf) / model.F_till - model.H_0) * exp(-model.Q / Q_c))

    # EfficientOnly drives Q_c to exactly 0 above so exp(-Q/Q_c) -> 0 for any Q > 0. At the same
    # degenerate Q == 0 cells update_S_inf! already special-cases (see its comment there), this
    # divides 0/0 = NaN instead of taking the correct Q_c -> 0 limit for Q == 0 (which is 1, not 0 --
    # exp(-Q/Q_c) with Q == 0 is exp(0) = 1 for every Q_c > 0, however small). That limit gives
    # H_soft = sqrt(S_inf)/F_till, which is 0 anyway since S_inf == 0 at those cells, so resolve it
    # the same way update_S_inf! resolves its own 0/0 case.
    if Q_c == 0
        overwrite_where!(grid, model.H_soft, model.Q, ==(0.0), 0.0)
    end

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

    # At degenerate cells with Q == 0 and abs_grad_phi0 == 0 simultaneously (which occurs at a
    # few corner/edge cells where the input data is flat outside the glacier extent), the formula
    # above evaluates 0^(negative) * 0^(positive) = Inf * 0 = NaN, since (1-beta)/alpha < 0.
    # Physically, zero flux implies zero conduit cross-section regardless of the gradient, so we
    # resolve this indeterminate form in favor of that limit.
    overwrite_where!(grid, model.S_inf, model.Q, ==(0.0), 0.0)

    fill_halo!(model.S_inf, grid)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Scalar 0/1 coefficients for the sliding-over-obstacles (inefficient) and melt-driven (efficient)
opening terms of Eq. (5b)/(6a), selected by `model.drainage_mode` -- see `AbstractDrainageMode`.
Resolved once outside the `@.` broadcast in `update_N_inf!`, the same "precompute the scalar
sub-expression" pattern as `denom_const` there, so `EfficientOnly`/`InefficientOnly` cost nothing
beyond multiplying the dropped term by 0.0.
"""
opening_coefficients(::BothDrainage, ::Type{T}) where {T} = (one(T), one(T))
opening_coefficients(::EfficientOnly, ::Type{T}) where {T} = (zero(T), one(T))
opening_coefficients(::InefficientOnly, ::Type{T}) where {T} = (one(T), zero(T))

"""
$(TYPEDSIGNATURES)

Update the far-field effective pressure N_inf based on conduit geometry and
basal velocity, constrained by ice overburden pressure limits.
"""
function update_N_inf!(model::KazmierczakHydroModel, grid::AbstractHydroGrid)

    # As in update_p_w!, the purely-scalar sub-expression `model.n^(-model.n)` must be
    # precomputed outside the broadcast to avoid breaking Oceananigans' AbstractOperation
    # conversion.
    denom_const = 2.0 * model.n^(-model.n) * model.rho_i * model.L_w

    # sliding_coeff/melt_coeff zero out the sliding-over-obstacles or melt-driven opening term for
    # EfficientOnly/InefficientOnly (see AbstractDrainageMode); both are 1.0 for the default
    # BothDrainage, reproducing the original unconditional sum.
    sliding_coeff, melt_coeff = opening_coefficients(model.drainage_mode, typeof(denom_const))

    # (H*H)/(S_inf*S_inf) rather than (H/S_inf)^2.0: Float64^Float64 dispatches to libm's pow() per
    # element, ~17x slower (benchmarked) than a plain multiply for no numerical difference -- and
    # this runs inside the (q, N) coupling Picard loop (up to max_coupling_iters times per solve)
    # for N-dependent sliding laws, so it's the hottest of the four spots this pattern showed up in.
    @. model.N_inf = min(max(
        ((model.H * model.H) / (model.S_inf * model.S_inf) * (sliding_coeff * model.rho_i * model.L_w * model.abs_v_b * model.h_b + melt_coeff * model.Q * model.abs_grad_phi0) # numerator
        / (denom_const * model.A_visc))^(1.0 / model.n), # denominator
        model.sigmat * model.Po), model.Po) # min and max values of N_inf

    overwrite_where!(grid, model.N_inf, model.S_inf, ==(0.0), model.Po)
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
