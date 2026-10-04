#################################
# Model: Kazmierczak et al 2024 #
#################################


"""
$(TYPEDSIGNATURES)

Update the effective pressure N using a complementary error function transition between geometric
potential and far-field effective pressure.

N is only defined under grounded ice (`state.mask == 1`), where the hydrology lives: under floating
ice N = 0, and over ocean or ice-free land it is undefined. So the whole chain -- Q, S_inf, H_hard,
H_soft, H, Po, N_inf, N -- is computed only on grounded cells, in one fused pass
(`update_N_grounded_kernel!`), and every other cell gets N = 0 and zero conduit fields (Po is still
rho_i*g*h everywhere). On grounded cells each quantity is the same expression, evaluated in the same
order, as the standalone `update_Q!`/`update_S_inf!`/`update_H!`/`update_Po!`/`update_N_inf!`
broadcasts below (kept for direct use and tests), so the grounded values are bit-identical to calling
those in sequence. Fusing them also avoids 8 separate full-grid passes, which mattered because this
runs every sweep of the `(q, N)` coupling loop.
"""
function update_N!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)

    # Scalars hoisted out of the per-cell loop, exactly as the standalone broadcasts compute them.
    Q_c = effective_Q_c(model.drainage_mode, model.Q_c)
    K_fac = model.K^(-1 / model.alpha)
    grad_exp = (1 - model.beta) / model.alpha
    Q_exp = 1 / model.alpha
    denom_const = 2.0 * model.n^(-model.n) * model.rho_i * model.L_w
    inv_n = 1.0 / model.n
    sliding_coeff, melt_coeff = opening_coefficients(model.drainage_mode, typeof(denom_const))
    sqrt_pi = sqrt(pi)

    update_N_grounded_kernel!(state.N, model.Q, model.S_inf, model.H_hard, model.H_soft, model.H, model.Po, model.N_inf,
        model.q, model.abs_grad_phi0, model.kappa, model.abs_v_b, model.A_visc, model.phi0, state.h, state.mask,
        grid.Nx, grid.Ny, model.l_c, K_fac, grad_exp, Q_exp, model.H_0, model.F_till, Q_c, model.rho_i, model.g,
        model.L_w, model.h_b, sliding_coeff, melt_coeff, denom_const, inv_n, model.sigmat, sqrt_pi)

    for f in (state.N, model.Q, model.S_inf, model.H, model.Po, model.N_inf)
        fill_halo!(f, grid)
    end

    return nothing

end

"""
$(TYPEDSIGNATURES)

`true` unless `drainage_mode` is [`EfficientOnly`](@ref), the one mode whose `N_inf` has no sliding-over-obstacles
term (see `opening_coefficients`) and so does not depend on `|u_b|`. See [`N_responds_to_ub`](@ref).
"""
N_responds_to_ub(model::KazmierczakHydroModel) = !(model.drainage_mode isa EfficientOnly)


"""
$(TYPEDSIGNATURES)

`state.N` for a new basal sliding speed `abs_v_b` [m/s], with the routing held: the distributed flux `q`, the
potential gradient `abs_grad_phi0` and the geometric potential `phi0` stay as the last full update
(`update_steady_state!`) left them. This is `update_N!` alone, after setting `model.abs_v_b`.

A host calls it inside its velocity iteration, so that `N` and `u_b` are solved together instead of lagged by a
step (a lag makes them alternate between two states every step: `N` rises with `u_b` through the cavity opening in
`N_inf`, and `u_b` falls steeply with `N` through the friction law). See [`N_responds_to_ub`](@ref).

Holding the routing is exact when `q` does not depend on `u_b`. With a friction law the frictional heat is in the
water source and the routing lags the velocity iteration by one full update; with [`NoFrictionSlidingLaw`](@ref) the
source at cold bases is `-Q_b/L` (`q_T` already contains the frictional heat) and the cycle stays, so coupled runs
use a friction law (`PrescribedFieldSlidingLaw` with the host's `tau_b`).

What `update_N!` reads, and so what must be current when this is called: `abs_v_b` (the argument); from the last
full update, `q`, `abs_grad_phi0`, `phi0` and `kappa`; and, describing the geometry the host's velocity solve
uses, `state.h` (overburden `Po`), `state.mask` and `model.A_visc` (basal rate factor). The host refreshes those in
place if they have changed since the last full update.
"""
function N_from_ub!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, abs_v_b)
    model.abs_v_b .= abs_v_b
    fill_halo!(model.abs_v_b, grid)
    update_N!(model, grid, state)
    return state.N
end

# Every cell is independent, so columns are split across Julia threads when there are several; the
# per-cell arithmetic is the same either way, so results are bit-identical for any thread count, and
# with one thread (`julia` without `-t`) it is a plain loop with no threading overhead at all.
# `Threads.@threads` uses the default (dynamic) scheduler, so it composes if the caller is itself
# running inside threaded code.
function update_N_grounded_kernel!(args...)
    Ny = args[18]
    if Threads.nthreads() == 1
        for j in 1:Ny
            update_N_grounded_column!(j, args...)
        end
    else
        Threads.@threads for j in 1:Ny
            update_N_grounded_column!(j, args...)
        end
    end
    return nothing
end

function update_N_grounded_column!(j, N, Q, S_inf, H_hard, H_soft, H, Po, N_inf,
                                   q, abs_grad_phi0, kappa, abs_v_b, A_visc, phi0, h, mask,
                                   Nx, Ny, l_c, K_fac, grad_exp, Q_exp, H_0, F_till, Q_c, rho_i, g,
                                   L_w, h_b, sliding_coeff, melt_coeff, denom_const, inv_n, sigmat, sqrt_pi)
    z = zero(eltype(N))
    @inbounds for i in 1:Nx

        Po_ij = rho_i * g * h[i, j]
        Po[i, j] = Po_ij

        if mask[i, j] != 1.0
            Q[i, j] = z; S_inf[i, j] = z; H_hard[i, j] = z; H_soft[i, j] = z; H[i, j] = z
            N_inf[i, j] = z; N[i, j] = z
            continue
        end

        # update_Q!
        Q_ij = q[i, j] * l_c

        # update_S_inf! -- zero flux means zero conduit cross-section (resolves the 0^neg * 0^pos NaN)
        S_ij = Q_ij == 0.0 ? z : K_fac * abs_grad_phi0[i, j]^grad_exp * Q_ij^Q_exp

        # update_H! -- the Q_c == 0 && Q == 0 case is the 0/0 limit resolved there
        Hh = sqrt(S_ij)
        Hs = max(0.0, H_0 + (sqrt(S_ij) / F_till - H_0) * exp(-Q_ij / Q_c))
        if Q_c == 0 && Q_ij == 0.0
            Hs = z
        end
        k = kappa[i, j]
        H_ij = (1 - k) * Hh + k * Hs

        # update_N_inf!
        Ninf_ij = if S_ij == 0.0
            Po_ij
        else
            min(max(
                ((H_ij * H_ij) / (S_ij * S_ij) * (sliding_coeff * rho_i * L_w * abs_v_b[i, j] * h_b + melt_coeff * Q_ij * abs_grad_phi0[i, j])
                / (denom_const * A_visc[i, j]))^inv_n,
                sigmat * Po_ij), Po_ij)
        end

        # N, with N_inf == 0 resolved to N = 0 (see the standalone version's comment)
        N_ij = Ninf_ij == 0.0 ? z : max(0.0, erf(sqrt_pi * phi0[i, j] / (2 * Ninf_ij)) * Ninf_ij)

        Q[i, j] = Q_ij; S_inf[i, j] = S_ij; H_hard[i, j] = Hh; H_soft[i, j] = Hs; H[i, j] = H_ij
        N_inf[i, j] = Ninf_ij; N[i, j] = N_ij
    end
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

    # Scalar factor and exponents hoisted out of the broadcast: written inline, K^(-1/alpha) was a
    # third pow() evaluated per cell (this function was ~60% of update_N!'s cost).
    K_fac = model.K^(-1 / model.alpha)
    grad_exp = (1 - model.beta) / model.alpha
    Q_exp = 1 / model.alpha
    @. model.S_inf = K_fac * model.abs_grad_phi0^grad_exp * model.Q^Q_exp

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
    inv_n = 1.0 / model.n

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
        / (denom_const * model.A_visc))^inv_n, # denominator
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
