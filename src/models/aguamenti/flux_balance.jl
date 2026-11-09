###########################################
# Model: Aguamenti v1.0 (Gregov et al.)   #
###########################################

# Flux balance (Sect. 2.6): in every wet cell the effective pressure is the root N* of
# q_tot(N) = q (Eq. 21) on the active branch, bounded by the regimes of Eq. (23) and corrected near the
# grounding line (Eq. 24). Scalar per-cell functions; the field-level driver is in solve.jl.


# Regime of a grounded cell, reported in `model.regime` (a float field, like every other field).
const AGUAMENTI_NOT_GROUNDED = -1   # outside the grounded mask
const AGUAMENTI_COLD         = 0    # cold base: no routed water, N = dry value (Eq. 26f)
const AGUAMENTI_DRY          = 1    # temperate, the till absorbs all the available water: N = dry value
const AGUAMENTI_UPLIFT       = 2    # no root, or min(p_i, N*, N_0) below the floor: N = eps_u p_i (Eq. 23a)
const AGUAMENTI_PRESSURIZED  = 3    # 0 < N* < p_i
const AGUAMENTI_FREE_SURFACE = 4    # N* >= p_i: partly filled cavities, N = p_i
const AGUAMENTI_STRENGTH     = 5    # connected till, N* > N_0: N = N_0 = F_b c_0


# Bisection for the root of a monotone f on [lo, hi] in log space; `f(lo)` and `f(hi)` have opposite signs.
@inline function log_bisect(f, lo, hi, iters)
    a, b = log(lo), log(hi)
    fa = f(lo)
    for _ in 1:iters
        m = (a + b) / 2
        fm = f(exp(m))
        if (fm > 0) == (fa > 0)
            a, fa = m, fm
        else
            b = m
        end
    end
    return exp((a + b) / 2)
end


# Golden-section minimum of a unimodal f on [lo, hi] in log space; returns (x_min, f(x_min)).
@inline function log_golden_min(f, lo, hi, iters)
    invphi = (sqrt(oftype(lo, 5)) - 1) / 2
    a, b = log(lo), log(hi)
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = f(exp(c)), f(exp(d))
    for _ in 1:iters
        if fc < fd
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(exp(c))
        else
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(exp(d))
        end
    end
    x = exp((a + b) / 2)
    return x, f(x)
end


"""
$(TYPEDSIGNATURES)

Saddle-node flux `q_sn` (Sect. 2.6, Appendix B4): the minimum over `N` of the total flux with a fully
developed channel (`chi = 1`), and the pressure `N_m` where it is reached. Below `q_sn` no steady
channel exists. Returns `(q_sn, N_m)`; on a bed without channels (`kappa = 1`) there is no channel
branch and `q_sn = 0`.
"""
function saddle_node_flux(p::AguamentiParams, kappa, Psi, u_b, A, D_s)
    kappa >= 1 && return zero(Psi), zero(Psi)
    f = N -> total_flux(p, N, one(Psi), kappa, Psi, u_b, A, D_s)
    N_m, q_m = log_golden_min(f, p.N_lo, p.N_hi, p.root_iters)
    return q_m, N_m
end


"""
$(TYPEDSIGNATURES)

Root `N*` of the flux balance `q_tot(N) = q` (Eq. 21) on the active branch, for one cell, together
with whether the channelized branch was used. Returns `N* = 0` when the active branch has no root
(the uplift regime), and `N* = p_i` when the root lies at or above the overburden (only `min(p_i, N*)`
enters Eq. 23a).

The active branch (Sect. 2.6, Fig. 8) is the channelized one when a channel is present (`chi > 0`) and
can be sustained (`q >= q_sn`): the larger root of `q_tot(N; chi) = q`, on which `q_tot` increases.
Otherwise (no channel, or a decaying one) it is the sheet branch: the root of the distributed and canal
terms alone (`q_tot(N; chi = 0)`), which is strictly decreasing (Appendix B4).

`developing` marks a channel that is growing (`q >= q_on`) and has not yet reached `chi_dev`. While
channels develop from the uplift regime the cell stays at the uplift floor (`N* = 0`, Sect. 2.6 and
Fig. 8); developing from a pressurized sheet (onset at `q_Lambda < q_h,sat`), the root uses the current
`chi` but is capped at the root of the developed channel (`chi = 1`), so a young channel never raises
`N` above the value of a developed one.
"""
function flux_balance_root(p::AguamentiParams, q, chi, kappa, Psi, u_b, A, D_s, p_i, q_sn; developing::Bool = false)
    z = zero(q)
    p_i > p.N_lo || return z, false
    iters = p.root_iters
    if chi > 0 && kappa < 1 && q >= q_sn
        if developing && chi < p.chi_dev
            # A channel still developing from the uplift regime keeps the cell at the uplift floor (Sect. 2.6)
            total_flux(p, p.N_lo, z, kappa, Psi, u_b, A, D_s) <= q && return z, true
            # From a pressurized sheet: the root with the current chi, but never above the developed channel's
            N_c = channelized_root(p, q, chi, kappa, Psi, u_b, A, D_s, p_i)
            N_1 = channelized_root(p, q, one(chi), kappa, Psi, u_b, A, D_s, p_i)
            return (N_c == 0 ? N_1 : min(N_c, N_1)), true
        end
        return channelized_root(p, q, chi, kappa, Psi, u_b, A, D_s, p_i), true
    else
        f = N -> total_flux(p, N, z, kappa, Psi, u_b, A, D_s)
        f(p.N_lo) <= q && return z, false         # flux above saturation: no pressurized state (uplift)
        f(p_i) >= q && return p_i, false          # root at or above p_i (free surface)
        return log_bisect(N -> f(N) - q, p.N_lo, p_i, iters), false
    end
end


# Larger root of q_tot(N; chi) = q, on the side where q_tot increases; 0 if there is none, p_i if it is at or above p_i.
function channelized_root(p::AguamentiParams, q, chi, kappa, Psi, u_b, A, D_s, p_i)
    z = zero(q)
    f = N -> total_flux(p, N, chi, kappa, Psi, u_b, A, D_s)
    N_m, q_m = log_golden_min(f, p.N_lo, p.N_hi, p.root_iters)
    q_m > q && return z                       # no root: a channel too small for this flux (uplift)
    N_m >= p_i && return p_i                  # the increasing side starts above the overburden
    f(p_i) <= q && return p_i                 # root at or above p_i
    return log_bisect(N -> f(N) - q, N_m, p_i, p.root_iters)
end


"""
$(TYPEDSIGNATURES)

Far-field effective pressure and regime from the root `N*` of the flux balance, Eq. (23a):
`N_inf = max(eps_u p_i, min(p_i, N*, N_0^kappa))`. Returns `(N_inf, regime)`.
"""
function regime_pressure(p::AguamentiParams, N_star, p_i, N0)
    floor_ = p.eps_u * p_i
    N_inf = max(floor_, min(p_i, N_star, N0))
    regime = if min(p_i, N_star, N0) <= floor_
        AGUAMENTI_UPLIFT
    elseif N0 < min(p_i, N_star)
        AGUAMENTI_STRENGTH
    elseif N_star >= p_i
        AGUAMENTI_FREE_SURFACE
    else
        AGUAMENTI_PRESSURIZED
    end
    return N_inf, regime
end


"""
$(TYPEDSIGNATURES)

Grounding-line boundary layer, Eq. (24): `N = erf(sqrt(pi)/2 phi0 / N_inf) N_inf`, with `phi0` taken
relative to sea level (`rho_i g H + rho_sw g B`, which vanishes at flotation), as in
[`KazmierczakHydroModel`](@ref).
"""
function grounding_line_correction(p::AguamentiParams, N_inf, h, b)
    N_inf > 0 || return zero(N_inf)
    phi0_sl = p.rho_i * p.g * h + p.rho_sw * p.g * b
    return max(zero(N_inf), erf(sqrt(oftype(N_inf, pi)) * phi0_sl / (2 * N_inf)) * N_inf)
end


"""
$(TYPEDSIGNATURES)

Advance the channel indicator over `dt`, Eq. (22): `dchi/dt = a (1 - chi) - b chi` with
`a = Lambda_0 Theta(q - q_on)` and `b = c_e A N^n Theta(q_sn - q)`, integrated exactly for rates held
over the step (stable for any `dt`). A decaying indicator below `chi_min` is set to 0.
"""
function advance_channel_indicator(p::AguamentiParams, chi, q, q_on, q_sn, Lambda0, N, A, dt)
    a = q >= q_on ? Lambda0 : zero(chi)
    b = q < q_sn ? p.c_e * A * N^p.n : zero(chi)
    s = a + b
    s > 0 || return chi
    chi_eq = a / s
    chi_new = chi_eq + (chi - chi_eq) * exp(-s * dt)
    (b > 0 && chi_new < p.chi_min) && return zero(chi)
    return clamp(chi_new, zero(chi), one(chi))
end


"""
$(TYPEDSIGNATURES)

Steady state of Eq. (22) for a cell, keeping the hysteresis: a channel forms where the flux reaches
the onset flux (`chi = 1` for `q >= q_on`), disappears where it cannot be sustained (`chi = 0` for
`q < q_sn`), and the current `chi` is kept in the bistable band between.
"""
steady_channel_indicator(chi, q, q_on, q_sn) = q >= q_on ? one(chi) : q < q_sn ? zero(chi) : chi
