###########################################
# Model: Aguamenti v1.0 (Gregov et al.)   #
###########################################

# Till reservoir (Sect. 2.5): a local reservoir described by its effective stress N_t, a storage law
# e(N_t) (Eq. 19) and an exchange rate with the interface (Eq. 20). Scalar functions, used per cell by
# the time step (run.jl).


"""
$(TYPEDSIGNATURES)

Void ratio of the till at effective stress `N_t`, the consolidation law of Eq. (19a):
`e = e_r - C_t log10(N_t / N_r)`.
"""
till_void_ratio(p::AguamentiParams, N_t) = p.e_r - p.C_t * log10(N_t / p.N_r)


"""
$(TYPEDSIGNATURES)

Inverse of [`till_void_ratio`](@ref): the effective stress at void ratio `e`, `N_r 10^((e_r - e)/C_t)`.
"""
till_stress(p::AguamentiParams, e) = p.N_r * exp10((p.e_r - e) / p.C_t)


"""
$(TYPEDSIGNATURES)

Bounds `(N_min, N_max)` of the till effective stress, Eqs. (19b)-(19c): the loosest state `e_max` and
the densest `e_min`, the latter also bounded by the overburden `p_i`. `N_max >= N_min` always (an
overburden below `N_min`, e.g. a very thin ice cover, pins the till at its loosest state).
"""
function till_bounds(p::AguamentiParams, p_i)
    N_min = till_stress(p, p.e_max)
    N_max = max(N_min, min(p_i, till_stress(p, p.e_min)))
    return N_min, N_max
end


"""
$(TYPEDSIGNATURES)

Consolidation time of the till, Eq. (20c): `T_t = C_t (1 - theta) h_t^2 eta_w / (2 k_t N_t ln 10)`,
with `theta = e/(1+e)` the volumetric water content at `N_t`.
"""
function till_consolidation_time(p::AguamentiParams, N_t, h_t, k_t)
    e = till_void_ratio(p, N_t)
    theta = e / (1 + e)
    return p.C_t * (1 - theta) * h_t * h_t * p.eta_w / (2 * k_t * N_t * log(oftype(N_t, 10)))
end


"""
$(TYPEDSIGNATURES)

Exchange rate of Eq. (20a), positive into the till [m/s]: `2 k_t (N_t - N) / (eta_w h_t)`. Before the
bounds of Eq. (9); used as a diagnostic and in tests (the time step derives the exchange from the
change of stored water, see [`till_relax`](@ref)).
"""
till_exchange_rate(p::AguamentiParams, N_t, N, h_t, k_t) = 2 * k_t * (N_t - N) / (p.eta_w * h_t)


"""
$(TYPEDSIGNATURES)

Advance the till effective stress over `dt` towards the interfacial `N` (Eq. 20b) and return
`(N_t_new, demand, N_t_undrained, storage)`:

1. Undrained response: `N_t_undrained = N_t + (1 - B) rho_i g dH_dt dt` (no water exchanged).
2. Drained relaxation: `N_t_new = N + (N_t_undrained - N) exp(-dt / T_t)`, with `T_t` evaluated at
   `N_t_undrained` -- exact for a constant `T_t`, so it is stable for any `dt`, also when `T_t` is far
   shorter than the step (permeable tills), where the till simply reaches `N`.
3. Clipped to the bounds of Eq. (19).

The exchange the relaxation asks for, `demand` [m/s, positive into the till], is the change of stored
water `(1 - theta) h_t de/dt` (Appendix B3) over the step, `storage * (e(N_t_new) - e(N_t_undrained)) / dt`
with `storage = (1 - theta) h_t` at `N_t_undrained`, so it is mass-consistent with the storage law. The
caller limits it by the water available at the interface (Eq. 9a) and then calls
[`till_from_exchange`](@ref).
"""
function till_relax(p::AguamentiParams, N_t, N, dt, h_t, k_t, p_i, dH_dt)
    c = till_relax_coefficients(p, N_t, dt, h_t, k_t, p_i, dH_dt)
    N_new, demand = till_relax_to(p, c, N)
    return N_new, demand, c.N_u, c.storage
end

# The parts of `till_relax` that do not depend on the interfacial N: bounds, undrained stress, its void
# ratio and storage, and the relaxation factor exp(-dt/T_t) over the step.
@inline function till_relax_coefficients(p::AguamentiParams, N_t, dt, h_t, k_t, p_i, dH_dt)
    N_min, N_max = till_bounds(p, p_i)
    N_u = clamp(N_t + (1 - p.B) * p.rho_i * p.g * dH_dt * dt, N_min, N_max)
    e_u = till_void_ratio(p, N_u)
    storage = (1 - e_u / (1 + e_u)) * h_t
    decay = exp(-dt / till_consolidation_time(p, N_u, h_t, k_t))
    return (; N_min, N_max, N_u, e_u, storage, decay, dt)
end

# `(N_new, demand)` of `till_relax` towards the interfacial `N`, from its coefficients.
@inline function till_relax_to(p::AguamentiParams, c, N)
    N_target = clamp(N, c.N_min, c.N_max)
    N_new = clamp(N_target + (c.N_u - N_target) * c.decay, c.N_min, c.N_max)
    return N_new, c.storage * (till_void_ratio(p, N_new) - c.e_u) / c.dt
end


"""
$(TYPEDSIGNATURES)

The till effective stress after exchanging `exchange` [m/s, positive into the till] over `dt`, starting
from `N_u` with storage `(1 - theta) h_t = storage` (see [`till_relax`](@ref)): the stored water changes
by `exchange dt`, and `N_t` follows from the storage law, clipped to its bounds (Eq. 9b).
"""
function till_from_exchange(p::AguamentiParams, N_u, exchange, dt, storage, p_i)
    N_min, N_max = till_bounds(p, p_i)
    e = till_void_ratio(p, N_u) + exchange * dt / storage
    return clamp(till_stress(p, e), N_min, N_max)
end
