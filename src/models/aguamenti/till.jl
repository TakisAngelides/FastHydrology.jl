###########################################
# Model: Aguamenti v1.0 (Gregov et al.)   #
###########################################

# Till reservoir (Sect. 3.5): a local reservoir described by its effective stress N_t, a storage law
# e(N_t) (Eq. 18) and an exchange rate with the interface (Eq. 19). Scalar functions, used per cell by
# the time step (solve.jl).


"""
$(TYPEDSIGNATURES)

Void ratio of the till at effective stress `N_t`, the consolidation law of Eq. (18a):
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

Bounds `(N_min, N_max)` of the till effective stress, Eqs. (18b)-(18c): the loosest state `e_max` and
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

Consolidation time of the till, Eq. (19c): `T_t = C_t (1 - theta) h_t^2 eta_w / (2 k_t N_t ln 10)`,
with `theta = e/(1+e)` the volumetric water content at `N_t`.
"""
function till_consolidation_time(p::AguamentiParams, N_t, h_t, k_t)
    e = till_void_ratio(p, N_t)
    theta = e / (1 + e)
    return p.C_t * (1 - theta) * h_t * h_t * p.eta_w / (2 * k_t * N_t * log(oftype(N_t, 10)))
end


"""
$(TYPEDSIGNATURES)

Exchange rate of Eq. (19a), positive into the till [m/s]: `2 k_t (N_t - N) / (eta_w h_t)`. Before the
bounds of Eq. (9); used as a diagnostic and in tests (the time step derives the exchange from the
change of stored water, see [`till_relax`](@ref)).
"""
till_exchange_rate(p::AguamentiParams, N_t, N, h_t, k_t) = 2 * k_t * (N_t - N) / (p.eta_w * h_t)


"""
$(TYPEDSIGNATURES)

Advance the till effective stress over `dt` towards the interfacial `N`, Eq. (28a), and return
`(N_t_new, demand, dN_u, storage)`:

1. Drained relaxation: `N_rel = N + (N_t - N) exp(-dt / T_t)`, with `T_t` evaluated at the stress
   `N_t` the step starts from (Eq. 19c) -- exact for a constant `T_t`, so it is stable for any `dt`,
   also when `T_t` is far shorter than the step (permeable tills), where the till simply reaches `N`.
2. Undrained response, added after the relaxation: `dN_u = (1 - B) rho_i g dH_dt dt` (no water
   exchanged), `N_t_new = N_rel + dN_u`.
3. Clipped to the bounds of Eq. (18).

The exchange the relaxation asks for, `demand` [m/s, positive into the till], is the change of stored
water `(1 - theta) h_t de/dt` (Appendix D3) of the relaxation part alone,
`storage * (e(N_rel) - e(N_t)) / dt` with `storage = (1 - theta) h_t` at `N_t`, so it is
mass-consistent with the storage law. The caller limits it by the water available at the interface
(Eqs. 9a, 27) and then calls [`till_from_exchange`](@ref) with `dN_u`.
"""
function till_relax(p::AguamentiParams, N_t, N, dt, h_t, k_t, p_i, dH_dt)
    c = till_relax_coefficients(p, N_t, dt, h_t, k_t, p_i, dH_dt)
    N_new, demand = till_relax_to(p, c, N)
    return N_new, demand, c.dN_u, c.storage
end

# The parts of `till_relax` that do not depend on the interfacial N: bounds, the stress the step starts
# from (in its bounds: the overburden, and so N_max, may have changed), its void ratio and storage, the
# consolidation time and relaxation factor over the step, and the undrained increment.
@inline function till_relax_coefficients(p::AguamentiParams, N_t, dt, h_t, k_t, p_i, dH_dt)
    N_min, N_max = till_bounds(p, p_i)
    N_0 = clamp(N_t, N_min, N_max)
    e_0 = till_void_ratio(p, N_0)
    storage = (1 - e_0 / (1 + e_0)) * h_t
    T_t = till_consolidation_time(p, N_0, h_t, k_t)
    decay = exp(-dt / T_t)
    dN_u = (1 - p.B) * p.rho_i * p.g * dH_dt * dt
    return (; N_min, N_max, N_0, e_0, storage, T_t, decay, dN_u, dt)
end

"""
$(TYPEDSIGNATURES)

One step of Eq. (28a) without bounds: `N + (N_t - N) decay + dN_u`, with `decay = exp(-dt/T_t)` and
`dN_u = (1 - B) rho_i g dH` the undrained increment, added after the relaxation. The exact solution of
`dN_t/dt = -(N_t - N)/T_t` for `N` and `T_t` held over the step, so stable for any `dt`. The building
block of [`till_relax`](@ref); exposed for tests of the reservoir alone (v2 Table B1, Appendix A4).
"""
@inline till_exponential_step(N_t, N, decay, dN_u = zero(N_t)) = N + (N_t - N) * decay + dN_u

# `(N_new, demand)` of `till_relax` towards the interfacial `N`, from its coefficients.
@inline function till_relax_to(p::AguamentiParams, c, N)
    N_target = clamp(N, c.N_min, c.N_max)
    N_rel = till_exponential_step(c.N_0, N_target, c.decay)
    demand = c.storage * (till_void_ratio(p, N_rel) - c.e_0) / c.dt
    return clamp(N_rel + c.dN_u, c.N_min, c.N_max), demand
end

# The exchange of Eq. (28b) towards the interfacial `N` (that of the previous step for
# `LaggedTillExchange`), from the coefficients of `till_relax`: the storage change of the Eq. (28a) step
# towards `N` clipped to the till bounds, `storage * (e(N_rel) - e(N_t)) / dt` (the `demand` of
# `till_relax_to`). For a short step and a small `N_t - N` it is the rate of Eq. (19a) averaged over the
# step, `2 k_t (N_t - N)/(eta_w h_t) T_t/dt (1 - exp(-dt/T_t))`, the literal Eq. (28b). Unlike that
# linear rate, which through the logarithmic storage law drains more than reaching `N` needs (the till
# overshot `N^(j-1)`, up to `N_max`, for `dt` of order `T_t`), it is consistent with the storage law:
# an exchange that is not limited by the water available takes the till exactly to the Eq. (28a) value,
# between `N_t^(j)` and the clipped `N^(j-1)`.
@inline till_lagged_exchange(p::AguamentiParams, c, N) = till_relax_to(p, c, N)[2]


"""
$(TYPEDSIGNATURES)

The till effective stress after exchanging `exchange` [m/s, positive into the till] over `dt`, starting
from `N_t` with storage `(1 - theta) h_t = storage` (see [`till_relax`](@ref)): the stored water changes
by `exchange dt` (`(theta h_t)^(j+1) = (theta h_t)^(j) + exchange dt`, Sect. 4), the stress follows
from the storage law, the undrained increment `dN_u` is added (Eq. 28a), and the result is clipped to
the bounds of Eq. (18).
"""
function till_from_exchange(p::AguamentiParams, N_t, exchange, dt, storage, p_i, dN_u = zero(N_t))
    N_min, N_max = till_bounds(p, p_i)
    e = till_void_ratio(p, clamp(N_t, N_min, N_max)) + exchange * dt / storage
    return clamp(till_stress(p, e) + dN_u, N_min, N_max)
end
