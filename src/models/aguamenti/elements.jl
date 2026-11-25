###########################################
# Model: Aguamenti v1.0 (Gregov et al.)   #
###########################################

# Element library (Sect. 3.4): the steady flux each drainage element carries at effective pressure N
# under the hydraulic gradient Psi, with its bound or existence condition, and the channel growth
# rate. Every function here is a scalar function of plain numbers, so it can be tested on its own and
# called per cell from the flux balance (flux_balance.jl). `p` is an `AguamentiParams`.


"""
$(TYPEDSIGNATURES)

Flow law of the distributed elements, the cavity sheet of a hard bed and the film of a soft bed
(`sheet_law` keyword of [`AguamentiHydroModel`](@ref)): the flux `q` a layer of height `h` carries under
the gradient `Psi`. Every closed form that depends on it dispatches on the law: the steady flux
(Eq. 10b), the saturation flux (Eq. 12a), the height and pressure of a given flux (Eqs. 11a-11b), and
the factors `phi_1`, `phi_2` of the channel growth rate and onset (Eqs. 14a, 15; Appendix D1), which
follow from the elasticity `d ln q / d ln h = 3 phi_1`, with `phi_2 = 3 phi_1 - 1`.

- [`LaminarTurbulentSheetLaw`](@ref)`()` (default): `q (1 + R/R_c) = k_d h^3 Psi` (Eq. 10b), with
  `k_d` and `R_c` the parameters `k_d`, `Re_c` of the model.
- [`TurbulentSheetLaw`](@ref)`(; k_s = 5e-3)`: the turbulent law of GlaDS, `q = k_s h^(5/4) Psi^(1/2)`
  (Table B2, Sects. 5.3 and 6.1).
"""
abstract type AbstractSheetLaw end

"""
$(TYPEDSIGNATURES)

The laminar-turbulent sheet law of Eq. (10b), `q (1 + rho_w q/(eta_w R_c)) = k_d h^3 Psi` (default; see
[`AbstractSheetLaw`](@ref)). Its elasticity `d ln q/d ln h = 3 (1 + R/R_c)/(1 + 2 R/R_c)` goes from 3
(laminar) to 3/2 (turbulent), so `phi_2` goes from 2 (Hewitt's critical discharge) to 1/2.
"""
struct LaminarTurbulentSheetLaw <: AbstractSheetLaw end

"""
$(TYPEDSIGNATURES)

The turbulent sheet law of GlaDS (Werder et al., 2013) as used in SHMIP, `q = k_s h^(5/4) Psi^(1/2)`,
with `k_s` [m^(7/4) kg^(-1/2)] (Table B2: `5e-3`); see [`AbstractSheetLaw`](@ref). `k_d` and `Re_c` are
not used. Its elasticity is `d ln q/d ln h = 5/4`, so `phi_1 = 5/12` and `phi_2 = 1/4` (Appendix D1):
a channelizing perturbation grows on a pressurized sheet above `q_Lambda = 4 rho_i L_w u_b h_r/(l_r Psi)`,
eight times Hewitt's laminar critical discharge, and at `Lambda_0 = (5/4) q Psi/(rho_i L_w h_w)` in
the uplift regime.
"""
struct TurbulentSheetLaw{T <: Real} <: AbstractSheetLaw
    k_s ::T
    function TurbulentSheetLaw{T}(k_s::T) where {T <: Real}
        k_s > 0 || throw(ArgumentError("k_s must be positive (got $k_s)"))
        return new{T}(k_s)
    end
end
TurbulentSheetLaw(k_s::T) where {T <: Real} = TurbulentSheetLaw{T}(k_s)
TurbulentSheetLaw(; k_s = 5e-3) = TurbulentSheetLaw(k_s)

# The law with its coefficients in the model's float type.
convert_sheet_law(::Type{T}, law::LaminarTurbulentSheetLaw) where {T} = law
convert_sheet_law(::Type{T}, law::TurbulentSheetLaw) where {T} = TurbulentSheetLaw(T(law.k_s))


"""
$(TYPEDSIGNATURES)

Physical constants and solver configuration of [`AguamentiHydroModel`](@ref), with the defaults of
Tables C1 and C2 of Gregov et al. (Aguamenti v1.0), the flow law of the distributed elements
(`sheet_law`, see [`AbstractSheetLaw`](@ref)) and whether channels are represented (`channels`).
Spatially varying bed data (`kappa`, `h_t`, `k_t`, `c_0`, `D_s`) live on the model's workspace as
fields instead.
"""
struct AguamentiParams{T <: AbstractFloat, SL <: AbstractSheetLaw}
    # Ice and water
    n       ::T   # Glen's flow-law exponent
    rho_i   ::T   # ice density [kg/m3]
    rho_w   ::T   # water density [kg/m3]
    rho_sw  ::T   # sea-water density [kg/m3], for the grounding-line correction (phi0 relative to sea level)
    L_w     ::T   # latent heat of fusion [J/kg]
    eta_w   ::T   # dynamic viscosity of water [Pa s]
    g       ::T   # gravitational acceleration [m/s2]
    # Sheet and film (distributed elements)
    h_r     ::T   # bedrock roughness height [m]
    l_r     ::T   # bedrock roughness wavelength [m]
    h_rs    ::T   # clast roughness height [m]
    l_rs    ::T   # clast roughness wavelength [m]
    c_d     ::T   # sheet creep coefficient (2/n^n)
    k_d     ::T   # laminar sheet conductivity [Pa^-1 s^-1]
    Re_c    ::T   # critical Reynolds number of the laminar-turbulent transition
    # Channel
    c_e     ::T   # conduit creep coefficient (2/n^n)
    f       ::T   # Darcy-Weisbach friction factor
    alpha   ::T   # flow-law exponent of the cross-sectional area
    beta    ::T   # flow-law exponent of the gradient
    K_h     ::T   # channel conductivity K, from f (semicircular conduit, as in K24)
    l_h     ::T   # channel spacing [m]
    # Canal
    l_s     ::T   # canal spacing [m]
    S_c     ::T   # Shields threshold
    drho_s  ::T   # grain-water density difference rho_s - rho_w [kg/m3]
    k_can   ::T   # canal conductivity sqrt(8/(rho_w f))
    # Till strength
    F_b     ::T   # bearing factor
    kappa_c ::T   # till connectivity threshold
    # Till reservoir
    e_r     ::T   # reference void ratio
    N_r     ::T   # reference effective stress [Pa]
    C_t     ::T   # compressibility coefficient
    e_max   ::T   # loosest void ratio
    e_min   ::T   # densest void ratio
    B       ::T   # Skempton coefficient
    # Flux balance
    eps_u   ::T   # uplift floor, fraction of p_i
    Psi_min ::T   # floor on the hydraulic gradient used by the elements [Pa/m] (numerical guard for flat potentials)
    N_lo    ::T   # lower end of the effective-pressure search interval [Pa]
    N_hi    ::T   # upper end of the search interval of the saddle-node minimum [Pa]
    root_iters ::Int  # cap on function evaluations of the per-cell Brent root finders (flux balance, till uptake); 2x for the saddle-node minimum
    chi_min ::T   # a decaying channel indicator below this is set to 0
    chi_dev ::T   # a growing channel indicator at or above this counts as a developed channel (Sect. 3.6)
    # Options
    sheet_law ::SL    # flow law of the sheet and the film (AbstractSheetLaw)
    channels  ::Bool  # false: no channels (chi = 0, no onset, no channelized branch; Table B2, experiment (a))
end


# Reynolds factor omega = rho_w / (eta_w Re_c), so that R/R_c = omega q.
@inline reynolds_factor(p::AguamentiParams) = p.rho_w / (p.eta_w * p.Re_c)


"""
$(TYPEDSIGNATURES)

Positive root of the sheet flow law with the Reynolds correction, `q (1 + omega q) = X` (Eq. 10b with
`X = k_d h^3 Psi`), written as `2X / (1 + sqrt(1 + 4 omega X))` so that it stays accurate in the
laminar limit (`q -> X`).
"""
@inline function sheet_flow_law(X, omega)
    return 2X / (1 + sqrt(1 + 4 * omega * X))
end


"""
$(TYPEDSIGNATURES)

Steady cavity height `h_hat(N) = h_r / (1 + l_r c_d A N^n / u_b)` (Eq. 10a), written as
`h_r u_b / (u_b + l_r c_d A N^n)` so that a bed without sliding (`u_b = 0`) has closed cavities for any
`N > 0`. `h_r`/`l_r` are the bedrock (sheet) or clast (film) roughness.
"""
@inline function cavity_height(N, u_b, A, h_r, l_r, c_d, n)
    den = u_b + l_r * c_d * A * N^n
    return den > 0 ? h_r * u_b / den : h_r
end


"""
$(TYPEDSIGNATURES)

Flux `q` [m2/s] of a distributed layer of height `h` under the gradient `Psi`, by the sheet law
`law` (see [`AbstractSheetLaw`](@ref)): the root of `q (1 + rho_w q/(eta_w R_c)) = k_d h^3 Psi`
(Eq. 10b), or `k_s h^(5/4) Psi^(1/2)` (turbulent, Table B2).
"""
@inline sheet_law_flux(p::AguamentiParams, ::LaminarTurbulentSheetLaw, h, Psi) =
    sheet_flow_law(p.k_d * h * h * h * Psi, reynolds_factor(p))
@inline sheet_law_flux(p::AguamentiParams, law::TurbulentSheetLaw, h, Psi) = law.k_s * h * sqrt(sqrt(h)) * sqrt(Psi)

"""
$(TYPEDSIGNATURES)

Steady flux of a distributed element (sheet on a hard bed, or film on a soft bed with the clast
roughness), Eqs. (10a)-(10b): the flux of the sheet law `p.sheet_law` at the cavity height `h_hat(N)`
(by default the root of `q (1 + rho_w q/(eta_w R_c)) = k_d h_hat(N)^3 Psi`). Strictly decreasing in
`N`, from the saturation flux at `N -> 0` to 0.
"""
@inline function distributed_flux(p::AguamentiParams, N, Psi, u_b, A, h_r, l_r)
    h = cavity_height(N, u_b, A, h_r, l_r, p.c_d, p.n)
    return sheet_law_flux(p, p.sheet_law, h, Psi)
end

sheet_flux(p::AguamentiParams, N, Psi, u_b, A) = distributed_flux(p, N, Psi, u_b, A, p.h_r, p.l_r)
film_flux(p::AguamentiParams, N, Psi, u_b, A)  = distributed_flux(p, N, Psi, u_b, A, p.h_rs, p.l_rs)


"""
$(TYPEDSIGNATURES)

Saturation flux of the sheet, Eq. (12a): the flux at `N -> 0`, cavities as high as the bumps.
`k_d h_r^3 Psi` in the laminar limit (Eq. 12b); `k_s h_r^(5/4) Psi^(1/2)` with
[`TurbulentSheetLaw`](@ref). Above it no pressurized sheet exists (uplift).
"""
sheet_saturation_flux(p::AguamentiParams, Psi) = saturation_flux(p, p.sheet_law, p.h_r, Psi)

"""
$(TYPEDSIGNATURES)

Saturation flux of the film (clast roughness), the soft-bed counterpart of [`sheet_saturation_flux`](@ref).
"""
film_saturation_flux(p::AguamentiParams, Psi) = saturation_flux(p, p.sheet_law, p.h_rs, Psi)

# The flux of a layer as high as the roughness h_r (the laminar form keeps h_r^3, as Eq. 12a).
saturation_flux(p::AguamentiParams, ::LaminarTurbulentSheetLaw, h_r, Psi) = sheet_flow_law(p.k_d * h_r^3 * Psi, reynolds_factor(p))
saturation_flux(p::AguamentiParams, law::TurbulentSheetLaw, h_r, Psi) = sheet_law_flux(p, law, h_r, Psi)


"""
$(TYPEDSIGNATURES)

Cavity height carrying the flux `q` alone, Eq. (11a): the inverse of the sheet law,
`h(q) = [q (1 + omega q) / (k_d Psi)]^(1/3)` (default), or `[q / (k_s Psi^(1/2))]^(4/5)` with
[`TurbulentSheetLaw`](@ref).
"""
sheet_height_for_flux(p::AguamentiParams, q, Psi) = sheet_height_for_flux(p, p.sheet_law, q, Psi)
sheet_height_for_flux(p::AguamentiParams, ::LaminarTurbulentSheetLaw, q, Psi) =
    cbrt(q * (1 + reynolds_factor(p) * q) / (p.k_d * Psi))
sheet_height_for_flux(p::AguamentiParams, law::TurbulentSheetLaw, q, Psi) = (q / (law.k_s * sqrt(Psi)))^oftype(q, 0.8)


"""
$(TYPEDSIGNATURES)

The factors `(phi_1, phi_2)` of the channel growth rate (Appendix D1) for a sheet carrying the flux `q`:
`3 phi_1 = d ln q / d ln h` (Eq. D1) and `phi_2 = 3 phi_1 - 1`. For the laminar-turbulent law
`phi_1 = (1 + R/R_c)/(1 + 2 R/R_c)` and `phi_2 = (2 + R/R_c)/(1 + 2 R/R_c)`; for the turbulent law
(`q ~ h^(5/4)`) the constants `5/12` and `1/4`.
"""
@inline function growth_factors(p::AguamentiParams, ::LaminarTurbulentSheetLaw, q)
    r = reynolds_factor(p) * q
    return (1 + r) / (1 + 2r), (2 + r) / (1 + 2r)
end
@inline growth_factors(p::AguamentiParams, ::TurbulentSheetLaw, q) = (oftype(q, 5) / 12, oftype(q, 1) / 4)


"""
$(TYPEDSIGNATURES)

Effective pressure of a sheet carrying the flux `q` alone, Eq. (11b), for `h(q) < h_r`; `0` (uplift)
otherwise. The inverse of [`sheet_flux`](@ref), used only as a diagnostic and in tests.
"""
function sheet_N_for_flux(p::AguamentiParams, q, Psi, u_b, A)
    h = sheet_height_for_flux(p, q, Psi)
    h < p.h_r || return zero(h)
    return (u_b * (p.h_r - h) / (p.l_r * p.c_d * A * h))^(1 / p.n)
end


"""
$(TYPEDSIGNATURES)

Steady discharge of a Röthlisberger channel, Eq. (13a):
`Q_h(N) = K Psi^(beta-1) [c_e A rho_i L_w N^n / (K Psi^beta)]^(alpha/(alpha-1))`, strictly increasing
(`~ N^15 Psi^-7` for `n = 3, alpha = 5/4, beta = 3/2`). Discharge per channel [m3/s]; one channel per
spacing `l_h` carries `Q_h / l_h` per unit width.
"""
@inline function channel_discharge(p::AguamentiParams, N, Psi, A)
    ex = p.alpha / (p.alpha - 1)
    return p.K_h * Psi^(p.beta - 1) * (p.c_e * A * p.rho_i * p.L_w * N^p.n / (p.K_h * Psi^p.beta))^ex
end


"""
$(TYPEDSIGNATURES)

Effective pressure of a channel carrying the flux per unit width `q` (discharge `q l_h`), Eq. (13b).
Diagnostic inverse of [`channel_discharge`](@ref).
"""
function channel_N_for_flux(p::AguamentiParams, q, Psi, A)
    a = p.alpha
    Q = q * p.l_h
    return (p.K_h^(1 / a) * Q^(1 - 1 / a) * Psi^(1 + (p.beta - 1) / a) / (p.rho_i * p.L_w * p.c_e * A))^(1 / p.n)
end


"""
$(TYPEDSIGNATURES)

Shields depth of a canal, Eq. (16b): `h_can = S_c (rho_s - rho_w) g D_s / Psi` (prescription P1).
"""
canal_depth(p::AguamentiParams, Psi, D_s) = p.S_c * p.drho_s * p.g * D_s / Psi


"""
$(TYPEDSIGNATURES)

Steady discharge of a canal incised in till, Eq. (16c):
`Q_s(N) = k_can^2 h_can^3 Psi^2 / (rho_i L_w c_e A N^n)`, strictly decreasing and unbounded as `N -> 0`.
Discharge per canal [m3/s]; one canal per spacing `l_s`.
"""
@inline function canal_discharge(p::AguamentiParams, N, Psi, A, D_s)
    h = canal_depth(p, Psi, D_s)
    return p.k_can * p.k_can * h * h * h * Psi * Psi / (p.rho_i * p.L_w * p.c_e * A * N^p.n)
end


"""
$(TYPEDSIGNATURES)

Effective pressure of a canal carrying the flux per unit width `q` (discharge `q l_s`), Eq. (16d).
Diagnostic inverse of [`canal_discharge`](@ref).
"""
function canal_N_for_flux(p::AguamentiParams, q, Psi, A, D_s)
    h = canal_depth(p, Psi, D_s)
    return (p.k_can^2 * h^3 * Psi^2 / (p.rho_i * p.L_w * p.c_e * A * q * p.l_s))^(1 / p.n)
end


"""
$(TYPEDSIGNATURES)

Till strength bound `N_0^kappa` of Eq. (22b): the bearing capacity `F_b c_0` (Eq. 17) where the till
fraction is connected (`kappa >= kappa_c`, prescription P2), unbounded otherwise.
"""
strength_bound(p::AguamentiParams, kappa, c_0) = kappa >= p.kappa_c ? p.F_b * c_0 : oftype(c_0, Inf)


"""
$(TYPEDSIGNATURES)

Total steady flux of the elements present in a cell, Eq. (20):
`q_tot(N) = (1 - kappa) [q_h(N) + chi Q_h(N)/l_h] + kappa [q_s(N) + Q_s(N)/l_s]`.
"""
@inline function total_flux(p::AguamentiParams, N, chi, kappa, Psi, u_b, A, D_s)
    hard = zero(N)
    if kappa < 1
        hard = sheet_flux(p, N, Psi, u_b, A)
        chi > 0 && (hard += chi * channel_discharge(p, N, Psi, A) / p.l_h)
    end
    soft = zero(N)
    if kappa > 0
        soft = film_flux(p, N, Psi, u_b, A) + canal_discharge(p, N, Psi, A, D_s) / p.l_s
    end
    return (1 - kappa) * hard + kappa * soft
end


"""
$(TYPEDSIGNATURES)

Flux `q_Lambda` at which the sheet's channelization growth rate (Eq. 14a) changes sign in the
pressurized regime: the positive root of `phi_2(q) q Psi = rho_i L_w u_b h_r / l_r` (Appendix D1). With
the laminar-turbulent law `phi_2 = (2 + omega q)/(1 + 2 omega q)`, which in the laminar limit gives
Hewitt's (2011) critical discharge `rho_i L_w u_b h_r / (2 l_r Psi)` quoted below Eq. (14b); with the
turbulent law `phi_2 = 1/4` and `q_Lambda = 4 rho_i L_w u_b h_r / (l_r Psi)`.
"""
function critical_flux(p::AguamentiParams, Psi, u_b)
    c = p.rho_i * p.L_w * u_b * p.h_r / (p.l_r * Psi)
    c > 0 || return zero(c)
    return critical_flux(p, p.sheet_law, c)
end

function critical_flux(p::AguamentiParams, ::LaminarTurbulentSheetLaw, c)
    w = reynolds_factor(p)
    # (2 + w q) q = c (1 + 2 w q)  <=>  w q^2 + (2 - 2 w c) q - c = 0; stable form of the positive root
    b = 2 - 2 * w * c
    disc = sqrt(b * b + 4 * w * c)
    return b >= 0 ? 2c / (b + disc) : (disc - b) / (2w)
end

critical_flux(p::AguamentiParams, law::TurbulentSheetLaw, c) = c / growth_factors(p, law, c)[2]


"""
$(TYPEDSIGNATURES)

Onset flux of channelization, Eq. (15): `q_on = min(q_h,sat, q_Lambda)`.
"""
onset_flux(p::AguamentiParams, Psi, u_b) = min(sheet_saturation_flux(p, Psi), critical_flux(p, Psi, u_b))


"""
$(TYPEDSIGNATURES)

Local growth rate `Lambda_0` [1/s] of channelizing perturbations of a sheet carrying the flux `q`
(Eq. 14a, Appendix D1). The sheet height is that of the flux alone, `h(q)` of Eq. (11a). Below the
saturation flux (pressurized sheet, `h < h_r`) `Lambda_0 rho_i L_w h = phi_2 q Psi - rho_i L_w u_b h_r / l_r`;
at or above it (uplift) the sliding and creep terms vanish and `Lambda_0 = 3 phi_1 q Psi / (rho_i L_w h_w)`
with `h_w = h(q)` the thickness of the uplifted layer. `phi_1`, `phi_2` are those of the sheet law
(`growth_factors`). Clipped at 0 (no growth). The branch is chosen on `q >= q_h,sat` (the uplift
condition) rather than on the recomputed `h(q) >= h_r`, which round-off can put on either side at
`q = q_h,sat = q_on`.
"""
function channel_growth_rate(p::AguamentiParams, q, Psi, u_b)
    q > 0 || return zero(q)
    h = sheet_height_for_flux(p, q, Psi)
    phi1, phi2 = growth_factors(p, p.sheet_law, q)
    rho_L = p.rho_i * p.L_w
    if q < saturation_flux(p, p.sheet_law, p.h_r, Psi)
        return max(zero(q), (phi2 * q * Psi - rho_L * u_b * p.h_r / p.l_r) / (rho_L * min(h, p.h_r)))
    else
        return 3 * phi1 * q * Psi / (rho_L * h)
    end
end
