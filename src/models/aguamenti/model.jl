###########################################
# Model: Aguamenti v1.0 (Gregov et al.)   #
###########################################


"""
$(TYPEDSIGNATURES)

Potential along which [`AguamentiHydroModel`](@ref) routes the cell flux `q` (Eq. 8b / 26a).

- `GeometricRouting()` (default): the geometric potential `phi0`, as in K24 -- the interior limit of
  approximation A2 (`G << 1`), where `-grad(phi0 - N)` reduces to `-grad(phi0)`. The routing graph is
  built once per solve.
- `HydraulicRouting()`: the reduced hydraulic potential `phi = phi0 - N` of Eq. (26a), so that the
  flow direction also responds to the effective pressure where `N` varies over a few cells
  (grounding zone, regime transitions). The `N` of the potential, `model.N_route`, is carried between
  solves and relaxed towards the flux-balance `N` every sweep, `N_route += routing_relaxation (N -
  N_route)`, and the routing graph is rebuilt every sweep. Where `N` varies between neighbouring cells
  by more than the geometric potential does (`G >~ 1`: lateral bed relief at coarse resolution), the
  feedback between the flow direction and `N` (water drawn to the high-`N` cells, which then lose their
  high `N`) can keep the Picard iteration from converging; check `verbose = true`.

The elements always use the geometric gradient `Psi = |grad phi0|` (A2), whichever is chosen.
"""
abstract type AbstractRoutingPotential end

"Route along the reduced hydraulic potential `phi0 - N` (Eq. 26a). See [`AbstractRoutingPotential`](@ref)."
struct HydraulicRouting <: AbstractRoutingPotential end

"Route along the geometric potential `phi0` (default). See [`AbstractRoutingPotential`](@ref)."
struct GeometricRouting <: AbstractRoutingPotential end


"""
$(TYPEDSIGNATURES)

Every array buffer of [`AguamentiHydroModel`](@ref) beyond those of its routing engine: the bed data,
the two state fields (`N_t`, `chi`), and the outputs of the flux balance and the till reservoir.
"""
struct AguamentiWorkspace{A}
    # Bed data (Fig. 2, Table A2)
    kappa    ::A  # soft-bed (till) area fraction [0-1]
    h_t      ::A  # active till thickness [m]
    k_t      ::A  # till permeability [m2]
    c_0      ::A  # till cohesion [Pa]
    D_s      ::A  # mobile grain size of the till [m]
    cold     ::A  # 1 where the base is cold (thermal mask from the ice model), else 0
    # State (carried in time)
    N_t      ::A  # till effective stress [Pa]
    chi      ::A  # channel indicator [0-1]
    # Supply
    p_i      ::A  # ice overburden pressure rho_i g H [Pa]
    Psi      ::A  # geometric hydraulic gradient |grad phi0| used by the elements (floored at Psi_min) [Pa/m]
    phi0_true::A  # scratch: the geometric potential while the routing engine holds phi0 - N
    N_route  ::A  # effective pressure of the routing potential phi0 - N_route (HydraulicRouting), relaxed towards N_wet [Pa]
    # Till reservoir
    mdot_t   ::A  # till exchange, positive into the till [m/s], after the bound of Eq. (9a)
    demand   ::A  # exchange the till relaxation asks for at the cell's N_wet [m/s] (diagnostic; the uptake is solved per cell, see TillUptake)
    N_u      ::A  # till effective stress after the undrained response of this step [Pa]
    storage  ::A  # (1 - theta) h_t at N_u [m]
    refreeze ::A  # routed water refrozen in cold cells [m/s] (returned to the ice model as accretion)
    release_max ::A  # most water the till can release this step above its residual content, Eq. (9b) [m/s]
    freeze_deficit ::A  # freeze-on neither the routed water nor the till could supply [m/s] (for the ice model's accretion)
    dry      ::A  # 1 where the cell has no interfacial water (cold, or the till absorbs it all), else 0
    # Flux balance
    q_in     ::A  # routed flux before dry cells are set to 0: the flux the flux balance (and the till target) sees [m2/s]
    chi_start::A  # channel indicator before a steady solve (the history its hysteresis keeps)
    N_star   ::A  # root of the flux balance on the active branch (0: no root) [Pa]
    N_inf    ::A  # far-field effective pressure, Eq. (23a) [Pa]
    N_wet    ::A  # flux-balance effective pressure with the grounding-line correction, Eq. (24) [Pa]
    regime   ::A  # regime code (AGUAMENTI_* constants)
    q_on     ::A  # onset flux of channelization, Eq. (16) [m2/s]
    q_sn     ::A  # saddle-node flux [m2/s]
    Lambda0  ::A  # channel growth rate, Eq. (15a) [1/s]
    q_h      ::A  # sheet flux at N [m2/s] (per unit width of the hard-bed part)
    q_ch     ::A  # channel flux chi Q_h / l_h at N [m2/s]
    q_s      ::A  # film flux at N [m2/s] (per unit width of the soft-bed part)
    q_can    ::A  # canal flux Q_s / l_s at N [m2/s]
    # Picard loop
    q_prev   ::A
    N_prev   ::A
end


"""
$(TYPEDSIGNATURES)

The Aguamenti v1.0 subglacial hydrology (Gregov, Angelides, Scheiter and Kazmierczak), a quasi-steady
extension of Kazmierczak et al. (2024) for hard, soft and mixed beds. In every grounded cell it
computes the effective pressure `N` from five components (Fig. 2):

1. **Supply** (Sect. 2.2): basal melt from the geothermal flux, frictional heating and conduction, plus
   the dissipation of the drainage system, `mdot_w^b` (Eq. 7) -- the same terms as K24
   (`G`, `q_T`, `sliding_law`, `i_eb`, dissipation melt on by default).
2. **Routing** (Sect. 2.3): the steady cell flux `q` along the hydraulic potential (Eq. 8), with the
   till exchange as a sink bounded by the water available (Eq. 9a). Done by K24's routing engine
   (`model.routing`, a [`KazmierczakHydroModel`](@ref) used only for its potential, routing and source
   terms), with Aguamenti's own replay of the routing tape for the till exchange and cold cells.
3. **Element library** (Sect. 2.4): the steady fluxes of the sheet, channel, film and canal as
   functions of `N` (elements.jl).
4. **Till reservoir** (Sect. 2.5): the prognostic till effective stress `N_t` (Eqs. 19-20, till.jl).
5. **Flux balance** (Sect. 2.6): `q_tot(N) = q` on the active branch, the regimes of Eq. (23), the
   grounding-line correction of Eq. (24), and the channel indicator `chi` (Eq. 22) (flux_balance.jl).

`state.N` is the effective pressure passed to the friction law, `N_slide` of Eq. (25b): `N` in wet
cells, `min(N_t, p_i)` in dry or cold cells (`p_i` on a hard bed). `state.W` is the mean height of the
distributed water layer at that `N` (sheet cavities and film), a diagnostic.

Two quantities evolve in time, the till effective stress `model.N_t` and the channel indicator
`model.chi`. Advance them with [`step!`](@ref)`(model, grid, state, dt; dH_dt)`; a
[`SteadyStateSimulation`](@ref) (or [`update_steady_state!`](@ref)) instead puts the till in
equilibrium with the interface (`N_t = N`, no exchange) and the channels in the steady state of
Eq. (22), keeping the current `chi` in the bistable band.

# Arguments

- `grid::AbstractHydroGrid`: grid of the simulation
- `kappa_in`: soft-bed (till) area fraction, 0 (hard bed) to 1 (soft bed)
- `abs_v_b_in`: basal sliding speed `u_b` [m/s]
- `A_visc_in`: rate factor of Glen's law at the bed [Pa^-n s^-1]
- `G_in`: geothermal heat flux [W/m2]
- `q_T_in`: conductive heat flux from the bed into the ice [W/m2]

# Keywords (bed data, scalar or field)

`h_t = 5.0` (till thickness [m]), `k_t = 1e-16` (till permeability [m2]), `c_0 = 20e3` (till cohesion
[Pa]), `D_s = 2e-3` (grain size [m]), `cold = nothing` (thermal mask: `true`/`1` where the base is
cold; `nothing` = temperate everywhere), `i_eb = nothing` (water from above [kg/m2/s]).

# Keywords (initial state)

`N_t0 = nothing` (initial till effective stress [Pa]; `nothing` = the reference stress `N_r`, clipped
to the bounds), `chi0 = 0.0` (initial channel indicator).

# Keywords (parameters, Tables A1-A2; names follow the paper)

`glen_n = 3`, `rho_i = 917`, `rho_w = 1000`, `rho_sw = 1028`, `L_w = 3.35e5`, `eta_w = 1.8e-3`,
`g = 9.81`, `h_r = 0.1`, `l_r = 2`, `h_rs = 0.02`, `l_rs = 0.5`, `c_d = 2/n^n`, `k_d = 1e-2`,
`Re_c = 1000`, `c_e = 2/n^n`, `f = 0.1`, `alpha = 5/4`, `beta = 3/2`, `l_h = 10e3`, `l_s = 300`,
`S_c = 0.045`, `drho_s = 1650`, `F_b = 5`, `kappa_c = 0.5`, `e_r = 0.69`, `N_r = 1e3`, `C_t = 0.12`,
`e_max = 0.9`, `e_min = 0.3`, `B = 0.8`, `eps_u = 0.01`.

Numerical: `Psi_min = 1.0` (floor on the gradient seen by the elements [Pa/m], for flat potentials),
`N_lo = 1e-3`, `N_hi = 1e10` (search interval [Pa]), `root_iters = 60`, `chi_min = 1e-6`
(a decaying indicator below it is set to 0), `chi_developed = 0.99` (a growing indicator below it is
a channel still developing, see [`flux_balance_root`](@ref FastHydrology.flux_balance_root)),
`max_iters = 30` and `rtol = 1e-6` (the Picard loop on `q` and `N`), `routing_relaxation = 0.5`
(under-relaxation of the `N` in the routing potential of `HydraulicRouting`, see
[`AbstractRoutingPotential`](@ref)), `verbose = false`.

# Keywords (routing, passed to the routing engine)

`routing_potential = GeometricRouting()` (see [`AbstractRoutingPotential`](@ref)),
`sliding_law = NoFrictionSlidingLaw()`, `routing_scheme = Warner()`, `coupling_length_kamb86 = 0.0`
(the stress-gradient smoothing of K24, off by default), `dissipation_melt = true`, and
`routing_options = (;)`, further routing keywords of [`KazmierczakHydroModel`](@ref): `fill_algorithm`,
`fill_iters`, `q_conversion`, `gradient_discretization`, `potential_weight`, `friction_discretization`,
`dissipation_discretization`, `q_min`, `q_max`, `max_psi_out_calls` (any other key is an error: the
K24 closure and loop settings are not used). The routing engine always uses `TapedPsiOut()`, and the
dissipation melt is the cell-centred `q Psi` of Eq. (7b) (`CellCentredDissipation()`) unless
`routing_options` says otherwise.
"""
struct AguamentiHydroModel{T <: AbstractFloat, A, RP <: AbstractRoutingPotential, KM <: KazmierczakHydroModel} <: AbstractHydroModel
    params            ::AguamentiParams{T}
    routing_potential ::RP
    routing           ::KM
    workspace         ::AguamentiWorkspace{A}
    max_iters         ::Int
    rtol              ::T
    routing_relaxation ::T
    verbose           ::Bool
end

function Base.getproperty(model::AguamentiHydroModel, name::Symbol)
    hasfield(AguamentiHydroModel, name) && return getfield(model, name)
    params = getfield(model, :params)
    hasfield(typeof(params), name) && return getfield(params, name)
    ws = getfield(model, :workspace)
    hasfield(typeof(ws), name) && return getfield(ws, name)
    return getproperty(getfield(model, :routing), name)   # q, psi_out, mdot_total, Q_b, Q_diss, phi0, ...
end

Base.propertynames(model::AguamentiHydroModel, private::Bool = false) =
    (fieldnames(AguamentiHydroModel)..., fieldnames(typeof(getfield(model, :params)))...,
     fieldnames(typeof(getfield(model, :workspace)))..., propertynames(getfield(model, :routing))...)


const AGUAMENTI_DEFAULT_L_W = 3.35e5

# KazmierczakHydroModel keywords that change the routing, potential or source discretization, the only
# ones the Aguamenti solve reads (its effective-pressure closure and Picard loops are not used).
const AGUAMENTI_ROUTING_OPTIONS = (:fill_algorithm, :fill_iters, :q_conversion, :gradient_discretization, :potential_weight,
                                   :friction_discretization, :dissipation_discretization, :q_min, :q_max, :max_psi_out_calls)

function AguamentiHydroModel(
    grid::AbstractHydroGrid,
    kappa_in::AbstractArray{<:AbstractFloat},
    abs_v_b_in::AbstractArray{<:AbstractFloat},
    A_visc_in::AbstractArray{<:AbstractFloat},
    G_in::AbstractArray{<:AbstractFloat},
    q_T_in::AbstractArray{<:AbstractFloat};
    # Bed data
    h_t = 5.0, k_t = 1e-16, c_0 = 20e3, D_s = 2e-3, cold = nothing, i_eb = nothing,
    # Initial state
    N_t0 = nothing, chi0 = 0.0,
    # Ice and water
    glen_n = 3.0, rho_i = 917.0, rho_w = 1000.0, rho_sw = 1028.0, L_w = AGUAMENTI_DEFAULT_L_W,
    eta_w = 1.8e-3, g = 9.81,
    # Sheet and film
    h_r = 0.1, l_r = 2.0, h_rs = 0.02, l_rs = 0.5, c_d = nothing, k_d = 1e-2, Re_c = 1000.0,
    # Channel
    c_e = nothing, f = 0.1, alpha = 5/4, beta = 3/2, l_h = 10e3,
    # Canal
    l_s = 300.0, S_c = 0.045, drho_s = 1650.0,
    # Till strength and reservoir
    F_b = 5.0, kappa_c = 0.5, e_r = 0.69, N_r = 1e3, C_t = 0.12, e_max = 0.9, e_min = 0.3, B = 0.8,
    # Flux balance
    eps_u = 0.01,
    # Numerical
    Psi_min = 1.0, N_lo = 1e-3, N_hi = 1e10, root_iters = 60, chi_min = 1e-6, chi_developed = 0.99,
    max_iters = 30, rtol = 1e-6, routing_relaxation = 0.5, verbose = false,
    # Routing
    routing_potential::AbstractRoutingPotential = GeometricRouting(),
    sliding_law = NoFrictionSlidingLaw(), routing_scheme = Warner(), coupling_length_kamb86 = 0.0,
    dissipation_melt = true, routing_options::NamedTuple = (;),
)
    expected_size = (grid.Nx, grid.Ny)
    for (name, arr) in [("kappa", kappa_in), ("abs_v_b", abs_v_b_in), ("A_visc", A_visc_in), ("G", G_in), ("q_T", q_T_in)]
        size(arr)[1:2] == expected_size || throw(ArgumentError("$name size $(size(arr)) != grid size $expected_size"))
    end
    all(k -> 0 <= k <= 1, kappa_in) || throw(ArgumentError("kappa must lie in [0, 1]"))
    haskey(routing_options, :psi_out_algorithm) &&
        throw(ArgumentError("AguamentiHydroModel replays the routing tape itself, so the routing engine always uses TapedPsiOut()"))
    for key in keys(routing_options)
        key in AGUAMENTI_ROUTING_OPTIONS || throw(ArgumentError("routing_options key $key is not used by AguamentiHydroModel; " *
            "allowed: $(join(AGUAMENTI_ROUTING_OPTIONS, ", "))"))
    end

    T = typeof(grid.dx)
    n = T(glen_n)
    c_d = c_d === nothing ? 2 / n^n : T(c_d)
    c_e = c_e === nothing ? 2 / n^n : T(c_e)
    f = T(f)
    rho_w = T(rho_w)
    K_h = (T(2) / T(pi))^(T(0.25)) * sqrt((T(pi) + T(2)) / (rho_w * f))   # semicircular conduit, as K24
    k_can = sqrt(T(8) / (rho_w * f))
    e_min < e_r < e_max || throw(ArgumentError("need e_min < e_r < e_max (got $e_min, $e_r, $e_max)"))

    params = AguamentiParams{T}(
        n, T(rho_i), rho_w, T(rho_sw), T(L_w), T(eta_w), T(g),
        T(h_r), T(l_r), T(h_rs), T(l_rs), c_d, T(k_d), T(Re_c),
        c_e, f, T(alpha), T(beta), K_h, T(l_h),
        T(l_s), T(S_c), T(drho_s), k_can,
        T(F_b), T(kappa_c),
        T(e_r), T(N_r), T(C_t), T(e_max), T(e_min), T(B),
        T(eps_u), T(Psi_min), T(N_lo), T(N_hi), Int(root_iters), T(chi_min), T(chi_developed))

    # The routing engine: K24's potential, routing and source terms. Its own effective-pressure
    # closure is never called.
    routing = KazmierczakHydroModel(grid, kappa_in, abs_v_b_in, A_visc_in, G_in, q_T_in;
        i_eb = i_eb, rho_w = rho_w, rho_sw = rho_sw, rho_i = rho_i, g = g, L_w = L_w, glen_n = glen_n,
        f = f, water_viscosity = eta_w, coupling_length_kamb86 = coupling_length_kamb86,
        routing_scheme = routing_scheme, sliding_law = sliding_law, dissipation_melt = dissipation_melt,
        dissipation_verbose = false, qN_verbose = false, dissipation_discretization = CellCentredDissipation(),
        routing_options..., psi_out_algorithm = TapedPsiOut())

    field(x) = x isa AbstractArray ? (size(x)[1:2] == expected_size ? alloc_field(grid, T.(x)) :
                                      throw(ArgumentError("field size $(size(x)) != grid size $expected_size"))) :
                                     (a = alloc_field(grid); a .= T(x); a)
    cold_field = cold === nothing ? alloc_field(grid) : field(T.(cold .!= 0))

    N_t = alloc_field(grid)
    if N_t0 === nothing
        N_t .= clamp(params.N_r, till_stress(params, params.e_max), till_stress(params, params.e_min))
    else
        N_t .= N_t0
    end
    chi = field(chi0)
    all(c -> 0 <= c <= 1, chi) || throw(ArgumentError("chi0 must lie in [0, 1]"))

    workspace = AguamentiWorkspace(
        alloc_field(grid, T.(kappa_in)), field(h_t), field(k_t), field(c_0), field(D_s), cold_field,
        N_t, chi,
        (alloc_field(grid) for _ in 1:(fieldcount(AguamentiWorkspace) - 8))...)

    return AguamentiHydroModel(params, routing_potential, routing, workspace, Int(max_iters), T(rtol), T(routing_relaxation), Bool(verbose))
end


"""
$(TYPEDSIGNATURES)

`true`: the sheet and film fluxes depend on the sliding speed through the cavity opening, so a host
solves `N` inside its velocity iteration with [`N_from_ub!`](@ref).
"""
N_responds_to_ub(::AguamentiHydroModel) = true


"""
$(TYPEDSIGNATURES)

Update the bed data and supply fields of an [`AguamentiHydroModel`](@ref) between solves, in place.
Any keyword left at `nothing` is unchanged: `kappa`, `abs_v_b`, `A_visc`, `h_t`, `k_t`, `c_0`, `D_s`,
`cold` (thermal mask), and the heat terms `G`, `q_T`, `i_eb` (via [`set_basal_terms!`](@ref)).
"""
function set_aguamenti_inputs!(model::AguamentiHydroModel; kappa = nothing, abs_v_b = nothing, A_visc = nothing,
                               h_t = nothing, k_t = nothing, c_0 = nothing, D_s = nothing, cold = nothing,
                               G = nothing, q_T = nothing, i_eb = nothing)
    r = model.routing
    if kappa !== nothing
        model.kappa .= kappa
        r.kappa .= kappa
    end
    abs_v_b === nothing || (r.abs_v_b .= abs_v_b)
    A_visc === nothing || (r.A_visc .= A_visc)
    h_t === nothing || (model.h_t .= h_t)
    k_t === nothing || (model.k_t .= k_t)
    c_0 === nothing || (model.c_0 .= c_0)
    D_s === nothing || (model.D_s .= D_s)
    cold === nothing || (model.cold .= (cold .!= 0))
    set_basal_terms!(r; G = G, q_T = q_T, i_eb = i_eb)
    return nothing
end
