###########################################
# Model: Aguamenti v1.0 (Gregov et al.)   #
###########################################


"""
$(TYPEDSIGNATURES)

Potential along which [`AguamentiHydroModel`](@ref) routes the cell flux `q` (Eqs. 8, 25a and 26).

- `GeometricRouting()` (default): the geometric potential `phi0`, as in K24 -- the interior limit of
  assumption A2, where `-grad(phi0 - N)` reduces to `-grad(phi0)`. The routing graph is built once
  per solve.
- [`WeightedRouting`](@ref)`(; gamma0 = 0.5)`: the weighted potential of Eq. (26),
  `phi_w = phi0 - gamma N^(j-1)` with `gamma = gamma0 (1 - chi)`, the `N` of the previous step and the
  correction clipped so that its gradient never exceeds `Psi` (Eq. 26c). The feedback of `N` on the
  routing is kept where it is stabilizing (sheet and soft-bed branches) and removed in channelized
  cells (`chi = 1`), where it is the channelization instability the flux balance already represents
  within the cell. In a step the graph is built once, from the state the step starts from; in a
  steady solve `N` and `chi` are those of the solve itself and the graph is rebuilt every sweep with
  `N` under-relaxed (as for `HydraulicRouting`), up to `max_iters ÷ 2` sweeps, after which it is held
  (the flow directions jump with `N`, and a strong feedback can make the graph cycle): the solve then
  reports `converged = false` and the remaining gap in `model.info.res_route`. `gamma0 = 0` is
  `GeometricRouting()`.
- `HydraulicRouting()`: the full reduced hydraulic potential `phi = phi0 - N` of Eq. (25a), so that
  the flow direction also responds to the effective pressure where `N` varies over a few cells
  (grounding zone, regime transitions). The `N` of the potential, `model.N_route`, is carried between
  solves and relaxed towards the flux-balance `N` every sweep, `N_route += routing_relaxation (N -
  N_route)`, and the routing graph is rebuilt every sweep. Where `N` varies between neighbouring cells
  by more than the geometric potential does (`G >~ 1`: lateral bed relief at coarse resolution), the
  feedback between the flow direction and `N` (water drawn to the high-`N` cells, which then lose their
  high `N`) can keep the Picard iteration from converging; check `verbose = true`. This is the
  unweighted, unclipped `gamma = 1` of Eq. (26), including in channelized cells.

The elements always use the geometric gradient `Psi = max(|grad phi0|, Psi_min)` (A2), whichever is
chosen.
"""
abstract type AbstractRoutingPotential end

"Route along the reduced hydraulic potential `phi0 - N` (Eq. 25a). See [`AbstractRoutingPotential`](@ref)."
struct HydraulicRouting <: AbstractRoutingPotential end

"Route along the geometric potential `phi0` (default). See [`AbstractRoutingPotential`](@ref)."
struct GeometricRouting <: AbstractRoutingPotential end

"""
$(TYPEDSIGNATURES)

Route along the weighted potential of Eq. (26) (see [`AbstractRoutingPotential`](@ref)):
`phi_w = phi0 - C`, with `C` the correction `gamma N^(j-1)`, `gamma = gamma0 (1 - chi^(j))`, clipped
so that `|grad C| <= Psi` (Eq. 26c). `gamma0` [-] is the weight of the pressure term away from
channels (Table C2: 0.5); `gamma0 = 0` routes along `phi0` (`GeometricRouting`), while
`HydraulicRouting` is, apart from the clip and the lagged `N`, `gamma = 1` everywhere.

`N^(j-1)` is `state.N` when the solve starts, the sliding pressure `N_slide` of Eq. (24b) of the
previous step: the flux-balance `N` on wet cells, and on dry or cold cells `min(N_t, p_i)` (`p_i` on a
hard bed), so that water arriving at a dry cell sees the pore pressure of its till (a user decision:
v2 does not say which `N` enters Eq. (26) on cells without water; the cold or dry cells of a hard bed,
at `N = p_i`, carry the largest correction and draw water towards them). On the first step `N^(j-1)` is
whatever `state.N` holds (the initial condition: a fresh `HydroState` has `N = 0`, so the first step
routes along `phi0`; call [`update_steady_state!`](@ref) first to start from a consistent `N`). In a
steady solve `N^(j-1)` is the `N` of the solve itself (the fixed point `N^(j-1) = N`, see
[`AbstractRoutingPotential`](@ref)). `chi^(j)` is the channel indicator before this step's update.
The correction is applied on the grounded cells only, like `HydraulicRouting`'s: `phi0 - C` steps by
`C` across a face to a cell off the grounded mask, which is not clipped (towards the ocean it only
draws water to the outlet; towards ice-free land it lowers the potential of the grounded side).

The clip (Eq. 26c) is a lower envelope, a cell-value field: of all the fields `0 <= C' <= C` whose
differences across the faces between grounded cells obey `|C'_k - C'_i| <= min(Psi_i, Psi_k) d_ik /
sqrt(2)` (`d_ik = dx` or `dy`), it takes the largest, `C'_i = min_k (C_k + dist(i, k))` over the
grounded cells, with the face bounds above as distances. It only ever lowers the weight (`0 <= gamma'
<= gamma`, so `gamma' = 0` stays `0` on channelized cells and no correction appears where there was
none). The `1/sqrt(2)` per face is what guarantees the literal Eq. (26c), the norm of the cell gradient
vector `|grad(gamma N)| <= Psi`, under both of K24's stencils (`CentredGradient`: each centred
component is at most `Psi/sqrt(2)`; `FaceGradient`: each face gradient is), wherever a cell's four
neighbours are grounded. It is conservative for a gradient along a grid axis, which it caps at
`Psi/sqrt(2)`. `Psi` is the elements' gradient, floored at `Psi_min` (A2, "floored on flat cells"):
where `|grad phi0| < Psi_min` the correction may exceed the geometric gradient there.

The solve records in `model.info` the largest ratio `G_gamma` of the unclipped correction's face
difference to that face bound (`> 1` where the clip acts), `sqrt(2) max(|dC/dx|, |dC/dy|) / Psi`: between
Table A2's `G_gamma = gamma |grad N| / Psi` (gradient along a diagonal) and `sqrt(2)` times it (along an
axis); and the number of cells the clip lowered (`clipped`).
"""
struct WeightedRouting{T <: Real} <: AbstractRoutingPotential
    gamma0 ::T
    function WeightedRouting{T}(gamma0::T) where {T <: Real}
        gamma0 >= 0 || throw(ArgumentError("gamma0 must be nonnegative (got $gamma0)"))
        return new{T}(gamma0)
    end
end
WeightedRouting(gamma0::T) where {T <: Real} = WeightedRouting{T}(gamma0)
WeightedRouting(; gamma0 = 0.5) = WeightedRouting(gamma0)


"""
$(TYPEDSIGNATURES)

How the till exchange of a step is computed (`till_exchange` keyword of [`AguamentiHydroModel`](@ref)).
Both bound it by the water available at the interface, `mdot_t <= mdot_b + I/dx` (Eqs. 9a, 27), and
by the water the till holds above its densest state, `mdot_t >= -(1 - theta) h_t (e - e_min) / dt`
(the storage law with the solids conserved, Appendix D3), with `e_min` the void ratio at `N_max`. v2's
Eq. (27) writes `(theta - theta_min) h_t / dt` (a fixed `h_t`), which is `1 - theta_min = 1/(1 + e_min)`
times this bound (0.77 for Table C2's `e_min = 0.3`); the till then follows from the storage law with the exchange actually applied (Sect. 4). A
steady solve has no exchange (the till is in equilibrium with the interface).

- `ImplicitTillExchange()` (default): the uptake of each cell is solved together with its own flux
  balance, during the routing: the till relaxes (Eq. 28a) towards the `N` of the water left in the cell
  after its uptake (see `FastHydrology.TillUptake`). A deliberate deviation from
  the lagged exchange of Eq. (28b): with a lagged `N`, the till of a cell that was wet asks for more
  water than reaches it, the cell turns dry and its `N` jumps, and whole flow lines flip between wet
  and dry from step to step.
- `LaggedTillExchange()`: the exchange of Eqs. (27)-(28b) at `N^(j-1)`, the `state.N` the solve starts
  from (as for [`WeightedRouting`](@ref): `N_slide`, so `min(N_t, p_i)` on a cell that was dry; on the
  first step whatever `state.N` holds, `N = 0` for a fresh state, i.e. a till that loosens towards
  `N_min`; call [`update_steady_state!`](@ref) first), clipped to the till bounds of Eq. (18). The
  exchange is the storage change of the Eq. (28a) step towards that `N`, `(1 - theta) h_t (e(N_rel) -
  e(N_t))/dt` with `T_t` at the till stress `N_t^(j)` the step starts from: for a short step and a
  small `N_t - N` the step average of Eq. (19a), `2 k_t (N_t - N^(j-1))/(eta_w h_t) T_t/dt (1 -
  exp(-dt/T_t))` (the literal Eq. 28b), but consistent with the logarithmic storage law, so that the
  till never passes `N^(j-1)`: unless the routing limits the exchange (by the water available, or
  `release_max`), `N_t^(j+1)` is the Eq. (28a) value, between `N_t^(j)` and `N^(j-1)`. (The linear rate
  through the storage law drained more water than reaching `N^(j-1)` needs, and sent `N_t` past it, up
  to `N_max`, for `dt` of order `T_t`.)
"""
abstract type AbstractTillExchange end

"""
$(TYPEDSIGNATURES)

Which channel indicator the per-cell uptake of [`ImplicitTillExchange`](@ref) holds within its root
(the `N(q)` the till relaxes towards, see `FastHydrology.TillUptake`):

- `StepStartChi()` (default): `chi^(j)` (`model.chi_start`), the indicator the step starts from, fixed
  over the sweeps. The uptake then does not feed back on the indicator of the same step: with the
  latest one, a cell that was wet in a sweep advanced `chi` at `q > 0`, its uptake saw a higher `N`, took
  all the water and turned the cell dry; the next sweep advanced `chi` at `q = 0`, `N` fell, the till
  released and the cell was wet again, a period-2 cycle of the sweeps that kept most steps of a transient
  from converging (AIS 16 km, dt = 0.1 yr: 801 of 1000 steps; 0 with this choice, the solution changed
  only on the cells that flipped). The flux balance still advances `chi` at the routed flux.
- `LatestChi()`: the indicator of the latest flux balance (`model.chi`: `chi^(j)` in the first sweep,
  then advanced over the step at the routed flux of the previous sweep, the same as the flux balance's
  once the sweeps converge).
"""
abstract type AbstractUptakeChi end

"The uptake holds the channel indicator the step starts from (default). See [`AbstractUptakeChi`](@ref)."
struct StepStartChi <: AbstractUptakeChi end

"The uptake holds the channel indicator of the latest flux balance. See [`AbstractUptakeChi`](@ref)."
struct LatestChi <: AbstractUptakeChi end

"""
Solve each cell's till uptake with its own flux balance (default). `chi` selects the channel indicator
held within the uptake's root, [`StepStartChi`](@ref) (default) or [`LatestChi`](@ref), see
[`AbstractUptakeChi`](@ref). See [`AbstractTillExchange`](@ref).
"""
struct ImplicitTillExchange{C <: AbstractUptakeChi} <: AbstractTillExchange
    chi::C
end
ImplicitTillExchange(; chi::AbstractUptakeChi = StepStartChi()) = ImplicitTillExchange(chi)

"The lagged, bounded till exchange of Eqs. (27)-(28b). See [`AbstractTillExchange`](@ref)."
struct LaggedTillExchange <: AbstractTillExchange end


"""
$(TYPEDSIGNATURES)

Every array buffer of [`AguamentiHydroModel`](@ref) beyond those of its routing engine: the bed data,
the two state fields (`N_t`, `chi`), and the outputs of the flux balance and the till reservoir.
"""
struct AguamentiWorkspace{A}
    # Bed data (Fig. 2, Table C2)
    kappa    ::A  # soft-bed (till) area fraction [0-1]
    h_t      ::A  # active till thickness [m]
    k_t      ::A  # till permeability [m2]
    c_0      ::A  # till cohesion [Pa]
    D_s      ::A  # mobile grain size of the till [m]
    cold     ::A  # 1 where the base is cold (thermal mask from the ice model), else 0
    # State (carried in time)
    N_t      ::A  # till effective stress [Pa]
    chi      ::A  # channel indicator [0-1]; after a step, chi^(j+1) (the flux balance of the step uses it)
    # Supply
    p_i      ::A  # ice overburden pressure rho_i g H [Pa]
    Psi      ::A  # geometric hydraulic gradient |grad phi0| used by the elements (floored at Psi_min) [Pa/m]
    phi0_true::A  # scratch: the geometric potential while the routing engine holds phi0 - N
    N_route  ::A  # effective pressure of the routing potential: phi0 - N_route (HydraulicRouting), relaxed towards N_wet; the N of the correction (WeightedRouting, steady solve) [Pa]
    N_lag    ::A  # state.N when the solve starts: N^(j-1) of Eqs. (26a), (28b) and of the creep term of Eq. (21) [Pa]
    route_corr ::A # correction C = gamma N subtracted from phi0 by WeightedRouting, after the clip of Eq. (26c) [Pa]
    # Till reservoir
    mdot_t   ::A  # till exchange, positive into the till [m/s], after the bound of Eq. (9a)
    demand   ::A  # exchange the till asks for before the bounds [m/s]: Eq. (28b) at N_lag (LaggedTillExchange), or the relaxation's at N_wet (ImplicitTillExchange, a diagnostic: the uptake is solved per cell, see TillUptake)
    dN_u     ::A  # undrained increment of the till stress over this step, (1 - B) rho_i g dH [Pa] (Eq. 28a)
    storage  ::A  # (1 - theta) h_t at the till stress the step starts from [m]
    refreeze ::A  # routed water refrozen in cold cells [m/s] (returned to the ice model as accretion)
    release_max ::A  # most water the till can release this step above its residual content, Eq. (9b) [m/s]
    freeze_deficit ::A  # freeze-on neither the routed water nor the till could supply [m/s] (for the ice model's accretion)
    dry      ::A  # 1 where the cell has no interfacial water (cold, or the till absorbs it all), else 0
    # Flux balance
    q_in     ::A  # routed flux before dry cells are set to 0: the flux the flux balance, the till target and the dissipation see [m2/s]
    chi_start::A  # channel indicator when the solve starts, chi^(j): advanced from it in every sweep of a step (Eq. 21), the history the steady state's hysteresis keeps
    N_star   ::A  # root of the flux balance on the active branch (0: no root) [Pa]
    N_inf    ::A  # far-field effective pressure, Eq. (22a) [Pa]
    N_wet    ::A  # flux-balance effective pressure with the grounding-line correction, Eq. (23) [Pa]
    regime   ::A  # regime code (AGUAMENTI_* constants)
    q_on     ::A  # onset flux of channelization, Eq. (15) [m2/s]
    q_sn     ::A  # saddle-node flux [m2/s]
    N_sn     ::A  # effective pressure of the saddle node, where q_tot(N; chi = 1) is smallest [Pa]
    Lambda0  ::A  # channel growth rate, Eq. (14a) [1/s]
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

Outcome of the last solve of an [`AguamentiHydroModel`](@ref) (`model.info`): the number of Picard
sweeps, whether it converged, the final relative changes of `q` and `N` over the last sweep, the wall
time [s], the residual history `(res_q, res_N)` of every sweep, the number of routing graphs built,
and, with [`WeightedRouting`](@ref), the ratio `G_gamma` of the unclipped correction's gradient to its
bound and the number of cells the clip of Eq. (26c) lowered (of the last graph built; `NaN` and 0
otherwise), and, in a steady solve with `WeightedRouting`, `res_route`, the largest change of `N`
relative to its maximum between the `N` the last graph was built with and the final one (the graph is
held after `max_iters ÷ 2` sweeps if it has not settled, and `converged` is then `false`; `NaN`
otherwise).
"""
mutable struct AguamentiSolveInfo{T}
    iters     ::Int
    converged ::Bool
    res_q     ::T
    res_N     ::T
    elapsed   ::Float64
    history   ::Vector{NTuple{2, T}}
    routing_builds ::Int
    G_gamma   ::T
    clipped   ::Int
    res_route ::T
end
AguamentiSolveInfo{T}() where {T} = AguamentiSolveInfo{T}(0, false, T(NaN), T(NaN), 0.0, NTuple{2, T}[], 0, T(NaN), 0, T(NaN))

"""
$(TYPEDSIGNATURES)

The Aguamenti v1.0 subglacial hydrology (Gregov, Angelides, Scheiter and Kazmierczak), a quasi-steady
extension of Kazmierczak et al. (2024) for hard, soft and mixed beds. In every grounded cell it
computes the effective pressure `N` from five components (Fig. 2), which together form the reduced
problem of Eqs. (25a)-(25f):

1. **Supply** (Sect. 3.2): basal melt from the geothermal flux, frictional heating and conduction, plus
   the dissipation of the drainage system, `mdot_w^b` (Eq. 7) -- the same terms as K24
   (`G`, `q_T`, `sliding_law`, `i_eb`, dissipation melt on by default).
2. **Routing** (Sect. 3.3): the steady cell flux `q` along the hydraulic potential (Eqs. 8, 26), with
   the till exchange as a sink bounded by the water available (Eqs. 9a, 27). Done by K24's routing
   engine (`model.routing`, a [`KazmierczakHydroModel`](@ref) used only for its potential, routing and
   source terms), with Aguamenti's own replay of the routing tape for the till exchange and cold cells.
3. **Element library** (Sect. 3.4): the steady fluxes of the sheet, channel, film and canal as
   functions of `N` (elements.jl).
4. **Till reservoir** (Sect. 3.5): the prognostic till effective stress `N_t` (Eqs. 18-19, till.jl).
5. **Flux balance** (Sect. 3.6): `q_tot(N) = q` (Eq. 20) on the active branch, the regimes of
   Eq. (22), the grounding-line correction of Eq. (23), and the channel indicator `chi` (Eq. 21)
   (flux_balance.jl).

`state.N` is the effective pressure passed to the friction law, `N_slide` of Eq. (24b): `N` in wet
cells, `min(N_t, p_i)` in dry or cold cells, and `p_i` in a dry or cold cell of a hard bed (a user
decision consistent with Eq. 24b: with no till, `N_t` is undefined and taken as `p_i`). It is also the
`N^(j-1)` of the next step (Eqs. 21, 26a, 28b). `state.W` is the
mean height of the distributed water layer at that `N` (sheet cavities and film), a diagnostic.

Two quantities evolve in time, the till effective stress `model.N_t` and the channel indicator
`model.chi`. Advance them with [`step!`](@ref)`(model, grid, state, dt; dH_dt)` (Sect. 4): within the
step, `chi` is advanced (Eq. 21) at the routed flux before the flux balance uses it, and the till
(Eq. 28a) with the exchange actually applied in the routing. A [`SteadyStateSimulation`](@ref) (or
[`update_steady_state!`](@ref)) instead puts the till in equilibrium with the interface (`N_t = N`, no
exchange) and the channels in the steady state of Eq. (21), keeping the current `chi` in the bistable
band.

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

# Keywords (parameters, Tables C1-C2; names follow the paper)

`glen_n = 3`, `rho_i = 917`, `rho_w = 1000`, `rho_sw = 1028`, `L_w = 3.35e5`, `eta_w = 1.8e-3`,
`g = 9.81`, `h_r = 0.1`, `l_r = 2`, `h_rs = 0.02`, `l_rs = 0.5`, `c_d = 2/n^n`, `k_d = 1e-2`,
`Re_c = 1000`, `c_e = 2/n^n`, `f = 0.1`, `alpha = 5/4`, `beta = 3/2`, `l_h = 10e3`, `l_s = 300`,
`S_c = 0.045`, `drho_s = 1650`, `F_b = 5`, `kappa_c = 0.5`, `e_r = 0.69`, `N_r = 1e3`, `C_t = 0.12`,
`e_max = 0.9`, `e_min = 0.3`, `B = 0.8`, `eps_u = 0.01`. The parameters of Table B2 (SHMIP, v2) are
passed the same way, e.g. `rho_i = 910`, `c_0 = 100e3`, with `A_visc_in = 2.5e-25` and
`abs_v_b_in = 1e-6` as fields.

# Keywords (element options)

`sheet_law = LaminarTurbulentSheetLaw()`: the flow law of the sheet and the film, or
[`TurbulentSheetLaw`](@ref)`(; k_s = 5e-3)` (GlaDS, Table B2); see [`AbstractSheetLaw`](@ref). It sets
the distributed fluxes, their saturation, the onset flux and the channel growth rate.

`channels = true`: `false` removes the channels (Table B2, a sheet-only run): the channel indicator
stays 0 (`chi0` must be 0), there is no onset or saddle-node flux, and the flux balance never takes the
channelized branch.

Numerical: `Psi_min = 1.0` (floor on the gradient seen by the elements [Pa/m], for flat potentials, Table C2),
`N_lo = 1e-3`, `N_hi = 1e10` (search interval [Pa]), `root_iters = 60` (cap on the evaluations of
each per-cell Brent search; they converge to ~1e-12 in far fewer), `chi_min = 1e-6`
(a decaying indicator below it is set to 0), `chi_developed = 0.99` (a growing indicator below it is
a channel still developing, see [`flux_balance_root`](@ref FastHydrology.flux_balance_root)),
`max_iters = 30` and `rtol = 1e-6` (the Picard loop on `q` and `N`), `routing_relaxation = 0.5`
(under-relaxation of the `N` in the routing potential of `HydraulicRouting`, and of `WeightedRouting`
in a steady solve, see [`AbstractRoutingPotential`](@ref)), `lagged_friction = true` (see below),
`verbose = false`.

`till_exchange = ImplicitTillExchange()`: how the till exchange of a step is computed, see
[`AbstractTillExchange`](@ref).

With a sliding law that depends on `N`, `lagged_friction = true` evaluates the basal stress (and so
the frictional melt) once per solve, from the `state.N` the solve starts from, and the Picard loop
iterates only routing, dissipation and flux balance; a coupled host iterates the friction feedback
across its own calls (as it does through its velocity anyway). `false` re-evaluates `tau_b` at the
current `N` in every sweep, a joint `(q, N)` loop that on real ice sheets ends in a period-2 cycle in
cells where the routed flux is a small residual of frictional melt and upstream refreezing, or where
`q` crosses the onset flux (as K24's joint loop, see the K24 friction-loop notes). For a sliding law
independent of `N` the two are the same.

# Keywords (routing, passed to the routing engine)

`routing_potential = GeometricRouting()` (or [`WeightedRouting`](@ref)`(; gamma0 = 0.5)` for Eq. (26),
or `HydraulicRouting()`, see [`AbstractRoutingPotential`](@ref)),
`sliding_law = NoFrictionSlidingLaw()`, `routing_scheme = Warner()`, `coupling_length_kamb86 = 0.0`
(the stress-gradient smoothing of K24, off by default), `dissipation_melt = true`, and
`routing_options = (;)`, further routing keywords of [`KazmierczakHydroModel`](@ref): `fill_algorithm`,
`fill_iters`, `q_conversion`, `gradient_discretization`, `potential_weight`, `friction_discretization`,
`dissipation_discretization`, `q_min`, `q_max`, `max_psi_out_calls` (any other key is an error: the
K24 closure and loop settings are not used). The routing engine always uses `TapedPsiOut()`, and the
dissipation melt is the cell-centred `q Psi` of Eq. (7b) (`CellCentredDissipation()`) unless
`routing_options` says otherwise.
"""
struct AguamentiHydroModel{T <: AbstractFloat, A, RP <: AbstractRoutingPotential, TE <: AbstractTillExchange,
                           KM <: KazmierczakHydroModel, P <: AguamentiParams{T}} <: AbstractHydroModel
    params            ::P
    routing_potential ::RP
    till_exchange     ::TE
    routing           ::KM
    workspace         ::AguamentiWorkspace{A}
    max_iters         ::Int
    rtol              ::T
    routing_relaxation ::T
    lagged_friction   ::Bool
    verbose           ::Bool
    info              ::AguamentiSolveInfo{T}
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
    # Element options
    sheet_law::AbstractSheetLaw = LaminarTurbulentSheetLaw(), channels::Bool = true,
    # Numerical
    Psi_min = 1.0, N_lo = 1e-3, N_hi = 1e10, root_iters = 60, chi_min = 1e-6, chi_developed = 0.99,
    max_iters = 30, rtol = 1e-6, routing_relaxation = 0.5, lagged_friction = true, verbose = false,
    # Routing
    routing_potential::AbstractRoutingPotential = GeometricRouting(), till_exchange::AbstractTillExchange = ImplicitTillExchange(),
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

    params = AguamentiParams{T, typeof(convert_sheet_law(T, sheet_law))}(
        n, T(rho_i), rho_w, T(rho_sw), T(L_w), T(eta_w), T(g),
        T(h_r), T(l_r), T(h_rs), T(l_rs), c_d, T(k_d), T(Re_c),
        c_e, f, T(alpha), T(beta), K_h, T(l_h),
        T(l_s), T(S_c), T(drho_s), k_can,
        T(F_b), T(kappa_c),
        T(e_r), T(N_r), T(C_t), T(e_max), T(e_min), T(B),
        T(eps_u), T(Psi_min), T(N_lo), T(N_hi), Int(root_iters), T(chi_min), T(chi_developed),
        convert_sheet_law(T, sheet_law), channels)

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
    channels || all(iszero, chi) || throw(ArgumentError("chi0 must be 0 with channels = false"))

    workspace = AguamentiWorkspace(
        alloc_field(grid, T.(kappa_in)), field(h_t), field(k_t), field(c_0), field(D_s), cold_field,
        N_t, chi,
        (alloc_field(grid) for _ in 1:(fieldcount(AguamentiWorkspace) - 8))...)

    return AguamentiHydroModel(params, routing_potential, till_exchange, routing, workspace, Int(max_iters), T(rtol), T(routing_relaxation), Bool(lagged_friction), Bool(verbose),
                               AguamentiSolveInfo{T}())
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
