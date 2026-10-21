#################################
# Model: Kazmierczak et al 2024 #
#################################


"""
$(TYPEDSIGNATURES)

Physical constants and solver configuration for `KazmierczakHydroModel` -- everything that is fixed
once the model is constructed and never touched again during a solve. Split out from
`KazmierczakWorkspace` (the mutable array buffers) so the two can be reasoned about and constructed
independently instead of interleaved as 40 flat fields on one struct; see `KazmierczakHydroModel`
for how the split is made transparent to callers.
"""
struct KazmierczakParams{T <: AbstractFloat, D <: AbstractDissipationMelt, L <: AbstractSlidingLaw, P <: AbstractPsiOutAlgorithm, WT <: AbstractWaterThicknessAlgorithm, M <: AbstractDrainageMode, S <: AbstractSensibleHeat,
                         FA <: AbstractFillAlgorithm, RS <: AbstractRoutingScheme, QC <: AbstractQConversion, DD <: AbstractDissipationDiscretization,
                         GD <: AbstractGradientDiscretization, PW <: AbstractPotentialWeight, FD <: AbstractFrictionDiscretization}

    rho_w           ::T    # Density of fresh water [kg/m3]
    rho_sw          ::T    # Density of sea water [kg/m3], used only for the b term of the erf argument in update_N! (so N -> 0 at flotation); defaults to 1028 (flotation density, so N = 0 at the grounding line); set rho_sw = rho_w for the literal K24 Eq. 7
    rho_i           ::T    # Density of ice [kg/m3]
    g               ::T    # Gravitational acceleration [m/s2]
    L_w             ::T    # Latent heat of fusion for ice [J/kg]
    n               ::T    # Glen's flow law exponent (typically 3)
    h_b             ::T    # Typical bed obstacle height [m]
    alpha           ::T    # Power law exponent for hydraulic transmissivity (m-scale)
    beta            ::T    # Power law exponent for hydraulic transmissivity (opening/closing)
    f               ::T    # Darcy-Weisbach friction factor. Shared by K24's own conduit closure (folded into K = (2/pi)^(1/4)*sqrt((pi+2)/(rho_w*f)), which drives S_inf/H/N_inf) and DarcyWeisbachThickness's sheet-flow inversion (used directly) -- one parameter, not two, since both derive from the same Darcy-Weisbach physics (Schoof 2010/Clarke 1996)
    F_till          ::T    # Till compressibility/yield factor for soft-bed transition
    Q_c             ::T    # Threshold discharge for laminar-to-turbulent transition [m3/s]
    drainage_mode   ::M    # BothDrainage()/EfficientOnly()/InefficientOnly(): which opening terms update_N_inf! includes and which effective Q_c update_H! uses -- see AbstractDrainageMode
    H_0             ::T    # Thickness of canals for soft bed deformation [m]
    l_c             ::T    # Distance between conduits [m]
    K               ::T    # Conductivity coefficient in Darcy-Weisbach relation
    eta_w           ::T    # Dynamic viscosity of water [Pa s]
    W_min            ::T    # Minimum subglacial water layer thickness [m]; only applied by the sheet-flow water_thickness_algorithm closures (DarcyWeisbachThickness, LaminarThickness). Defaults to 0.0 (no floor) -- pass 1e-8 for KORI-ULB's own Wdmin if you want that bound back
    W_max            ::T    # Maximum subglacial water layer thickness [m]; only applied by the sheet-flow water_thickness_algorithm closures (DarcyWeisbachThickness, LaminarThickness). Defaults to Inf (no ceiling) -- pass 0.015 for KORI-ULB's own Wdmax if you want that bound back
    water_thickness_algorithm ::WT # ArealConduitThickness()/DarcyWeisbachThickness()/LaminarThickness(): which closure update_W! uses to compute state.W
    longcoupwater   ::T    # Longitudinal coupling factor for the stress-gradient coupling smoothing of the geometric potential gradients -- KORI-ULB's own internal parameter, unchanged in meaning. Set via the constructor's `coupling_length_kamb86` keyword (longcoupwater = coupling_length_kamb86 / 2), not directly -- see that keyword's docstring for why
    N_min           ::T    # Minimum allowed effective pressure N_inf [Pa]. Defaults to 0.0 (no floor); also the N_inf of a cell with flux but a flat potential (S_inf = Inf)
    q_min           ::T    # Minimum allowed value for the distributed water flux
    q_max           ::T    # Maximum allowed value for the distributed water flux. Defaults to Inf (no ceiling) -- pass perYear2perSecond(1e5) for KORI-ULB's own SubWaterFlux.m numerical-stability cap if you want that bound back
    fill_iters      ::Int  # How many iterations to perform for the filling of local minima of the geometric potential phi0 (JacobiFill only)
    fill_algorithm  ::FA                     # JacobiFill()/LowestNeighbourFill()/PriorityFloodFill(): how potential_filling! removes pits (dispatched once per update_q! call)
    routing_scheme  ::RS                     # Warner() (default), GDSWarner() (original K24) or another Le Brocq et al. (2006) scheme -- see AbstractRoutingScheme
    q_conversion    ::QC                     # QFromOutflow() or QFromFaceAverage() (default follows routing_scheme) -- see AbstractQConversion
    dissipation_discretization ::DD                                 # CellCentredDissipation() or FaceDissipation() (default follows routing_scheme) -- see AbstractDissipationDiscretization
    gradient_discretization ::GD                              # CentredGradient() or FaceGradient() (default follows routing_scheme): stencil of the gradient magnitudes -- see AbstractGradientDiscretization
    potential_weight ::PW                       # GroundedIce() (default) or GroundedMask(): which cells have a meaningful potential (model.phi_weight) for the smoothing, the true gradient and the face dissipation -- see AbstractPotentialWeight
    friction_discretization ::FD                              # CellCentredFriction() (default) or StaggeredFriction(ux, uy) -- see AbstractFrictionDiscretization
    max_psi_out_calls ::Int  # Safety cap on the number of accumulate_psi_out! calls in one update_psi_out! sweep, mirroring KORI-ULB's funcnt <= 5e4 cap in DpareaWarGds.m
    psi_out_algorithm ::P  # RecursivePsiOut() or IterativePsiOut(): which flow-routing implementation resolve_q! uses to compute psi_out each sweep
    max_dissipation_iters ::Int  # Safety cap on the number of Picard iterations for the dissipation melt term in update_q!
    dissipation_rtol       ::T    # Relative tolerance on q for the dissipation melt term's Picard iteration to be considered converged
    dissipation_melt        ::D    # DissipationMeltOn() or DissipationMeltOff(): whether update_q! includes the |q * grad(phi0)| / L_w term
    dissipation_verbose     ::Bool # Whether the dissipation melt term's Picard iteration logs its timing/convergence summary each call
    sensible_heat           ::S    # SensibleHeatOn() or SensibleHeatOff(): whether update_q! includes the ct*cw*rho_w*(q . grad(p_w)) / L_w term
    ct                      ::T    # Change of the pressure-melting point with pressure [K/Pa]
    cw                      ::T    # Heat capacity of water [J/kg/K]
    sliding_law             ::L    # AbstractSlidingLaw instance used to compute tau_b for the frictional-heating term tau_b*v_b in mdot
    max_qN_iters      ::Int  # Safety cap on the number of Picard iterations for the (q, N) loop (N-dependent sliding_law or sensible_heat on)
    qN_rtol           ::T    # Relative tolerance on q and N for the (q, N) Picard iteration to be considered converged
    qN_verbose        ::Bool # Whether the (q, N) coupling Picard iteration logs its timing/convergence summary each call

end


"""
$(TYPEDSIGNATURES)

Every array buffer `KazmierczakHydroModel` touches during a solve -- allocated once at construction
and updated in place by `update_q!`/`update_W!`/`update_N!` and their helpers. Split out from
`KazmierczakParams` (the immutable physical constants/config); see `KazmierczakHydroModel` for how
the split is made transparent to callers. Not `mutable` itself: nothing ever reassigns a field of
this struct, only the contents of the arrays it holds (`model.q .= ...`, never `model.q = ...`).

Buffers that only some configurations read are allocated empty (`0 x 0`) for the others, so the
model's memory follows what it actually solves: `q_prev` (only the Picard loops: dissipation melt, an `N`-dependent sliding law or sensible heat) and
`N_prev` (only the `(q, N)` loop); likewise the face-flux and routing-weight arrays of the
`routing_tape` (see `RoutingTape`). The reported terms (`Q_b`, `Q_diss`, `Q_sens`, `H_hard`, `H_soft`,
...) are always full size, since a host may read them.
"""
struct KazmierczakWorkspace{T, A, R <: RoutingTape{T}}

    # Geometric potential
    phi0                   ::A  # True geometric potential rho_i*g*h + rho_w*g*b [Pa]; used for N and for the local gradient magnitude
    phi0_filled            ::A  # phi0 with local minima filled; used only to route water (flow direction), never for N/S_inf/dissipation [Pa]
    phi0_tmp               ::A  # Temporary storage for potential filling of phi0 to smoothen local minima and avoid stuck water; also scratch for the weighted coupling smoothing
    minus_grad_phi0_x      ::A  # Geometric potential gradient x-component [Pa/m]
    minus_grad_phi0_y      ::A  # Geometric potential gradient y-component [Pa/m]
    abs_grad_phi0          ::A  # Magnitude of the geometric potential gradient [Pa/m]
    minus_grad_phi0_sx     ::A  # Smoothed gradient x-component of the geometric potential [Pa/m]
    minus_grad_phi0_sy     ::A  # Smoothed gradient y-component of the geometric potential [Pa/m]
    abs_grad_phi0_s        ::A  # Euclidean magnitude of the smoothed gradient of the geometric potential [Pa/m]
    phi_weight             ::A  # 1 where phi0 is a meaningful potential (per potential_weight), else 0; set by update_phi0!
    phi_weight_s           ::A  # phi_weight convolved with the coupling kernel (the normalisation of the smoothed potential)

    # Water flux
    h          ::A  # ice thickness after geometric potential filling serves as a temporary storage [m]
    G            ::A  # Geothermal heat flux into the bed [W/m2]
    q_T          ::A  # Conductive heat flux from the bed into the ice [W/m2]
    i_eb         ::A  # Water reaching the bed from above, not melted there (i_eb of Sommers et al 2018: drained englacial water, surface input) [kg/m2/s]
    Q_b          ::A  # Frictional heat tau_b . u_b [W/m2], from sliding_law and friction_discretization (0 with NoFrictionSlidingLaw)
    Q_diss       ::A  # Heat dissipated by the water flow |q . grad(phi0)| [W/m2] (0 when dissipation_melt is off)
    Q_sens       ::A  # Sensible heat ct*cw*rho_w*(q . grad(p_w)) [W/m2], signed (0 when sensible_heat is off)
    mdot_fixed   ::A  # Fixed part of the basal melt rate, (G - q_T)/L_w [kg/m2/s]; set by set_basal_terms!
    mdot_total   ::A  # Water source routed by the flux solver, mdot_fixed + (Q_b + Q_diss + Q_sens)/L_w + i_eb [kg/m2/s]
    psi_out    ::A  # Integrated scalar water flux [m3/s]
    corfac     ::A  # Correction factor to go from psi_out to q
    q          ::A  # Distributed water flux [m2/s]
    q_prev     ::A  # q from the previous Picard sweep, for the dissipation-melt/coupling convergence check
    tau_b      ::A  # Basal shear stress from model.sliding_law, set by update_tau_b!(model, state, sliding_law) [Pa]
    N_prev     ::A  # N from the previous Picard sweep, for the (q, N) coupling convergence check (only used when sliding_law is N-dependent)
    routing_tape ::R  # Recorded psi_out traversal replayed every Picard sweep by TapedPsiOut
    filled_cells ::Vector{Tuple{Int32, Int32}}  # Scratch list of the cells potential_filling! filled in its current pass
    fill_candidates ::Vector{Int32}  # Scratch list (linear indices) of the cells potential_filling! re-checks in its next pass
    fill_stamp      ::Matrix{Int32}  # Per-cell pass stamp used to dedupe fill_candidates
    fill_heap_p     ::Vector{T}      # PriorityFloodFill's min-heap (priorities), reused across calls
    fill_heap_k     ::Vector{Int32}  # PriorityFloodFill's min-heap (linear cell indices)
    coupling_kernel_cache ::Base.RefValue{Tuple{T, Matrix{T}}}  # (h_avg, kernel) of the last coupling_kernel call

    # Effective pressure and Bed state
    Q       ::A  # Volumetric water flux within a conduit [m3/s]
    kappa   ::A  # Bed type indicator (0: hard, 1: soft)
    abs_v_b ::A  # Magnitude of basal sliding velocity [m/s]
    A_visc  ::A  # Ice flow law rate factor (Glen's A) [Pa^-n s^-1]
    S_inf   ::A  # Far-field (away from grounding line) conduit cross-sectional area [m2]
    H_hard  ::A  # Thickness of conduits over a hard bed [m]
    H_soft  ::A  # Thickness of conduits over a soft bed [m]
    H       ::A  # Thickness of conduits [m]
    N_inf   ::A  # Far-field (away from grounding line) effective pressure [Pa]
    Po      ::A  # Ice overburden pressure (rho_i * g * ice_thickness) [Pa]

end


"""
$(TYPEDSIGNATURES)

The hydrology model described in Kazmierczak et al 2024 (https://doi.org/10.5194/tc-18-5887-2024).
dx != dy grids are supported: every step of the water-flux calculation (including the
stress-gradient coupling kernel in `update_smoothed_potential_gradients!`) works from dx and dy
separately rather than assuming square cells.

Composed of a `KazmierczakParams` (physical constants and solver configuration, immutable) and a
`KazmierczakWorkspace` (every array buffer the solve touches, allocated once and updated in place),
rather than one struct with the ~40 fields of both flattened together. This keeps the constructor's
positional argument list short and grouped by struct instead of one long tuple that silently breaks
if a field is inserted out of order.

`model.<field>` (e.g. `model.rho_w`, `model.q`) still resolves exactly as before the split:
`Base.getproperty` is overridden below to look up unrecognized field names on `params` then
`workspace`, so none of the physics files needed to
change. Use `model.params`/`model.workspace` to get the sub-structs themselves.
"""
struct KazmierczakHydroModel{T <: AbstractFloat, A, D <: AbstractDissipationMelt, L <: AbstractSlidingLaw, P <: AbstractPsiOutAlgorithm, WT <: AbstractWaterThicknessAlgorithm, M <: AbstractDrainageMode, S <: AbstractSensibleHeat,
                             PR <: KazmierczakParams{T, D, L, P, WT, M, S}} <: AbstractKazmierczakHydroModel
    params    ::PR   # concrete, so the remaining options (routing scheme, fill algorithm, ...) are type parameters too
    workspace ::KazmierczakWorkspace{T, A, RoutingTape{T}}
end

function Base.getproperty(model::KazmierczakHydroModel, name::Symbol)
    name === :params    && return getfield(model, :params)
    name === :workspace && return getfield(model, :workspace)
    params = getfield(model, :params)
    hasfield(typeof(params), name) && return getfield(params, name)
    return getfield(getfield(model, :workspace), name)
end

function Base.propertynames(model::KazmierczakHydroModel, private::Bool = false)
    return (fieldnames(typeof(getfield(model, :params)))..., fieldnames(typeof(getfield(model, :workspace)))..., :params, :workspace)
end


"""
$(TYPEDSIGNATURES)

The Kazmierczak et al 2024 hydrology model. The water source is built from individual terms (Eq. 3 of
Kazmierczak et al 2024), never supplied as a complete melt rate:

    mdot_fixed = (G - q_T) / L_w                     (set before the solve)
    mdot_total = mdot_fixed + (Q_b + Q_diss + Q_sens) / L_w + i_eb   (water source routed)

`mdot_fixed + (Q_b + Q_diss + Q_sens)/L_w` is the basal melt rate; `Q_b`, `Q_diss` and `Q_sens` depend
on the water flux (through N and q), so they are recomputed every Picard sweep.

`G` and `q_T` are given (positional, [W/m^2]); `Q_b = tau_b . u_b` comes from `sliding_law` and
`friction_discretization` (zero for `NoFrictionSlidingLaw`, the default); `Q_diss = |q . grad(phi0)|`
is added when `dissipation_melt` is on; `Q_sens = ct*cw*rho_w*(q . grad(p_w))`, with `p_w = Po - N`, is
added when `sensible_heat` is on (off by default, as in K24; see `AbstractSensibleHeat` in melt_sources.jl); `i_eb` [kg/m^2/s] is water reaching the bed from above that
was not melted there (drained englacial water, surface input), zero by default. As in Sommers et al
2018, `i_eb` is a source of water but not part of the melt rate. Each term is kept as its own field
(`model.G`, `model.q_T`, `model.Q_b`, `model.Q_diss`, `model.Q_sens`, `model.i_eb`; `model.mdot_fixed` holds only the fixed
part `(G - q_T)/L_w`), so a
coupled model can exchange terms rather than a melt rate. Update the given terms between solves with
[`set_basal_terms!`](@ref).

See the `AbstractSlidingLaw` docstring (sliding_law.jl) for the available laws and `resolve_q!` in
water_flux.jl for how N-dependent laws widen the existing dissipation-melt Picard loop into a joint
(q, N) fixed point.

Keyword names for the paper's symbols (the `KazmierczakParams` fields and `model.<field>` keep the paper symbol):
`glen_n` (n, Glen's flow law exponent), `flux_W_exponent` (alpha), `flux_grad_exponent` (beta),
`bed_bump_height` (h_b, typical bed obstacle height [m]), `H0_efficient` (H_0, canal thickness for soft bed
deformation [m]), `conduit_spacing` (l_c, distance between conduits [m]), `water_viscosity` (eta_w, dynamic
viscosity of water [Pa s]). The (q, N) Picard loop is controlled by `max_qN_iters`, `qN_rtol` and `qN_verbose`.
Unset, `qN_rtol` and `dissipation_rtol` are `1e-8` and `1e-12`, raised to `100 eps(T)` (about `1.2e-5`) when that is larger, i.e. for `Float32`, whose precision cannot reach the double-precision values (the loops would then run to their iteration caps every solve).

The `psi_out_algorithm` keyword (`TapedPsiOut()` by default) selects which flow-routing
implementation `resolve_q!` uses each sweep to compute psi_out -- see the `AbstractPsiOutAlgorithm`
docstring (psi_out.jl) for the `RecursivePsiOut`/`IterativePsiOut` speed-vs-stack-robustness trade-off.

The `water_thickness_algorithm` keyword (`DarcyWeisbachThickness()` by default) selects which closure
`update_W!` uses to compute the reportable `state.W` -- see the `AbstractWaterThicknessAlgorithm`
docstring (water_thickness.jl) for the three available closures (`ArealConduitThickness`, `DarcyWeisbachThickness`,
`LaminarThickness`) and how they differ.

`W_min`/`W_max`/`q_max`/`N_min` all default to no-op values (`0.0`/`Inf`/`Inf`/`0.0` respectively) --
i.e. nothing is clamped unless you ask for it. KORI-ULB's own values (`1e-8`/`0.015`/
`perYear2perSecond(1e5)`; and `0.02*Po`, which is cell-dependent so has no single `N_min` equivalent) are still available, just not silently applied: pass them explicitly if you
want that fidelity, or if you hit the numerical-stability edge cases they exist to guard against. Two
of the four are genuinely load-bearing for robustness, not just KORI-ULB fidelity, so removing them by
default is a real tradeoff, not a free one:
- `q_max = Inf` removes the cap Kori-ULB's own `SubWaterFlux.m` applies "for numerical stability"
  (per its own comment) -- without it, a sufficiently pathological cell (steep local gradient, small
  drainage area, high local melt) can drive `q` arbitrarily large, propagating into `S_inf`/`H`/`N_inf`.
- `N_min = 0.0` removes any floor on `N_inf`. A cell with flux but no potential gradient (a flat
  potential, e.g. an idealised flat bed) has an infinitely open conduit (`S_inf = H = Inf`), and its
  `N_inf` takes that limit, `N_min`, so with the default it gets `N = 0`. `N_inf == 0` is resolved to
  `N = 0` in `update_N!` (the `erf` argument would be `0/0` there), so neither case gives a `NaN`.

`W_min`/`W_max` are lower risk to leave unclamped: `state.W` is a terminal diagnostic field (see
`update_W!`'s docstring in water_thickness.jl) that feeds back into nothing else in the model, so an
unclamped value there can't itself destabilize the rest of a solve.

The constructor has no `longcoupwater` keyword directly -- pass `coupling_length_kamb86` instead.
`longcoupwater` itself (the field on `KazmierczakParams`, and the quantity `update_smoothed_potential_gradients!`
in potential.jl actually uses) keeps its original meaning unchanged, matching KORI-ULB's own internal
parameter of the same name; only the constructor's *public entry point* to it has moved. This is because
`longcoupwater` alone was never Kamb & Echelmeyer (1986)'s number directly: the kernel's true (2D
area-weighted) effective coupling length works out to `2 * longcoupwater * mean_ice_thickness`, not
`longcoupwater * mean_ice_thickness` -- see the derivation in `update_smoothed_potential_gradients!`
(potential.jl) and the field comment on `KazmierczakParams`. `coupling_length_kamb86` is defined so it
*is* that number directly: `longcoupwater = coupling_length_kamb86 / 2`, so effective coupling length =
`coupling_length_kamb86 * mean_ice_thickness` exactly, and the caller never has to work out the factor
of 2 themselves.

`coupling_length_kamb86` has no numerically safe no-op default (its correct value genuinely depends on
grid resolution relative to ice thickness), so leaving it unspecified emits a `@warn` and falls back to
`10.0` -- the upper edge of Kamb & Echelmeyer (1986)'s theoretical range for ice sheets. Their full set
of ranges:
- Ice sheets: ~4 to 10 ice thicknesses.
- Valley/mountain glaciers: a shorter ~1 to 3 ice thicknesses -- lateral drag against the valley walls
  transmits stress locally instead of over a long distance, so the coupling length is shorter than an
  unconfined ice sheet's.
- A glacier in surge: even longer, ~12 ice thicknesses.

Pick a value from whichever range matches your setting, or `0.0` to disable the smoothing entirely
(must be `>= 0`; a negative value throws an `ArgumentError`). See the `@warn` this constructor emits if
`coupling_length_kamb86` is left unspecified for how to tell, from your own grid resolution, when the
resulting coupling length is too small for your grid to resolve at all (in which case `0.0` is the
honest choice, not a value the grid can't represent).

`potential_weight` picks the cells whose potential the coupling smoothing, the true gradient
magnitude and the face dissipation may use: `GroundedIce()` (default) takes the `GROUNDED` cells and
the `OTHER_BASIN` cells with grounded ice, `GroundedMask()` the `GROUNDED` cells only (e.g. for a basin
cut whose outside ice is not evolved). Cells outside it, such as the `h = b = 0` padding around a rasterised
mesh, would otherwise pull the smoothed potential down near the edge (spurious divides, high-N hot
spots) and give the boundary cells a gradient of the whole overburden -- see `AbstractPotentialWeight`.

The boundary conditions come from the state's [`HydroMask`](@ref): water leaves only into `OCEAN` and
`LAND` cells, whose routing potential is that of the ocean or atmospheric pressure at the bed
(`set_outlet_potential!`); faces to `OTHER_BASIN` cells and the edge of the grid are closed.

Works with any concrete subtype of AbstractHydroGrid -- changing the grid does not require changing this constructor.

# Arguments

- `grid::AbstractHydroGrid`: grid of the simulation
- `kappa_in::AbstractArray{<:AbstractFloat}`: Bed type indicator (0: hard, 1: soft)
- `abs_v_b_in::AbstractArray{<:AbstractFloat}`: Magnitude of basal sliding velocity [m/s]
- `A_visc_in::AbstractArray{<:AbstractFloat}`: Ice flow law rate factor (Glen's A) [Pa^-n s^-1]
- `G_in::AbstractArray{<:AbstractFloat}`: geothermal heat flux into the bed [W/m^2]
- `q_T_in::AbstractArray{<:AbstractFloat}`: conductive heat flux from the bed into the ice [W/m^2]
- `i_eb` (keyword): water reaching the bed from above, not melted there [kg/m^2/s];
  `nothing` (default) means zero
"""
const KAZMIERCZAK_DEFAULT_L_W = 3.34e5 # Latent heat of fusion for ice [J/kg]

function KazmierczakHydroModel(
    grid::AbstractHydroGrid,
    kappa_in::AbstractArray{<:AbstractFloat},
    abs_v_b_in::AbstractArray{<:AbstractFloat},
    A_visc_in::AbstractArray{<:AbstractFloat},
    G_in::AbstractArray{<:AbstractFloat},
    q_T_in::AbstractArray{<:AbstractFloat};
    i_eb          = nothing,                      # Water reaching the bed from above, not melted there [kg/m2/s]; nothing = zero
    rho_w         = 1000.0,# Density of fresh water [kg/m3]
    rho_sw        = 1028.0,                       # Density of sea water [kg/m3] for the erf argument of N, so N = 0 at flotation (match the ice model's flotation density, 1028 for Yelmo). Pass rho_sw = rho_w to reproduce K24 Eq. 7 as written
    rho_i         = 917.0,                        # Density of ice [kg/m3]
    g             = 9.81,                         # Gravitational acceleration [m/s2]
    L_w           = KAZMIERCZAK_DEFAULT_L_W,       # Latent heat of fusion for ice [J/kg]
    glen_n        = 3.0,                          # n: Glen's flow law exponent (typically 3)
    bed_bump_height = 0.1,                        # h_b: typical bed obstacle height [m]
    flux_W_exponent = 5/4,                        # alpha: power law exponent for hydraulic transmissivity (m-scale)
    flux_grad_exponent = 3/2,                     # beta: power law exponent for hydraulic transmissivity (opening/closing)
    f             = 0.1,                          # Darcy-Weisbach friction factor -- shared between K (S_inf/H/N_inf) and DarcyWeisbachThickness, not a separate value for each; see KazmierczakParams' field comment
    F_till        = 1.1,                          # Till compressibility/yield factor for soft-bed transition
    Q_c           = 1.0,                          # Threshold discharge for laminar-to-turbulent transition [m3/s]
    drainage_mode = BothDrainage(),               # BothDrainage()/EfficientOnly()/InefficientOnly(): which opening terms update_N_inf! includes and which effective Q_c update_H! uses -- see AbstractDrainageMode
    H0_efficient  = 0.1,                          # H_0: thickness of canals for soft bed deformation [m]
    conduit_spacing = 10000.0,                    # l_c: distance between conduits [m]
    water_viscosity = perYear2perSecond(1.8e-3),  # eta_w: dynamic viscosity of water [Pa s] -- matches KORI-ULB's own par.waterviscosity, not literal SI water viscosity
    W_min          = 0.0,                          # Minimum subglacial water layer thickness [m]; no floor by default -- pass 1e-8 for KORI-ULB's own Wdmin
    W_max          = Inf,                          # Maximum subglacial water layer thickness [m]; no ceiling by default -- pass 0.015 for KORI-ULB's own Wdmax
    water_thickness_algorithm = DarcyWeisbachThickness(), # ArealConduitThickness()/DarcyWeisbachThickness()/LaminarThickness(): which closure update_W! uses to compute state.W
    coupling_length_kamb86 = nothing,             # Kamb & Echelmeyer (1986) stress-gradient-coupling length, as a direct multiple of mean ice thickness (>= 0; 0 disables the smoothing); no safe no-op default, so leaving this unspecified defaults to 10.0 and emits a @warn explaining how to choose it
    N_min         = 0.0,                          # Minimum effective pressure N_inf [Pa]; no floor by default
    q_min         = 0.0,                          # Minimum allowed value for the distributed water flux
    q_max         = Inf,                          # Maximum allowed value for the distributed water flux; no ceiling by default -- pass perYear2perSecond(1e5) for KORI-ULB's own SubWaterFlux.m numerical-stability cap
    fill_iters    = 10,                           # How many iterations to perform for the filling of local minima of the geometric potential phi0 (JacobiFill only)
    fill_algorithm = nothing,                     # JacobiFill()/LowestNeighbourFill()/PriorityFloodFill(); nothing = the routing scheme's partner (PriorityFloodFill for Warner) -- see AbstractFillAlgorithm/AbstractRoutingScheme
    routing_scheme = Warner(),                    # Flux-routing scheme (Le Brocq et al. 2006): Warner() (default)/GDSWarner() (original K24)/Quinn()/Tarboton()/ModifiedTarboton()/GDSTarboton() -- see AbstractRoutingScheme
    q_conversion = nothing,                       # QFromOutflow()/QFromFaceAverage(); nothing = QFromFaceAverage for Warner, QFromOutflow otherwise -- see AbstractQConversion
    dissipation_discretization = nothing,         # CellCentredDissipation()/FaceDissipation(); nothing = FaceDissipation for Warner, CellCentredDissipation otherwise -- see AbstractDissipationDiscretization
    gradient_discretization = nothing,            # CentredGradient()/FaceGradient(); nothing = FaceGradient for Warner, CentredGradient otherwise -- see AbstractGradientDiscretization
    potential_weight = GroundedIce(),             # GroundedIce()/GroundedMask(): cells with a meaningful potential, kept in the smoothing, true gradient and face dissipation -- GROUNDED and OTHER_BASIN cells with grounded ice, or GROUNDED only; see AbstractPotentialWeight
    friction_discretization = CellCentredFriction(),       # CellCentredFriction()/StaggeredFriction(ux_acx, uy_acy; quadrature): frictional heat from C-grid velocities -- see AbstractFrictionDiscretization
    max_psi_out_calls = 100_000,                   # Safety cap on the number of accumulate_psi_out! calls in one update_psi_out! sweep, mirroring KORI-ULB's funcnt <= 5e4 cap
    psi_out_algorithm = TapedPsiOut(),            # TapedPsiOut()/RecursivePsiOut()/IterativePsiOut()/TopologicalPsiOut(): which flow-routing implementation resolve_q! uses to compute psi_out
    max_dissipation_iters = 20,                   # Safety cap on the number of Picard iterations for the dissipation melt term in update_q!
    dissipation_rtol       = nothing,              # Relative tolerance on q for the dissipation melt term's Picard iteration to be considered converged; nothing = max(1e-12, 100 eps(T))
    dissipation_melt        = true,                # Whether update_q! includes the |q * grad(phi0)| / L_w term
    dissipation_verbose     = true,                # Whether the dissipation melt term's Picard iteration logs its timing/convergence summary each call
    sensible_heat = false,                        # Whether update_q! includes the sensible heat ct*cw*rho_w*(q . grad(p_w)) / L_w, p_w = Po - N (solved in the (q, N) loop)
    ct            = 7.5e-8,                       # Change of the pressure-melting point with pressure [K/Pa]
    cw            = 4.22e3,                       # Heat capacity of water [J/kg/K]
    sliding_law         = NoFrictionSlidingLaw(),          # AbstractSlidingLaw instance used to compute tau_b for the frictional-heating term tau_b*v_b in mdot
    max_qN_iters  = 20,                      # Safety cap on the number of Picard iterations for the (q, N) loop (N-dependent sliding_law or sensible_heat)
    qN_rtol       = nothing,                 # Relative tolerance on q and N for the (q, N) Picard iteration to be considered converged; nothing = max(1e-8, 100 eps(T))
    qN_verbose    = true,                    # Whether the (q, N) coupling Picard iteration logs its timing/convergence summary each call
)

    expected_size = (grid.Nx, grid.Ny)
    for (name, arr) in [("kappa", kappa_in), ("abs_v_b", abs_v_b_in), ("A_visc", A_visc_in), ("G", G_in), ("q_T", q_T_in)]
        size(arr)[1:2] == expected_size || throw(ArgumentError("$name size $(size(arr)) != grid size $expected_size"))
    end
    i_eb === nothing || size(i_eb)[1:2] == expected_size ||
        throw(ArgumentError("i_eb size $(size(i_eb)) != grid size $expected_size"))

    T = typeof(grid.dx)

    # Physical constants
    rho_w         = T(rho_w)
    rho_sw        = T(rho_sw)
    rho_i         = T(rho_i)
    g             = T(g)
    L_w           = T(L_w)
    n             = T(glen_n)
    h_b           = T(bed_bump_height)
    alpha         = T(flux_W_exponent)
    beta          = T(flux_grad_exponent)
    f             = T(f)
    F_till        = T(F_till)
    Q_c           = T(Q_c)
    H_0           = T(H0_efficient)
    l_c           = T(conduit_spacing)
    K             = (T(2)/T(pi))^(T(0.25)) * sqrt((T(pi) + T(2)) / (rho_w * f))
    eta_w         = T(water_viscosity)
    W_min          = T(W_min)
    W_max          = T(W_max)

    # coupling_length_kamb86 has no numerically-safe "no-op" default the way W_min/W_max/q_max/N_min do
    # (see this constructor's docstring): its correct value genuinely depends on grid resolution
    # relative to ice thickness, and silently picking a value the grid can't resolve produces different
    # (not obviously wrong) physics rather than an out-of-range number, so there's no way to make an
    # oblivious default "safe" the way clamping the others off does. Warn instead, but only if the
    # caller didn't actively choose a value themselves.
    if coupling_length_kamb86 === nothing
        coupling_length_kamb86 = 10.0
        @warn "coupling_length_kamb86 not specified, defaulting to $coupling_length_kamb86. This directly sets Kamb & Echelmeyer (1986)'s stress-gradient-coupling length as a multiple of mean grounded-ice thickness (update_smoothed_potential_gradients!): effective coupling length = coupling_length_kamb86 * mean_ice_thickness = $(coupling_length_kamb86)x ice thickness at this value -- the upper edge of Kamb & Echelmeyer (1986)'s theoretical range for ice sheets (~4-10x ice thickness; a shorter ~1-3x for valley/mountain glaciers instead, where lateral drag against the valley walls transmits stress locally rather than over a long distance, and an even longer ~12x for a glacier in surge). This is the kernel's 2D area-weighted effective width (the right quantity to compare against Kamb & Echelmeyer's number, since the smoothing kernel is a genuine 2D kernel, not a 1D profile); a naive 1D average over radius would understate it at ~2/3 of this. Choose it based on your grid resolution: if the resulting coupling length is smaller than your grid spacing (dx/dy), the smoothing can't be resolved and should be turned off (coupling_length_kamb86 = 0) rather than left at a value the grid can't represent -- e.g. at 16-32 km resolution with ~1500 m ice, even the ice-sheet range's coupling length (6-15 km) is already smaller than one grid cell. Pass coupling_length_kamb86 explicitly (0.0 to disable smoothing, ~4-10 for ice sheets, ~1-3 for valley/mountain glaciers, ~12 for a surging glacier, or your own estimate) to silence this warning."
    elseif coupling_length_kamb86 < 0
        throw(ArgumentError("coupling_length_kamb86 must be >= 0 (got $coupling_length_kamb86): it is Kamb & Echelmeyer (1986)'s stress-gradient-coupling length as a multiple of ice thickness, which is not a signed quantity -- 0 disables the smoothing entirely, it is not itself negative"))
    end
    # longcoupwater is KORI-ULB's own internal parameter (unchanged meaning, see the field comment on
    # KazmierczakParams); the 2D area-weighted effective coupling length works out to
    # 2 * longcoupwater * mean_ice_thickness (not 1x), so dividing by 2 here is what makes
    # coupling_length_kamb86 equal Kamb & Echelmeyer's ice-thickness multiple directly.
    longcoupwater = T(coupling_length_kamb86) / T(2)

    N_min         = T(N_min)
    q_min         = T(q_min)
    q_max         = T(q_max)
    fill_iters    = Int(fill_iters)
    # Unset options follow the routing scheme (see AbstractRoutingScheme): Warner gets its natural
    # face-based partners, the GDS schemes the original K24 choices.
    fill_algorithm === nothing &&
        (fill_algorithm = routing_scheme isa Union{GDSWarner, GDSTarboton} ? JacobiFill() : PriorityFloodFill())
    q_conversion === nothing &&
        (q_conversion = routing_scheme isa Warner ? QFromFaceAverage() : QFromOutflow())
    dissipation_discretization === nothing &&
        (dissipation_discretization = routing_scheme isa Warner ? FaceDissipation() : CellCentredDissipation())
    gradient_discretization === nothing &&
        (gradient_discretization = routing_scheme isa Warner ? FaceGradient() : CentredGradient())
    (routing_scheme isa GDSWarner || psi_out_algorithm isa TapedPsiOut) ||
        throw(ArgumentError("routing_scheme = $(routing_scheme) requires psi_out_algorithm = TapedPsiOut() (got $(psi_out_algorithm)); the other psi_out algorithms only implement the original GDSWarner routing"))
    if friction_discretization isa StaggeredFriction
        (size(friction_discretization.ux) == expected_size && size(friction_discretization.uy) == expected_size) ||
            throw(ArgumentError("StaggeredFriction ux/uy must be $(expected_size) (acx/acy fields on the model grid), got $(size(friction_discretization.ux)) and $(size(friction_discretization.uy))"))
    end
    if (q_conversion isa QFromFaceAverage || dissipation_discretization isa FaceDissipation) && n_directions(routing_scheme) != 4
        throw(ArgumentError("QFromFaceAverage/FaceDissipation need a 4-neighbour routing scheme (GDSWarner or Warner); $(routing_scheme) sends water diagonally, which does not cross cell faces"))
    end
    max_psi_out_calls = Int(max_psi_out_calls)
    max_dissipation_iters = Int(max_dissipation_iters)
    dissipation_rtol       = T(dissipation_rtol === nothing ? max(1e-12, 100 * eps(T)) : dissipation_rtol)
    dissipation_melt_trait = dissipation_melt ? DissipationMeltOn() : DissipationMeltOff()
    sensible_heat_trait    = sensible_heat ? SensibleHeatOn() : SensibleHeatOff()
    max_qN_iters  = Int(max_qN_iters)
    qN_rtol       = T(qN_rtol === nothing ? max(1e-8, 100 * eps(T)) : qN_rtol)
    sliding_law         = convert_sliding_law(T, sliding_law)
    friction_discretization = convert_friction_discretization(T, friction_discretization)

    # Geometric potential
    phi0          = alloc_field(grid)
    phi0_filled   = alloc_field(grid)
    phi0_tmp      = alloc_field(grid)
    minus_grad_phi0_x = alloc_field(grid)
    minus_grad_phi0_y = alloc_field(grid)
    abs_grad_phi0     = alloc_field(grid)
    minus_grad_phi0_sx = alloc_field(grid)
    minus_grad_phi0_sy = alloc_field(grid)
    abs_grad_phi0_s    = alloc_field(grid)
    phi_weight         = alloc_field(grid)
    phi_weight_s       = alloc_field(grid)

    # Buffers only some configurations read (see KazmierczakWorkspace) are empty for the others.
    empty_field = similar(phi0, 0, 0)
    qN_loop = uses_qN_loop(sliding_law, sensible_heat_trait)
    picard = dissipation_melt || qN_loop

    # Water flux
    h          = alloc_field(grid)
    G          = alloc_field(grid, G_in)
    q_T        = alloc_field(grid, q_T_in)
    i_eb = i_eb === nothing ? alloc_field(grid) : alloc_field(grid, i_eb)
    Q_b        = alloc_field(grid)
    Q_diss     = alloc_field(grid)
    Q_sens     = alloc_field(grid)
    mdot_fixed = alloc_field(grid)
    @. mdot_fixed = (G - q_T) / L_w
    mdot_total = alloc_field(grid)
    psi_out    = alloc_field(grid)
    corfac     = alloc_field(grid)
    q          = alloc_field(grid)
    q_prev     = picard ? alloc_field(grid) : empty_field
    tau_b      = alloc_field(grid)
    N_prev     = qN_loop ? alloc_field(grid) : similar(empty_field)

    # Effective pressure
    Q       = alloc_field(grid)
    kappa   = alloc_field(grid, kappa_in)
    abs_v_b = alloc_field(grid, abs_v_b_in)
    A_visc  = alloc_field(grid, A_visc_in)
    S_inf   = alloc_field(grid)
    H_hard  = alloc_field(grid)
    H_soft  = alloc_field(grid)
    H       = alloc_field(grid)
    N_inf   = alloc_field(grid)
    Po      = alloc_field(grid)

    # Routing scratch for this configuration (see RoutingTape): routing weights for every scheme but
    # GDSWarner (which allocates them on demand), face arrays only if a term reads them.
    face_fluxes = needs_face_fluxes(q_conversion, dissipation_melt_trait, sensible_heat_trait, dissipation_discretization)
    routing_tape = RoutingTape{T}(grid.Nx, grid.Ny;
        ndirs = routing_scheme isa GDSWarner && !face_fluxes ? 0 : n_directions(routing_scheme),
        face_fluxes = face_fluxes,
        face_dissipation = needs_face_dissipation(dissipation_melt_trait, sensible_heat_trait, dissipation_discretization),
        topological = psi_out_algorithm isa TopologicalPsiOut)
    # PriorityFloodFill's heap: sized once here, it keeps its capacity between solves.
    fill_heap_p, fill_heap_k = T[], Int32[]
    fill_algorithm isa PriorityFloodFill && (sizehint!(fill_heap_p, grid.Nx * grid.Ny ÷ 4); sizehint!(fill_heap_k, grid.Nx * grid.Ny ÷ 4))

    params = KazmierczakParams(
        rho_w, rho_sw, rho_i, g, L_w, n, h_b, alpha, beta, f, F_till, Q_c, drainage_mode, H_0, l_c, K, eta_w, W_min, W_max, water_thickness_algorithm, longcoupwater, N_min, q_min, q_max, fill_iters, fill_algorithm, routing_scheme, q_conversion, dissipation_discretization, gradient_discretization, potential_weight, friction_discretization,
        max_psi_out_calls, psi_out_algorithm, max_dissipation_iters, dissipation_rtol, dissipation_melt_trait, dissipation_verbose,
        sensible_heat_trait, T(ct), T(cw),
        sliding_law, max_qN_iters, qN_rtol, qN_verbose
)

    workspace = KazmierczakWorkspace(
        phi0, phi0_filled, phi0_tmp, minus_grad_phi0_x, minus_grad_phi0_y,
        abs_grad_phi0, minus_grad_phi0_sx, minus_grad_phi0_sy, abs_grad_phi0_s, phi_weight, phi_weight_s,
        h, G, q_T, i_eb, Q_b, Q_diss, Q_sens, mdot_fixed, mdot_total, psi_out, corfac, q, q_prev, tau_b, N_prev, routing_tape,
        Tuple{Int32, Int32}[], Int32[], zeros(Int32, grid.Nx, grid.Ny), fill_heap_p, fill_heap_k, Ref((T(NaN), zeros(T, 0, 0))),
        Q, kappa, abs_v_b, A_visc, S_inf, H_hard, H_soft, H, N_inf, Po
    )

    return KazmierczakHydroModel(params, workspace)

end
