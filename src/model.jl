"""
$(TYPEDSIGNATURES)

An abstract type for the hydrology model to be simulated. The model can hold revelant constants and model-specific fields.
"""
abstract type AbstractHydroModel end


#################################
# Model: Kazmierczak et al 2024 #
#################################


"""
$(TYPEDSIGNATURES)

Trait controlling whether `update_q!` includes the dissipation melt rate |q * grad(phi0)| / L_w in the water source
(see `KazmierczakHydroModel`'s `dissipation_melt` keyword). Stored as a type parameter so the on/off choice is
resolved by multiple dispatch at compile time -- `update_q!` calls a helper on `model.dissipation_melt` that has one
method for `DissipationMeltOn` and one for `DissipationMeltOff`, rather than branching on a `Bool` field at runtime.
"""
abstract type AbstractDissipationMelt end
struct DissipationMeltOn  <: AbstractDissipationMelt end
struct DissipationMeltOff <: AbstractDissipationMelt end


"""
$(TYPEDSIGNATURES)

Trait selecting which flow-routing implementation `resolve_q!` uses to compute psi_out each sweep
(see `KazmierczakHydroModel`'s `psi_out_algorithm` keyword). Stored as a type parameter, resolved by
multiple dispatch at compile time via `route_psi_out!` in water_flux.jl, the same pattern as
`AbstractDissipationMelt` above.

`RecursivePsiOut` (`update_psi_out!`/`accumulate_psi_out!`) and `IterativePsiOut`
(`update_psi_out_iterative!`) compute exactly the same field -- verified to match bit-for-bit,
`max_psi_out_calls` cutoff included, on synthetic grids and on real Thwaites/pan-Antarctica datasets
-- so switching between them changes performance and robustness, not results:
- `RecursivePsiOut` (the default, preserving prior behaviour) is substantially faster (~30-40x
  faster per sweep, benchmarked on Thwaites 2km and pan-Antarctica 8km) since Julia's native call
  stack has far less overhead than an explicit heap-allocated stack. But it recurses as deep as the
  longest flow-routing chain in the domain, which for real (non-toy) ice-sheet grids can exceed the
  calling process's `ulimit -s` and crash with a `StackOverflowError` -- independent of, and
  potentially before, `max_psi_out_calls` ever binds (that cap limits total cells visited per sweep,
  not recursion depth along any one chain).
- `IterativePsiOut` has no such requirement (its "stack" is a plain `Vector`, bounded only by heap
  memory), at the cost of that ~30-40x slowdown. Prefer it for large/convoluted domains, or whenever
  running under a constrained/unknown stack limit (CI, HPC job schedulers, etc.) where raising
  `ulimit -s` isn't an option.
"""
abstract type AbstractPsiOutAlgorithm end

"""
$(TYPEDSIGNATURES)

Route psi_out with the original recursive algorithm (`update_psi_out!`/`accumulate_psi_out!`;
`KazmierczakHydroModel`'s default `psi_out_algorithm`). Substantially faster than
[`IterativePsiOut`](@ref) (~30-40x per sweep, benchmarked on Thwaites 2km and pan-Antarctica 8km),
but recurses as deep as the domain's longest flow-routing chain -- see
[`AbstractPsiOutAlgorithm`](@ref)'s docstring for when that's a problem.
"""
struct RecursivePsiOut <: AbstractPsiOutAlgorithm end

"""
$(TYPEDSIGNATURES)

Route psi_out with a stack-based rewrite of the same algorithm (`update_psi_out_iterative!`),
verified to match [`RecursivePsiOut`](@ref) bit-for-bit. Its "stack" is a heap-allocated `Vector`
rather than the native call stack, so it has no recursion-depth limit -- at the cost of that
~30-40x slowdown. See [`AbstractPsiOutAlgorithm`](@ref)'s docstring for the full trade-off.
"""
struct IterativePsiOut <: AbstractPsiOutAlgorithm end


"""
$(TYPEDSIGNATURES)

Basal sliding law used to compute the frictional-heating term tau_b * v_b in the melt rate (Eq. 3,
Sec. 2.2.1 of Kazmierczak et al 2024): mdot = (G + tau_b*v_b - q_T) / L_w + mdot_w. `calc_tau_b`
(a plain scalar formula) and `update_tau_b!` (the field-broadcast version actually used in
`resolve_q!`; see sliding_law.jl for why they're separate) both dispatch on this type to turn
`model.abs_v_b` and the current effective pressure `state.N` into a basal shear stress tau_b [Pa].

Split into two branches:
- `NoSlidingLaw`/`WeertmanSlidingLaw` do not depend on N, so they contribute a source term that is
  either zero or a fixed offset computed once -- no new fixed point to resolve.
- `AbstractPressureDependentSlidingLaw` (`PowerPlasticSlidingLaw`, `RegularizedCoulombSlidingLaw`)
  scale with N, so tau_b now depends on N which itself depends on q which depends on mdot which
  depends on tau_b: a genuine (q, N) fixed point. `resolve_q!` dispatches on this hierarchy to
  decide whether the existing q-only Picard loop needs widening to also update N each sweep (see
  water_flux.jl for the loop and the reasoning).

The two N-dependent laws mirror the two families of sliding law implemented in Yelmo.jl
(`Yelmo.jl/src/dyn/basal_dragging.jl`, `beta_method` 1/2/4 and 3/5 respectively) so that, when this
model is eventually coupled to Yelmo via Kryonomos.jl, both sides can be configured with numerically
matching laws rather than independently-drifting formulas: `PowerPlasticSlidingLaw` matches
`_calc_beta_aa_power_plastic!` (Bueler & van Pelt 2015) and `RegularizedCoulombSlidingLaw` matches
`_calc_beta_aa_reg_coulomb!` (Joughin et al. 2019, GRL Eq. 2).
"""
abstract type AbstractSlidingLaw end
abstract type AbstractPressureDependentSlidingLaw <: AbstractSlidingLaw end

"""
$(TYPEDSIGNATURES)

No frictional-heating feedback: tau_b = 0 everywhere, so mdot is unaffected by sliding
(`KazmierczakHydroModel`'s default `sliding_law`). Preserves the model's original behaviour, where
`mdot_in` alone is assumed to already represent the full melt rate.
"""
struct NoSlidingLaw <: AbstractSlidingLaw end

"""
$(TYPEDSIGNATURES)

Weertman-type power sliding law: tau_b = C * |v_b|^q, independent of effective pressure N. Included
for comparison/testing and for domains where N-independent sliding is the intended approximation;
since it does not depend on N it does not introduce a (q, N) feedback and costs nothing beyond a
single elementwise evaluation.

# Fields
- `C::T`: sliding coefficient [Pa (s/m)^q]
- `q::T`: velocity exponent (dimensionless; classically 1/n with n Glen's law exponent, so ~1/3)
"""
struct WeertmanSlidingLaw{T <: AbstractFloat} <: AbstractSlidingLaw
    C ::T
    q ::T
end
WeertmanSlidingLaw(; C, q = 1/3) = WeertmanSlidingLaw(promote(float(C), float(q))...)

"""
$(TYPEDSIGNATURES)

Power-plastic sliding law (Bueler & van Pelt 2015): tau_b = c_till * N * (|v_b| / u0)^q. Matches
Yelmo.jl's `_calc_beta_aa_power_plastic!` (`beta_method` 1 when `q = 1`, 2 for general `q`).

# Fields
- `c_till::T`: till-strength coefficient (dimensionless, ~ tan of the till friction angle)
- `q::T`: velocity exponent (dimensionless; `q = 1` gives a linear-in-velocity law)
- `u0::T`: velocity scale [m/s]
"""
struct PowerPlasticSlidingLaw{T <: AbstractFloat} <: AbstractPressureDependentSlidingLaw
    c_till ::T
    q      ::T
    u0     ::T
end
PowerPlasticSlidingLaw(; c_till, q = 1.0, u0 = perYear2perSecond(100.0)) =
    PowerPlasticSlidingLaw(promote(float(c_till), float(q), float(u0))...)

"""
$(TYPEDSIGNATURES)

Regularized-Coulomb sliding law (Joughin et al. 2019, GRL Eq. 2): tau_b = c_till * N * (|v_b| /
(|v_b| + u0))^q. Saturates toward the Coulomb-friction limit c_till * N as |v_b| -> infinity;
behaves like a Weertman power law for |v_b| << u0. Matches Yelmo.jl's `_calc_beta_aa_reg_coulomb!`
(`beta_method` 3/5).

# Fields
- `c_till::T`: till-strength coefficient (dimensionless, ~ tan of the till friction angle)
- `q::T`: velocity exponent (dimensionless)
- `u0::T`: velocity scale [m/s]
"""
struct RegularizedCoulombSlidingLaw{T <: AbstractFloat} <: AbstractPressureDependentSlidingLaw
    c_till ::T
    q      ::T
    u0     ::T
end
RegularizedCoulombSlidingLaw(; c_till, q = 1/3, u0 = perYear2perSecond(100.0)) =
    RegularizedCoulombSlidingLaw(promote(float(c_till), float(q), float(u0))...)

"""
$(TYPEDSIGNATURES)

Convert a sliding law's parameters to float type `T`, mirroring the explicit `T(...)` conversions
`KazmierczakHydroModel`'s constructor applies to its own scalar parameters -- keeps `model.sliding_law`
type-stable with the rest of the model when `T` is not `Float64` (e.g. `Float32` grids).
"""
convert_sliding_law(::Type{T}, law::NoSlidingLaw) where {T <: AbstractFloat} = law
convert_sliding_law(::Type{T}, law::WeertmanSlidingLaw) where {T <: AbstractFloat} =
    WeertmanSlidingLaw(C = T(law.C), q = T(law.q))
convert_sliding_law(::Type{T}, law::PowerPlasticSlidingLaw) where {T <: AbstractFloat} =
    PowerPlasticSlidingLaw(c_till = T(law.c_till), q = T(law.q), u0 = T(law.u0))
convert_sliding_law(::Type{T}, law::RegularizedCoulombSlidingLaw) where {T <: AbstractFloat} =
    RegularizedCoulombSlidingLaw(c_till = T(law.c_till), q = T(law.q), u0 = T(law.u0))


"""
$(TYPEDSIGNATURES)

Physical constants and solver configuration for `KazmierczakHydroModel` -- everything that is fixed
once the model is constructed and never touched again during a solve. Split out from
`KazmierczakWorkspace` (the mutable array buffers) so the two can be reasoned about and constructed
independently instead of interleaved as 40 flat fields on one struct; see `KazmierczakHydroModel`
for how the split is made transparent to callers.
"""
struct KazmierczakParams{T <: AbstractFloat, D <: AbstractDissipationMelt, L <: AbstractSlidingLaw, P <: AbstractPsiOutAlgorithm}

    rho_w           ::T    # Density of fresh water [kg/m3]
    rho_i           ::T    # Density of ice [kg/m3]
    g               ::T    # Gravitational acceleration [m/s2]
    L_w             ::T    # Latent heat of fusion for ice [J/kg]
    n               ::T    # Glen's flow law exponent (typically 3)
    h_b             ::T    # Typical bed obstacle height [m]
    alpha           ::T    # Power law exponent for hydraulic transmissivity (m-scale)
    beta            ::T    # Power law exponent for hydraulic transmissivity (opening/closing)
    f               ::T    # Darcy-Weisbach friction factor
    F_till          ::T    # Till compressibility/yield factor for soft-bed transition
    Q_c             ::T    # Threshold discharge for laminar-to-turbulent transition [m3/s]
    H_0             ::T    # Thickness of canals for soft bed deformation [m]
    l_c             ::T    # Distance between conduits [m]
    K               ::T    # Conductivity coefficient in Darcy-Weisbach relation
    eta_w           ::T    # Dynamic viscosity of water [Pa s]
    Wmin            ::T    # Minimum subglacial water layer thickness [m]
    Wmax            ::T    # Maximum subglacial water layer thickness [m]
    longcoupwater   ::T    # Longitudinal coupling factor for the stress-gradient coupling smoothing of the geometric potential gradients
    sigmat          ::T    # Effective pressure lower bound as fraction of overburden pressure
    q_min           ::T    # Minimum allowed value for the distributed water flux
    q_max           ::T    # Maximum allowed value for the distributed water flux, default to perYear2perSecond(1e5) following KORI-ULB
    fill_iters      ::Int  # How many iterations to perform for the filling of local minima of the geometric potential phi0
    max_psi_out_calls ::Int  # Safety cap on the number of accumulate_psi_out! calls in one update_psi_out! sweep, mirroring KORI-ULB's funcnt <= 5e4 cap in DpareaWarGds.m
    psi_out_algorithm ::P  # RecursivePsiOut() or IterativePsiOut(): which flow-routing implementation resolve_q! uses to compute psi_out each sweep
    max_dissipation_iters ::Int  # Safety cap on the number of Picard iterations for the dissipation melt term in update_q!
    dissipation_rtol       ::T    # Relative tolerance on q for the dissipation melt term's Picard iteration to be considered converged
    dissipation_melt        ::D    # DissipationMeltOn() or DissipationMeltOff(): whether update_q! includes the |q * grad(phi0)| / L_w term
    dissipation_verbose     ::Bool # Whether the dissipation melt term's Picard iteration logs its timing/convergence summary each call
    sliding_law             ::L    # AbstractSlidingLaw instance used to compute tau_b for the frictional-heating term tau_b*v_b in mdot
    max_coupling_iters      ::Int  # Safety cap on the number of Picard iterations for the (q, N) loop when sliding_law is N-dependent
    coupling_rtol           ::T    # Relative tolerance on q and N for the (q, N) Picard iteration to be considered converged
    coupling_verbose        ::Bool # Whether the (q, N) coupling Picard iteration logs its timing/convergence summary each call

end


"""
$(TYPEDSIGNATURES)

Every array buffer `KazmierczakHydroModel` touches during a solve -- allocated once at construction
and updated in place by `update_q!`/`update_W!`/`update_N!` and their helpers. Split out from
`KazmierczakParams` (the immutable physical constants/config); see `KazmierczakHydroModel` for how
the split is made transparent to callers. Not `mutable` itself: nothing ever reassigns a field of
this struct, only the contents of the arrays it holds (`model.q .= ...`, never `model.q = ...`).
"""
struct KazmierczakWorkspace{A}

    # Geometric potential
    phi0                   ::A  # Geometric potential [Pa]
    phi0_tmp               ::A  # Temporary storage for potential filling of phi0 to smoothen local minima and avoid stuck water
    minus_grad_phi0_x      ::A  # Geometric potential gradient x-component [Pa/m]
    minus_grad_phi0_y      ::A  # Geometric potential gradient y-component [Pa/m]
    abs_grad_phi0          ::A  # Magnitude of the geometric potential gradient [Pa/m]
    minus_grad_phi0_sx     ::A  # Smoothed gradient x-component of the geometric potential [Pa/m]
    minus_grad_phi0_sy     ::A  # Smoothed gradient y-component of the geometric potential [Pa/m]
    abs_grad_phi0_s        ::A  # Magnitude of the smoothed gradient of the geometric potential [Pa/m]

    # Water flux
    visited    ::A  # visited cells during the recursive algorithm to calculate psi_out
    h          ::A  # ice thickness after geometric potential filling serves as a temporary storage [m]
    mdot       ::A  # mass basal melt rate per unit area, background (G - q_T)/L_w term supplied by the caller [Kg / m^2 / s]
    mdot_total ::A  # mdot plus the dissipation melt term and/or the frictional-heating term tau_b*v_b/L_w, whichever are active [Kg / m^2 / s]
    psi_out    ::A  # Integrated scalar water flux [m3/s]
    corfac     ::A  # Correction factor to go from psi_out to q
    q          ::A  # Distributed water flux [m2/s]
    q_prev     ::A  # q from the previous Picard sweep, for the dissipation-melt/coupling convergence check
    tau_b      ::A  # Basal shear stress from model.sliding_law, set by update_tau_b!(model, state, sliding_law) [Pa]
    N_prev     ::A  # N from the previous Picard sweep, for the (q, N) coupling convergence check (only used when sliding_law is N-dependent)

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
`workspace`, so nothing in water_flux.jl/effective_pressure.jl/sliding_law.jl/run.jl needed to
change. Use `model.params`/`model.workspace` to get the sub-structs themselves.
"""
struct KazmierczakHydroModel{T <: AbstractFloat, A, D <: AbstractDissipationMelt, L <: AbstractSlidingLaw, P <: AbstractPsiOutAlgorithm} <: AbstractHydroModel
    params    ::KazmierczakParams{T, D, L, P}
    workspace ::KazmierczakWorkspace{A}
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

The "external mdot" constructor to the Kazmierczak et al 2024 hydrology model: `mdot_in` is taken as a
complete, already-converged basal melt rate -- e.g. straight from another model's own output, such as
`load_Kazmierczak`'s/`load_yelmox`'s `ṁ` (KORI-ULB's `Bmelt` or Yelmo's `bmb`), which already bake in
that source model's own frictional-heating (and possibly dissipation) physics.

This constructor is meant to be paired with `sliding_law = NoSlidingLaw()` (the default): it implicitly
assumes the melt rate's dependence on the effective pressure N is weak enough to treat as fixed, exogenous
forcing -- decoupled from N, no Picard iteration needed. If you instead pass a real `sliding_law`
(`WeertmanSlidingLaw`, `PowerPlasticSlidingLaw`, `RegularizedCoulombSlidingLaw`), its `tau_b*v_b/L_w`
frictional-heating term (Eq. 3 of Kazmierczak et al 2024) is added on top of `mdot_in` dynamically each
sweep -- it is then your responsibility to ensure `mdot_in` doesn't already include a friction estimate of
its own, or you will double-count it. Similarly, `dissipation_melt = true` (the default) adds
`|q*grad(phi0)|/L_w` on top; if your `mdot_in` source itself already includes a flow-driven dissipation
term (as Shakti.jl's own `mdot` does, for example), pass `dissipation_melt = false` to avoid double-counting
that too.

If you want FastHydrology to own the melt-rate physics end-to-end and compute `mdot` faithfully from Eq. 3
itself, use the other constructor method (`G_in`, `q_T_in`) instead.

See the `AbstractSlidingLaw` docstring in model.jl for the available laws and `resolve_q!` in
water_flux.jl for how N-dependent laws widen the existing dissipation-melt Picard loop into a joint
(q, N) fixed point.

The `psi_out_algorithm` keyword (`RecursivePsiOut()` by default) selects which flow-routing
implementation `resolve_q!` uses each sweep to compute psi_out -- see the `AbstractPsiOutAlgorithm`
docstring above for the `RecursivePsiOut`/`IterativePsiOut` speed-vs-stack-robustness trade-off.

Works with any concrete subtype of AbstractHydroGrid -- changing the grid does not require changing this constructor.

# Arguments

- `grid::AbstractHydroGrid`: grid of the simulation
- `kappa_in::AbstractArray{<:AbstractFloat}`: Bed type indicator (0: hard, 1: soft)
- `abs_v_b_in::AbstractArray{<:AbstractFloat}`: Magnitude of basal sliding velocity [m/s]
- `A_visc_in::AbstractArray{<:AbstractFloat}`: Ice flow law rate factor (Glen's A) [Pa^-n s^-1]
- `mdot_in::AbstractArray{<:AbstractFloat}`: complete mass basal melt rate per unit area [Kg / m^2 / s]
"""
function KazmierczakHydroModel(
    grid::AbstractHydroGrid,
    kappa_in::AbstractArray{<:AbstractFloat},
    abs_v_b_in::AbstractArray{<:AbstractFloat},
    A_visc_in::AbstractArray{<:AbstractFloat},
    mdot_in::AbstractArray{<:AbstractFloat};
    rho_w         = 1000.0,
    rho_i         = 917.0,
    g             = 9.81,
    L_w           = 3.34e5,
    n             = 3.0,
    h_b           = 0.1,
    alpha         = 5/4,
    beta          = 3/2,
    f             = 0.1,
    F_till        = 1.1,
    Q_c           = 1.0,
    H_0           = 0.1,
    l_c           = 10000.0,
    eta_w         = perYear2perSecond(1.8e-3),
    Wmin          = 1e-8,
    Wmax          = 0.015,
    longcoupwater = 5.0,
    sigmat        = 0.02,
    q_min         = 0.0,
    q_max         = perYear2perSecond(1e5),
    fill_iters    = 10,
    max_psi_out_calls = 50_000,
    psi_out_algorithm = RecursivePsiOut(),
    max_dissipation_iters = 20,
    dissipation_rtol       = 1e-12,
    dissipation_melt        = true,
    dissipation_verbose     = true,
    sliding_law         = NoSlidingLaw(),
    max_coupling_iters  = 20,
    coupling_rtol       = 1e-8,
    coupling_verbose    = true
)

    expected_size = (grid.Nx, grid.Ny)
    for (name, arr) in [("kappa", kappa_in), ("abs_v_b", abs_v_b_in), ("A_visc", A_visc_in), ("mdot_in", mdot_in)]
        size(arr)[1:2] == expected_size || throw(ArgumentError("$name size $(size(arr)) != grid size $expected_size"))
    end

    T = typeof(grid.dx)

    # Physical constants
    rho_w         = T(rho_w)
    rho_i         = T(rho_i)
    g             = T(g)
    L_w           = T(L_w)
    n             = T(n)
    h_b           = T(h_b)
    alpha         = T(alpha)
    beta          = T(beta)
    f             = T(f)
    F_till        = T(F_till)
    Q_c           = T(Q_c)
    H_0           = T(H_0)
    l_c           = T(l_c)
    K             = (T(2)/T(pi))^(T(0.25)) * sqrt((T(pi) + T(2)) / (rho_w * f))
    eta_w         = T(eta_w)
    Wmin          = T(Wmin)
    Wmax          = T(Wmax)
    longcoupwater = T(longcoupwater)
    sigmat        = T(sigmat)
    q_min         = T(q_min)
    q_max         = T(q_max)
    fill_iters    = Int(fill_iters)
    max_psi_out_calls = Int(max_psi_out_calls)
    max_dissipation_iters = Int(max_dissipation_iters)
    dissipation_rtol       = T(dissipation_rtol)
    dissipation_melt_trait = dissipation_melt ? DissipationMeltOn() : DissipationMeltOff()
    max_coupling_iters  = Int(max_coupling_iters)
    coupling_rtol       = T(coupling_rtol)
    sliding_law         = convert_sliding_law(T, sliding_law)

    # Geometric potential
    phi0          = alloc_field(grid)
    phi0_tmp      = alloc_field(grid)
    minus_grad_phi0_x = alloc_field(grid)
    minus_grad_phi0_y = alloc_field(grid)
    abs_grad_phi0     = alloc_field(grid)
    minus_grad_phi0_sx = alloc_field(grid)
    minus_grad_phi0_sy = alloc_field(grid)
    abs_grad_phi0_s    = alloc_field(grid)

    # Water flux
    visited    = alloc_field(grid)
    h          = alloc_field(grid)
    mdot       = alloc_field(grid, mdot_in)
    mdot_total = alloc_field(grid)
    psi_out    = alloc_field(grid)
    corfac     = alloc_field(grid)
    q          = alloc_field(grid)
    q_prev     = alloc_field(grid)
    tau_b      = alloc_field(grid)
    N_prev     = alloc_field(grid)

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

    params = KazmierczakParams(
        rho_w, rho_i, g, L_w, n, h_b, alpha, beta, f, F_till, Q_c, H_0, l_c, K, eta_w, Wmin, Wmax, longcoupwater, sigmat, q_min, q_max, fill_iters,
        max_psi_out_calls, psi_out_algorithm, max_dissipation_iters, dissipation_rtol, dissipation_melt_trait, dissipation_verbose,
        sliding_law, max_coupling_iters, coupling_rtol, coupling_verbose
    )

    workspace = KazmierczakWorkspace(
        phi0, phi0_tmp, minus_grad_phi0_x, minus_grad_phi0_y,
        abs_grad_phi0, minus_grad_phi0_sx, minus_grad_phi0_sy, abs_grad_phi0_s,
        visited, h, mdot, mdot_total, psi_out, corfac, q, q_prev, tau_b, N_prev,
        Q, kappa, abs_v_b, A_visc, S_inf, H_hard, H_soft, H, N_inf, Po
    )

    return KazmierczakHydroModel(params, workspace)

end

"""
$(TYPEDSIGNATURES)

The "faithful Eq. 3" constructor to the Kazmierczak et al 2024 hydrology model: rather than accepting a
complete melt rate, it takes the geothermal heat flux `G_in` and the conductive heat flux into the ice at
the bed `q_T_in` (both [W/m^2]) and computes the background melt rate `mdot = (G_in - q_T_in) / L_w`
itself -- exactly the `(G - q_T)/L_w` background term of Eq. 3 of Kazmierczak et al 2024. The
frictional-heating term `tau_b*v_b/L_w` (from `sliding_law`, `NoSlidingLaw()` by default) and the
flow-dissipation term `|q*grad(phi0)|/L_w` (`dissipation_melt = true` by default) are then added on top
dynamically during the simulation, same as for the other constructor -- but here they can never
double-count anything already baked into `mdot`, since `mdot` is built from nothing but `G_in`/`q_T_in`.

Both `G_in` and `q_T_in` are mandatory (no default): Eq. 3 needs both terms to be well posed, and silently
defaulting `q_T_in` to zero would hide the temperate-bed assumption that implies. If your data source has
no `q_T` field of its own (e.g. `load_Kazmierczak`), pass an explicit zero field so that assumption is
visible at the call site.

If instead you already have a complete, externally-computed melt rate (e.g. straight from another model's
own output, such as `load_Kazmierczak`'s/`load_yelmox`'s `ṁ`), use the other constructor method (`mdot_in`)
instead.

See the docstring on the `mdot_in` constructor above for the rest of the keyword arguments -- they are
identical here.

# Arguments

- `grid::AbstractHydroGrid`: grid of the simulation
- `kappa_in::AbstractArray{<:AbstractFloat}`: Bed type indicator (0: hard, 1: soft)
- `abs_v_b_in::AbstractArray{<:AbstractFloat}`: Magnitude of basal sliding velocity [m/s]
- `A_visc_in::AbstractArray{<:AbstractFloat}`: Ice flow law rate factor (Glen's A) [Pa^-n s^-1]
- `G_in::AbstractArray{<:AbstractFloat}`: geothermal heat flux [W/m^2]
- `q_T_in::AbstractArray{<:AbstractFloat}`: conductive heat flux into the ice at the bed [W/m^2]
"""
function KazmierczakHydroModel(
    grid::AbstractHydroGrid,
    kappa_in::AbstractArray{<:AbstractFloat},
    abs_v_b_in::AbstractArray{<:AbstractFloat},
    A_visc_in::AbstractArray{<:AbstractFloat},
    G_in::AbstractArray{<:AbstractFloat},
    q_T_in::AbstractArray{<:AbstractFloat};
    L_w = 3.34e5,
    kwargs...
)
    mdot_in = (G_in .- q_T_in) ./ L_w
    return KazmierczakHydroModel(grid, kappa_in, abs_v_b_in, A_visc_in, mdot_in; L_w = L_w, kwargs...)
end


####################################
# Model: Height above buoyancy (HAB) #
####################################


"""
$(TYPEDSIGNATURES)

The height above buoyancy (HAB) hydrology model described in Sec. 2.1.1 of Kazmierczak et al 2022 (https://doi.org/10.5194/tc-16-4537-2022).
"""
mutable struct HABHydroModel{T <: AbstractFloat, A} <: AbstractHydroModel

    # Model constants
    rho_sw ::T  # Density of sea water [kg/m3]
    rho_i  ::T  # Density of ice [kg/m3]
    g      ::T  # Gravitational acceleration [m/s2]
    P_w    ::T  # Water pressure coefficient; see below Eq. (3) of Kazmierczak et al 2022
    sigmat ::T  # Effective pressure lower bound as fraction of overburden pressure

    # Effective pressure
    Po  ::A  # Ice overburden pressure (rho_i * g * ice_thickness) [Pa]
    p_w ::A  # Water pressure [Pa]

end


"""
$(TYPEDSIGNATURES)

The constructor to the HAB hydrology model.

Works with any concrete subtype of AbstractHydroGrid -- changing the grid does not require changing this constructor.

# Arguments

- `grid::AbstractHydroGrid`: grid of the simulation
"""
function HABHydroModel(grid::AbstractHydroGrid)

    T = typeof(grid.dx)

    # Physical constants
    rho_sw = T(1027.0)
    rho_i  = T(917.0)
    g      = T(9.81)
    P_w    = T(0.96)
    sigmat = T(0.02)

    # Effective pressure fields
    Po  = alloc_field(grid)
    p_w = alloc_field(grid)

    return HABHydroModel(rho_sw, rho_i, g, P_w, sigmat, Po, p_w)

end


#################################
# Model: Shakti                 #
#################################


"""
$(TYPEDSIGNATURES)

Wraps a `Shakti.Simulation` (see the separate `Shakti.jl` package) so it can be run as an
`AbstractHydroModel`'s time evolution via `TimeSimulation`. Shakti's `Simulation` already owns its
own grid, state, and model parameters, so this wrapper holds nothing else -- FastHydrology's
`run!(::TimeSimulation)` for this model just delegates straight to `Shakti.run!`.

Defined unconditionally (no dependency on `Shakti` itself, since the field is generic over `S`), but
only usable once `Shakti` is loaded alongside `FastHydrology`: the `run!` method that dispatches on
this type is added by the `FastHydrologyShaktiExt` package extension (see `ext/FastHydrologyShaktiExt.jl`),
which Julia loads automatically once both packages are in scope (`using FastHydrology, Shakti`).

# Arguments

- `sim`: a `Shakti.Simulation` built the usual way (`Shakti.Simulation(grid, state, tsteps, dt, p, ...)`).
"""
struct ShaktiHydroModel{S} <: AbstractHydroModel
    sim::S
end
