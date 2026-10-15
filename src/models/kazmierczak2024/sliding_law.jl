#################################
# Model: Kazmierczak et al 2024 #
#################################

# Sliding laws: the basal shear stress tau_b that feeds the frictional heat (see melt_sources.jl).


"""
$(TYPEDSIGNATURES)

Basal sliding law used to compute the frictional-heating term tau_b * v_b in the melt rate (Eq. 3,
Sec. 2.2.1 of Kazmierczak et al 2024): mdot = (G + tau_b*v_b - q_T) / L_w + mdot_w. `calc_tau_b`
(a plain scalar formula) and `update_tau_b!` (the field-broadcast version actually used in
`resolve_q!`; see sliding_law.jl for why they're separate) both dispatch on this type to turn
`model.abs_v_b` and the current effective pressure `state.N` into a basal shear stress tau_b [Pa].

Split into two branches:
- `NoFrictionSlidingLaw`/`WeertmanSlidingLaw` do not depend on N, so they contribute a source
  term that is either zero or a fixed offset computed once -- no new fixed point to resolve.
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

No basal friction: tau_b = 0 everywhere, so the frictional-heating term Q_b of the melt rate is zero
(`KazmierczakHydroModel`'s default `sliding_law`). Use it when there is no sliding or when frictional
heating is deliberately left out; for friction, pass a law that computes tau_b
(`PrescribedFieldSlidingLaw` for a given tau_b field, or one of the sliding laws).
"""
struct NoFrictionSlidingLaw <: AbstractSlidingLaw end

"""
$(TYPEDSIGNATURES)

Sliding law holding a fixed, externally-prescribed `tau_b` field -- mirrors Shakti.jl's own
`PrescribedSlidingLaw` (`taub_x`/`taub_y` set once from real data, e.g. Yelmo's own
`taub_acx`/`taub_acy`, and never recomputed from `N`/`v_b`). Use this for parity with a Shakti.jl run
that used its own `PrescribedSlidingLaw` (e.g. Shakti.jl's Greenland v3 production run, which holds
`taub` fixed at Yelmo's own real directional basal shear stress rather than deriving it from a
regularized-Coulomb law -- see that script's own header note on why: a real `taub` field, when
available, is a better-grounded choice than guessing a Coulomb coefficient).

# Notes

`tau_b` here is a SCALAR magnitude field (e.g. `sqrt(taub_acx^2 + taub_acy^2)`, cell-centered) --
unlike Shakti.jl's own directional `taub_x`/`taub_y` (used in a dot product with the directional
`ub_x`/`ub_y`), this model's frictional-heating term is `tau_b*v_b` with `v_b = abs_v_b` already a
scalar magnitude, so directional information can't be carried through here. Passing `|taub|` is the
direct scalar analogue, and reproduces Shakti.jl's own term exactly whenever `taub` and `v_b` are
co-directional -- the case whenever both derive from the same underlying sliding physics, as for a
real ice-sheet-model field like Yelmo's.

Not a subtype of `AbstractPressureDependentSlidingLaw`: `tau_b` never depends on `N` here, so no
`(q, N)` coupling loop is needed -- matching Shakti.jl's own `PrescribedSlidingLaw`, chosen there for
exactly this reason (cheaper, and there's no N-dependent sliding physics left to approximate once a
real `taub` field is already available).

# Fields
- `tau_b::A`: fixed per-cell basal shear stress magnitude [Pa], wrapped via `alloc_field(grid, ...)`
  at construction (same idiom `KazmierczakHydroModel`'s own constructor uses for
  `kappa`/`abs_v_b`/`A_visc`) so it has the grid's own field type
"""
struct PrescribedFieldSlidingLaw{A} <: AbstractSlidingLaw
    tau_b::A
end
"""
$(TYPEDSIGNATURES)

Builds a [`PrescribedFieldSlidingLaw`](@ref) on grid `g` from a per-cell `tau_b` field, wrapping it
via `alloc_field(g, tau_b)`.
"""
PrescribedFieldSlidingLaw(g::AbstractHydroGrid, tau_b) = PrescribedFieldSlidingLaw(alloc_field(g, float.(tau_b)))

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

Regularized-Coulomb sliding law (Joughin et al. 2019, GRL Eq. 2), same formula as
[`RegularizedCoulombSlidingLaw`](@ref) but with a per-cell `c_till` FIELD rather than a uniform
scalar -- e.g. an ice-sheet model's own bed-friction-coefficient field, such as Yelmo's `cb_ref`
(bundled in its own restart output: `c_bed = cb_ref * N_eff` is Yelmo's own formula, see
`basal_dragging.f90`'s `calc_c_bed`). Using the real per-cell coefficient here, with `N` still the
*hydrology model's own* live effective pressure (not Yelmo's static `N_eff`), gives a genuinely
N-coupled sliding law parameterized by real data instead of one uniform guessed constant. Mirrors
Shakti.jl's own `RegularizedCoulombV0SlidingLaw` (same field-`C`-plus-fixed-`u0` structure).

# Fields
- `c_till::A`: per-cell till-strength coefficient field, wrapped via `alloc_field(grid, ...)` at
  construction (same units/role as the scalar version's `c_till`), so it has the grid's own field
  type and mixes cleanly with the model's other fields in `update_tau_b!`'s `@.` expression (see
  `sliding_law.jl`).
- `q::F`: velocity exponent (dimensionless)
- `u0::F`: velocity scale [m/s]
"""
struct RegularizedCoulombFieldSlidingLaw{A, F <: AbstractFloat} <: AbstractPressureDependentSlidingLaw
    c_till ::A
    q      ::F
    u0     ::F
end
"""
$(TYPEDSIGNATURES)

Builds a [`RegularizedCoulombFieldSlidingLaw`](@ref) on grid `g` from a per-cell `c_till` field,
wrapping it via `alloc_field(g, c_till)` (same idiom `KazmierczakHydroModel`'s own constructor uses
for `kappa`/`abs_v_b`/`A_visc`) so it has the grid's own field type.
"""
RegularizedCoulombFieldSlidingLaw(g::AbstractHydroGrid, c_till; q = 1/3, u0 = perYear2perSecond(100.0)) =
    RegularizedCoulombFieldSlidingLaw(alloc_field(g, float.(c_till)), float(q), float(u0))

"""
$(TYPEDSIGNATURES)

Regularized-Coulomb sliding law replicating Shakti.jl's own native regularization exactly (Sommers
et al. 2018 cavity-opening physics, `RegularizedCoulombSlidingLaw` in Shakti.jl's own
`sliding_law.jl`): `tau_b = C*N*(|v_b| / (|v_b| + |N|^n*lambda))^(1/n)`, where `lambda` is a per-cell
velocity-scale field (Shakti.jl: `lambda = 1.5 * A_visc`, see its `initial_conditions.jl`) -- NOT
[`RegularizedCoulombSlidingLaw`](@ref)'s fixed-`u0` regularization (Kazmierczak et al 2024/Joughin et
al 2019's own choice). Use this type only for bit-for-bit parity with a specific Shakti.jl run using
its own `RegularizedCoulombSlidingLaw(C)`; for any other purpose prefer
[`RegularizedCoulombSlidingLaw`](@ref) (the paper's own regularization).

# Notes

Unlike every other sliding law here, `lambda` is a per-cell field, not a scalar -- Shakti.jl derives
it from the ice-viscosity field `A_visc`, which this model already receives independently (see
`KazmierczakHydroModel`'s constructor) and genuinely varies in space. `calc_tau_b` (the plain-scalar
diagnostic helper the other laws provide for tests, see its own docstring in `sliding_law.jl`) is
intentionally not defined for this type since `lambda` can't be reduced to one number; only
`update_tau_b!` (the field version `resolve_q!` actually calls) is implemented.

`lambda` is wrapped via `alloc_field(grid, ...)` at construction -- see
[`RegularizedCoulombFieldSlidingLaw`](@ref)'s own field docstring for why.

# Fields
- `C::F`: Coulomb friction coefficient (matches Shakti.jl's `RegularizedCoulombSlidingLaw.C`)
- `n::F`: Glen's flow law exponent (matches Shakti.jl's `ModelParameters.n_exp`, default 3)
- `inv_n::F`: `1/n`, precomputed once (matches Shakti.jl's `canonical_exponent`/`pow` idiom)
- `lambda::A`: per-cell velocity scale, `lambda_coeff * A_visc` (Shakti.jl default `lambda_coeff = 1.5`)
"""
struct ShaktiRegularizedCoulombSlidingLaw{A, F <: AbstractFloat} <: AbstractPressureDependentSlidingLaw
    C     ::F
    n     ::F
    inv_n ::F
    lambda::A
end

"""
$(TYPEDSIGNATURES)

Builds a [`ShaktiRegularizedCoulombSlidingLaw`](@ref) on grid `g` from a Coulomb coefficient `C` and
the same ice-viscosity field `A_visc` already passed to `KazmierczakHydroModel`, computing `lambda =
lambda_coeff * A_visc` once here exactly as Shakti.jl's own `initial_conditions.jl` does, then
wrapping it via `alloc_field(g, ...)` (same idiom `KazmierczakHydroModel`'s own constructor uses for
`kappa`/`abs_v_b`/`A_visc`). Defaults (`n = 3.0`, `lambda_coeff = 1.5`) match Shakti.jl's own
defaults -- pass non-default values only if the Shakti.jl run being matched used a non-default
`ModelParameters(n_exp = ...)` or cavity constant.
"""
function ShaktiRegularizedCoulombSlidingLaw(g::AbstractHydroGrid, A_visc, C; n = 3.0, lambda_coeff = 1.5)
    F = eltype(A_visc)
    return ShaktiRegularizedCoulombSlidingLaw(F(C), F(n), F(1 / n), alloc_field(g, lambda_coeff .* A_visc))
end

"""
$(TYPEDSIGNATURES)

Convert a sliding law's parameters to float type `T`, mirroring the explicit `T(...)` conversions
`KazmierczakHydroModel`'s constructor applies to its own scalar parameters -- keeps `model.sliding_law`
type-stable with the rest of the model when `T` is not `Float64` (e.g. `Float32` grids).
"""
convert_sliding_law(::Type{T}, law::NoFrictionSlidingLaw) where {T <: AbstractFloat} = law
# Field-valued laws: returned unchanged (no copy) when the eltype already matches.
convert_sliding_law(::Type{T}, law::PrescribedFieldSlidingLaw) where {T <: AbstractFloat} =
    eltype(law.tau_b) === T ? law : PrescribedFieldSlidingLaw(T.(law.tau_b))
convert_sliding_law(::Type{T}, law::WeertmanSlidingLaw) where {T <: AbstractFloat} =
    WeertmanSlidingLaw(C = T(law.C), q = T(law.q))
convert_sliding_law(::Type{T}, law::PowerPlasticSlidingLaw) where {T <: AbstractFloat} =
    PowerPlasticSlidingLaw(c_till = T(law.c_till), q = T(law.q), u0 = T(law.u0))
convert_sliding_law(::Type{T}, law::RegularizedCoulombSlidingLaw) where {T <: AbstractFloat} =
    RegularizedCoulombSlidingLaw(c_till = T(law.c_till), q = T(law.q), u0 = T(law.u0))
convert_sliding_law(::Type{T}, law::RegularizedCoulombFieldSlidingLaw) where {T <: AbstractFloat} =
    eltype(law.c_till) === T ? law : RegularizedCoulombFieldSlidingLaw(T.(law.c_till), T(law.q), T(law.u0))
convert_sliding_law(::Type{T}, law::ShaktiRegularizedCoulombSlidingLaw) where {T <: AbstractFloat} =
    eltype(law.lambda) === T ? law : ShaktiRegularizedCoulombSlidingLaw(T(law.C), T(law.n), T(law.inv_n), T.(law.lambda))


"""
$(TYPEDSIGNATURES)

Basal shear stress tau_b [Pa] from `law`, given the current effective pressure `N` [Pa] and the
magnitude of the basal sliding velocity `abs_v_b` [m/s]. Feeds the frictional-heating term
tau_b*v_b/L_w of the melt rate (Eq. 3, Sec. 2.2.1 of Kazmierczak et al 2024). See the
`AbstractSlidingLaw` docstring above for the physics/provenance of each law. A plain scalar
function (used for tests/diagnostics on ordinary numbers) -- `update_tau_b!` below is the version
actually used inside the model's field broadcasts.
"""
calc_tau_b(::NoFrictionSlidingLaw, N, abs_v_b) = zero(abs_v_b)

calc_tau_b(law::WeertmanSlidingLaw, N, abs_v_b) = law.C * abs_v_b^law.q

calc_tau_b(law::PowerPlasticSlidingLaw, N, abs_v_b) = law.c_till * N * (abs_v_b / law.u0)^law.q

calc_tau_b(law::RegularizedCoulombSlidingLaw, N, abs_v_b) = law.c_till * N * (abs_v_b / (abs_v_b + law.u0))^law.q


"""
$(TYPEDSIGNATURES)

Update `model.tau_b` in place from `sliding_law` and the current `state.N`/`model.abs_v_b`.
Duplicates the `calc_tau_b` formulas above with the law's parameters pulled out as plain scalar
locals first, rather than calling `calc_tau_b` from inside the `@.` broadcast, so the broadcast
contains only plain arithmetic (`+`, `*`, `/`, `^`) on fields and scalar locals, exactly as the rest
of this model already does (e.g. `update_N_inf!` in effective_pressure.jl). That keeps it valid for
any grid backend whose fields support standard broadcasting, not just plain `Array`s.
"""
function update_tau_b!(model::AbstractKazmierczakHydroModel, state::HydroState, ::NoFrictionSlidingLaw)
    model.tau_b .= 0.0
    return nothing
end

function update_tau_b!(model::AbstractKazmierczakHydroModel, state::HydroState, law::PrescribedFieldSlidingLaw)
    model.tau_b .= law.tau_b
    return nothing
end

function update_tau_b!(model::AbstractKazmierczakHydroModel, state::HydroState, law::WeertmanSlidingLaw)
    C, q = law.C, law.q
    @. model.tau_b = C * model.abs_v_b^q
    return nothing
end

function update_tau_b!(model::AbstractKazmierczakHydroModel, state::HydroState, law::PowerPlasticSlidingLaw)
    c_till, q, u0 = law.c_till, law.q, law.u0
    @. model.tau_b = c_till * state.N * (model.abs_v_b / u0)^q
    return nothing
end

# RegularizedCoulombSlidingLaw (scalar c_till) and RegularizedCoulombFieldSlidingLaw (per-cell
# c_till) share this one method: the formula is identical, and broadcasting already handles a
# scalar or a Field for c_till transparently, so there is nothing that actually differs per type.
function update_tau_b!(model::AbstractKazmierczakHydroModel, state::HydroState, law::Union{RegularizedCoulombSlidingLaw, RegularizedCoulombFieldSlidingLaw})
    c_till, q, u0 = law.c_till, law.q, law.u0
    @. model.tau_b = c_till * state.N * (model.abs_v_b / (model.abs_v_b + u0))^q
    return nothing
end

function update_tau_b!(model::AbstractKazmierczakHydroModel, state::HydroState, law::ShaktiRegularizedCoulombSlidingLaw)
    C, n, inv_n, lambda = law.C, law.n, law.inv_n, law.lambda
    @. model.tau_b = C * state.N * (model.abs_v_b / (model.abs_v_b + abs(state.N)^n * lambda))^inv_n
    return nothing
end
