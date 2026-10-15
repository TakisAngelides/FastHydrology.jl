#################################
# Model: Kazmierczak et al 2024 #
#################################

# The reportable water layer thickness state.W: the closures update_W! selects between.


"""
$(TYPEDSIGNATURES)

Trait selecting which closure `update_W!` uses to compute the reportable subglacial water thickness
`state.W` (see `KazmierczakHydroModel`'s `water_thickness_algorithm` keyword). Stored as a type
parameter, resolved by multiple dispatch at compile time -- the same pattern as
`AbstractPsiOutAlgorithm`/`AbstractDissipationMelt` -- so only the selected closure's fields
are touched each call, not all three.

`state.W` is documented as a *water layer thickness* -- an areal, grid-cell-averaged quantity, the
same kind of thing a sheet model's `W` is -- so every closure offered here reports one. K24's own
local conduit depth `model.H` (`H = (1-kappa)*H_hard + kappa*H_soft`, already computed by `update_H!`
for `N_inf`) deliberately has no closure here: it answers a different question (the depth *inside*
one conduit, not smeared over a grid cell) and is not areal, so reporting it as `state.W` would be
comparing apples to oranges against the other closures. Use `model.H` directly if you want that
quantity -- there is nothing stopping you, it just isn't one of `state.W`'s options.

The three closures still answer different physical questions and are not interchangeable:
- [`DarcyWeisbachThickness`](@ref) (the default): inverts the turbulent parallel-plate
  Darcy-Weisbach closure for a wide slot -- the turbulent analogue of the laminar Le Brocq/Weertman
  closure below, and the closure consistent with K24's own turbulent-flow assumption for `q` -- using
  the model's own distributed flux `q` and, by default, a domain-wide masked mean of the potential
  gradient (`gradient_convention = MeanGradient()`, the same averaging convention `LaminarThickness`
  uses, kept consistent between the package's two sheet-flow closures): `d = (f*rho_w*q^2 /
  (4*abs_grad_phi0))^(1/3)`. Clamped to `[Wmin, Wmax]`, since it represents a thin-sheet quantity.
  Pass `gradient_convention = LocalGradient()` for the local, per-cell gradient instead (matching the
  convention `update_S_inf!` uses) -- see `AbstractGradientConvention`'s docstring below.
- [`ArealConduitThickness`](@ref): `S_inf / l_c`, the conduit's cross-sectional area smeared over the
  inter-conduit spacing -- a grid-cell-averaged "equivalent film thickness" comparable to a sheet
  model's W, derived from the same turbulent Manning-Strickler `S_inf` that drives `N_inf`. Bed-type
  independent (kappa only blends H's shape assumption, not S_inf itself) and not clamped to `[Wmin,
  Wmax]`: those bounds were chosen for a thin distributed sheet, and this areal quantity can
  legitimately exceed them too (a dense-enough conduit network smeared over its spacing is not
  bounded the same way a single laminar/turbulent sheet-flow inversion is).
- [`LaminarThickness`](@ref): the original closure (Eq. 8, Kazmierczak et al 2022 / Le Brocq et al
  2009 Eq. 2), `d = (12*eta_w*q / abs_grad_phi0_s)^(1/3)`, using a single domain-mean smoothed
  gradient by default (as Kori-ULB's own `SubWaterFlux.m`) rather than the local one -- see its
  `gradient_convention` type parameter. The magnitude is Euclidean, unlike Kori-ULB's L1 `|sx| + |sy|`. Kept for direct comparison against Kori-ULB's `Wd`/SHAKTI's
  laminar sheet closure, but physically inconsistent with K24's own turbulent-flow assumption for
  `q` -- not the default, and not recommended as "the" reported thickness for a K24 run. Clamped to
  `[Wmin, Wmax]`.

Both `DarcyWeisbachThickness` and `LaminarThickness` take an `AbstractGradientConvention` (see below)
as a type parameter, the same trait-dispatch pattern as `AbstractDissipationMelt`/`AbstractPsiOutAlgorithm`
above, rather than a plain `Bool` field: the local-gradient and domain-mean-gradient cases have a
genuinely different type for the gradient term (a `Field` vs a bare scalar) in the broadcast each
closure runs, so branching on a runtime `Bool` field would leave that a small `Union` type and risk
keeping the `@.` broadcast from being fully specialized. Baking the choice into the algorithm's own
type instead -- so it's already fixed by the time `typeof(model)` is known, not read from a field at
runtime -- means `update_W!` dispatches directly to a concretely-typed method body, no extra
conversion step needed.
"""
abstract type AbstractWaterThicknessAlgorithm end

"""
$(TYPEDSIGNATURES)

Trait selecting which gradient value `DarcyWeisbachThickness`/`LaminarThickness` divide by: the
local, per-cell gradient ([`LocalGradient`](@ref)) or a single domain-wide masked mean
([`MeanGradient`](@ref)). See [`AbstractWaterThicknessAlgorithm`](@ref)'s docstring for why this is a
type parameter rather than a `Bool` field.
"""
abstract type AbstractGradientConvention end

"""
$(TYPEDSIGNATURES)

Use the local, per-cell gradient magnitude -- matches the convention `update_S_inf!` already uses.
`LaminarThickness`'s alternative to its own default ([`MeanGradient`](@ref)); available on
`DarcyWeisbachThickness` too, as the alternative to its own default (also `MeanGradient`, chosen for
consistency between the two closures -- see [`MeanGradient`](@ref)'s docstring), for callers who want
the locally-varying-gradient convention that's the more usual physical choice for a turbulent closure.
"""
struct LocalGradient <: AbstractGradientConvention end

"""
$(TYPEDSIGNATURES)

Use a single domain-wide masked mean of the gradient magnitude, in place of its local, per-cell
value. `LaminarThickness`'s default, matching Kori-ULB's own `SubWaterFlux.m` (`mean(gdsmag(...))`);
also `DarcyWeisbachThickness`'s default, for consistency between the package's two sheet-flow
closures (both then use the same averaging convention, differing only in laminar-vs-turbulent
physics) -- pass `gradient_convention = LocalGradient()` to either if you want the locally-varying
gradient instead (the more usual physical choice for a turbulent closure specifically, since a
domain-wide mean has no particular physical justification there the way it does, by construction, for
reproducing Kori-ULB's own laminar code).
"""
struct MeanGradient <: AbstractGradientConvention end

"""
$(TYPEDSIGNATURES)

Report `model.S_inf / model.l_c` (the conduit cross-section smeared over the inter-conduit spacing)
as `state.W`. See [`AbstractWaterThicknessAlgorithm`](@ref)'s docstring for the full comparison.
"""
struct ArealConduitThickness <: AbstractWaterThicknessAlgorithm end

"""
$(TYPEDSIGNATURES)

Report the turbulent Darcy-Weisbach sheet-flow inversion `(f*rho_w*q^2 / (4*abs_grad_phi0))^(1/3)` as
`state.W`, clamped to `[Wmin, Wmax]`. `KazmierczakHydroModel`'s default `water_thickness_algorithm`
(with `gradient_convention = MeanGradient()`). See [`AbstractWaterThicknessAlgorithm`](@ref)'s
docstring for the full comparison, and [`AbstractGradientConvention`](@ref)'s docstring for what
`gradient_convention = LocalGradient()` does here instead.
"""
struct DarcyWeisbachThickness{G <: AbstractGradientConvention} <: AbstractWaterThicknessAlgorithm
    gradient_convention::G
end
DarcyWeisbachThickness(; gradient_convention = MeanGradient()) = DarcyWeisbachThickness(gradient_convention)

"""
$(TYPEDSIGNATURES)

Report the original laminar Le Brocq/Weertman sheet-flow inversion `(12*eta_w*q /
abs_grad_phi0_s)^(1/3)` as `state.W`, clamped to `[Wmin, Wmax]`, with `gradient_convention =
MeanGradient()` by default (as Kori-ULB's own `SubWaterFlux.m`, but with the Euclidean gradient
magnitude rather than its L1 `|sx| + |sy|`). See
[`AbstractWaterThicknessAlgorithm`](@ref)'s docstring for why this closure is kept but not the
default, and [`AbstractGradientConvention`](@ref)'s docstring for what `gradient_convention =
LocalGradient()` does here.
"""
struct LaminarThickness{G <: AbstractGradientConvention} <: AbstractWaterThicknessAlgorithm
    gradient_convention::G
end
LaminarThickness(; gradient_convention = MeanGradient()) = LaminarThickness(gradient_convention)


"""
$(TYPEDSIGNATURES)

Update the water layer thickness `state.W`, dispatching on `model.water_thickness_algorithm`
(`ArealConduitThickness()`, `DarcyWeisbachThickness()`, or `LaminarThickness()` -- see
`AbstractWaterThicknessAlgorithm`'s docstring above) to whichever single closure is selected,
so only that closure's fields are touched each call.

Must run after `update_N!` (specifically after its `update_S_inf!` call), not before --
`ArealConduitThickness` reads `model.S_inf`, which `update_N!` is what keeps current.
`update_steady_state!` in run.jl calls `update_q!`, then `update_N!`, then `update_W!` in that order
for this reason; `state.W` itself feeds back into nothing else in the model, so it is safe to compute
last.
"""
function update_W!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)
    update_W!(model, grid, state, model.water_thickness_algorithm)
    return nothing
end


"""
$(TYPEDSIGNATURES)

`ArealConduitThickness`: report `model.S_inf / model.l_c` -- the conduit cross-section smeared over
the inter-conduit spacing -- as `state.W`, unclamped. See `AbstractWaterThicknessAlgorithm`'s
docstring above for the derivation and why `S_inf` (not `H`) is the right numerator.
"""
function update_W!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::ArealConduitThickness)
    @. state.W = model.S_inf / model.l_c
    return nothing
end


"""
$(TYPEDSIGNATURES)

`DarcyWeisbachThickness` (`KazmierczakHydroModel`'s default `water_thickness_algorithm`): invert the
turbulent parallel-plate Darcy-Weisbach closure for a wide slot, `d = (f*rho_w*q^2 /
(4*abs_grad_phi0))^(1/3)`. `model.f` here is the same Darcy-Weisbach friction factor K24's own
conduit closure uses (folded into `K` by `update_S_inf!`) -- one shared parameter, not a second one
introduced for this closure, since both derive from the same underlying physics (Schoof 2010/Clarke
1996); see `f`'s field comment on `KazmierczakParams` in model.jl. Uses a domain-wide masked mean of
the potential gradient by default
(`gradient_convention = MeanGradient()`, the same averaging convention `LaminarThickness` uses, kept
consistent between the package's two sheet-flow closures -- see `AbstractGradientConvention`'s
docstring above), or, with `gradient_convention = LocalGradient()`, the local, per-cell
gradient instead (matching the convention `update_S_inf!` uses). The turbulent analogue of
`LaminarThickness` below, and the closure consistent with K24's own turbulent-flow assumption for `q`
(unlike `LaminarThickness`) -- clamped to `[Wmin, Wmax]` since it represents the same kind of
thin-sheet quantity. The `+ eps(T)` guards degenerate cells where `abs_grad_phi0` is exactly zero
(e.g. flat cells outside the glacier extent). Uses `q*q` rather than `q^2.0`: `Float64^Float64`
dispatches to libm's `pow()` per element, ~17x slower (benchmarked) than a plain multiply for no
numerical difference, and `q*q` also sidesteps `@.`'s `literal_pow` rewrite of `q^2`, which not
every field type's broadcasting supports.
"""
function update_W!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, algorithm::DarcyWeisbachThickness)
    update_W_darcy_weisbach!(model, grid, state, algorithm.gradient_convention)
    return nothing
end

function update_W_darcy_weisbach!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::LocalGradient)
    # eps(T) computed once as a plain scalar before the `@.` broadcast -- see update_q!'s own note.
    T = eltype(model.abs_grad_phi0)
    epsT = eps(T)
    @. state.W = min(model.Wmax, max(model.Wmin,
        (model.f * model.rho_w * model.q * model.q / (4 * model.abs_grad_phi0 + epsT))^(1/3)))
    return nothing
end

function update_W_darcy_weisbach!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::MeanGradient)
    abs_grad_phi0_mean = masked_mean(grid, model.abs_grad_phi0, state.mask)
    # eps(T) computed as a plain scalar BEFORE the `@.` broadcast -- see update_q!'s own note.
    T = eltype(model.abs_grad_phi0)
    epsT = eps(T)
    @. state.W = min(model.Wmax, max(model.Wmin,
        (model.f * model.rho_w * model.q * model.q / (4 * abs_grad_phi0_mean + epsT))^(1/3)))
    return nothing
end


"""
$(TYPEDSIGNATURES)

`LaminarThickness`: the original closure (Eq. 8, Kazmierczak et al 2022 / Le Brocq et al 2009 Eq. 2),
clamped to `[Wmin, Wmax]`. Kept for direct comparison against Kori-ULB's `Wd`/SHAKTI's laminar sheet
closure -- see `AbstractWaterThicknessAlgorithm`'s docstring above for why this closure is not
the default (`DarcyWeisbachThickness` is). Uses a single domain-mean smoothed gradient magnitude by
default (`gradient_convention = MeanGradient()`), as Kori-ULB's own `mean(gdsmag(...))` in
`SubWaterFlux.m` -- but of the Euclidean magnitude, where `gdsmag` is the L1 `|sx| + |sy|` (so `d` is
up to ~11% larger than Kori-ULB's for flow at 45 degrees to the grid, ~8% on a domain mean) -- and matching `DarcyWeisbachThickness`'s own default too (kept
consistent between the package's two sheet-flow closures); pass `gradient_convention =
LocalGradient()` for the local-per-cell-gradient variant instead, e.g. to isolate how much of a
closure's spatial pattern comes from that domain-mean averaging versus from its own physics --
`DarcyWeisbachThickness`'s `gradient_convention` runs the identical comparison for the turbulent
closure.
"""
function update_W!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, algorithm::LaminarThickness)
    update_W_laminar!(model, grid, state, algorithm.gradient_convention)
    return nothing
end

function update_W_laminar!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::MeanGradient)
    abs_grad_phi0_s_mean = masked_mean(grid, model.abs_grad_phi0_s, state.mask)
    @. state.W = min(model.Wmax, max(model.Wmin, (12 * model.eta_w * model.q / abs_grad_phi0_s_mean)^(1/3)))
    return nothing
end

function update_W_laminar!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::LocalGradient)
    # eps(T) computed as a plain scalar BEFORE the `@.` broadcast -- see update_q!'s own note.
    T = eltype(model.abs_grad_phi0_s)
    epsT = eps(T)
    @. state.W = min(model.Wmax, max(model.Wmin, (12 * model.eta_w * model.q / (model.abs_grad_phi0_s + epsT))^(1/3)))
    return nothing
end
