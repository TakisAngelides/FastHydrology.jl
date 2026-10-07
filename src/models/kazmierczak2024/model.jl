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
  longest flow-routing chain in the domain, which for real ice-sheet grids can exceed the
  calling process's `ulimit -s` and crash with a `StackOverflowError` -- independent of, and
  potentially before, `max_psi_out_calls` ever binds (that cap limits total cells visited per sweep,
  not recursion depth along any one chain).
- `IterativePsiOut` has no such requirement (its "stack" is a plain `Vector`, bounded only by heap
  memory), at the cost of that ~30-40x slowdown. Both `RecursivePsiOut` and `IterativePsiOut` are, at
  bottom, the same graph traversal (a DFS with an explicit or implicit stack, revisiting cells to fold
  child contributions back in) -- neither exploits the fact that the underlying dependency graph has
  no cycles.

`TopologicalPsiOut` is a genuinely different algorithm, not just a different traversal strategy for
the same one: since `psi_out[i,j]` only ever depends on hydraulically-upstream neighbours (`w > 0`),
*if* the dependency graph those `w > 0` edges define is a DAG, it can be processed in one O(N) pass
via Kahn's algorithm (a BFS ordered by in-degree: a cell becomes ready once every upstream neighbour
that flows into it has already been finalized, at which point its own contribution is added to
whichever neighbours *it* flows into and their remaining in-degree is decremented, in turn making
more cells ready) -- no recursion, no explicit revisit stack, and no need for `max_psi_out_calls`.

The load-bearing word above is "if". `w` is evaluated from `minus_grad_phi0_sx/sy` -- the *smoothed*
gradient (from `update_smoothed_potential_gradients!`'s stress-gradient-coupling convolution) when
`longcoupwater != 0`, or the raw central-difference gradient (from `update_potential_gradients!`)
when `longcoupwater == 0` -- and neither is guaranteed to be a true conservative/gradient-derived
vector field once normalized per cell, so the graph they define is not guaranteed acyclic. It *is*
acyclic on every idealized synthetic test grid used in this package's own test suite (smooth,
monotonic sloped surfaces), at any `longcoupwater`. It is **not** acyclic on the real Thwaites-2km
dataset used to develop this function, confirmed by explicit DFS cycle detection (not inferred) --
and, contrary to an earlier version of this docstring's claim, this is true regardless of
`longcoupwater`: cycles affect roughly half of all grounded cells at the model's default
`longcoupwater = 5.0`, but *more* than half (37514 of 51945) with smoothing turned off entirely
(`longcoupwater = 0`), and persist at a coarsened ~8km version of the same grid too (2573 of 3242).
Real, noisy topography is apparently enough on its own to produce local circulation in a
per-cell-normalized central-difference gradient field -- this is not specific to the
stress-gradient-coupling smoothing step. There is currently no known setting that makes this
algorithm reliably exact on real (non-idealized) ice-sheet data.

Recursion-based traversal doesn't actually solve the underlying problem either --
`RecursivePsiOut`/`IterativePsiOut` silently return an early, incomplete snapshot for a cell that's
already mid-computation when a cycle loops back to it, giving some order-dependent approximate value
with no warning at all -- but it never leaves a cell entirely unprocessed the way a topological sort
must, so it doesn't fail visibly. Given that, this implementation treats a detected cycle as a hard
error by default (`TopologicalPsiOut()`, `allow_cycles = false`): silently returning a badly wrong
field (in the case that surfaced this, q correlation against `RecursivePsiOut` collapsed from -0.03
to ~0.53 depending on configuration, neither acceptable) is worse than failing loudly. Pass
`allow_cycles = true` to instead get the old "locally truncated but finite" behaviour -- cells inside
a cycle keep only their partial upstream contributions from outside it (their own source term never
added) -- with a warning (once per sweep) instead of an error. In practice, expect this algorithm to
error on any real (non-idealized) domain; it is exact and useful on synthetic/idealized grids, or on
a real domain a caller has independently verified to be acyclic for their specific gradient field.
Prefer `RecursivePsiOut`/`IterativePsiOut` for real ice-sheet data.
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

Route psi_out with a single-pass topological sort (Kahn's algorithm) over the actual flow-direction
dependency graph (`update_psi_out_topological!`), rather than traversing it via recursion/an explicit
revisit stack. Exact (matches `RecursivePsiOut`/`IterativePsiOut` to floating-point precision) only
when that graph is genuinely acyclic -- true on every idealized synthetic test grid in this package's
test suite, but confirmed **not** true (real cycles, at any `longcoupwater`) on the real Thwaites-2km
dataset used to develop this function. See [`AbstractPsiOutAlgorithm`](@ref)'s docstring for the full
story, including why `longcoupwater = 0` does not make this reliably safe on real data either.

# Fields
- `allow_cycles::Bool`: if `false` (the default), a detected cycle throws an error rather than
  silently returning a wrong field. If `true`, falls back to the "locally truncated but finite"
  behaviour instead (a warning, not an error) -- cells inside a cycle keep only their partial
  upstream contributions.
"""
struct TopologicalPsiOut <: AbstractPsiOutAlgorithm
    allow_cycles::Bool
end
TopologicalPsiOut(; allow_cycles = false) = TopologicalPsiOut(allow_cycles)

"""
$(TYPEDSIGNATURES)

Route psi_out by replaying a recorded "tape" of [`RecursivePsiOut`](@ref)'s traversal
(`update_psi_out_taped!`; `KazmierczakHydroModel`'s default `psi_out_algorithm`). The order in which
the recursion visits cells and folds upstream contributions in depends only on the mask and the
routing weights (i.e. the smoothed potential gradients), never on `mdot_total`. Those are fixed for a
whole `update_q!` call, while the dissipation-melt and `(q, N)` Picard loops re-route the same graph
with a new `mdot_total` every sweep. So the traversal is recorded once per `update_q!` call as a flat
list of `psi_out[c] += psi_out[n] * w` / `psi_out[c] = max(0, psi_out[c])` operations, in exactly the
order the recursion executes them (including its partial reads of cells still on the stack when the
flow graph has a cycle, and its `max_psi_out_calls` cutoff), and each sweep just replays that list.
The result matches `RecursivePsiOut` bit-for-bit. Recording uses an explicit stack, so there is no
recursion-depth limit either.

The tape is invalidated at the start of every `update_q!` (when the gradients are recomputed). If you
call `route_psi_out!` yourself after changing the mask or the gradients some other way, call
`invalidate_routing_tape!(model)` first.
"""
struct TapedPsiOut <: AbstractPsiOutAlgorithm end

"""
$(TYPEDSIGNATURES)

How the frictional-heating term `tau_b . u_b / L_w` of the melt rate is discretised (see
`KazmierczakHydroModel`'s `friction_discretization` keyword). Either way it is a cell-centred melt
rate, recomputed every Picard sweep from the current `model.tau_b` (so it follows an N-dependent
sliding law through the `(q, N)` loop):

- [`CellCentredFriction`](@ref) (default): `tau_b * |u_b|` from the cell-centred `model.tau_b` and
  `model.abs_v_b`.
- [`StaggeredFriction`](@ref): for velocities on an Arakawa C-grid, as an ice model such as Yelmo
  provides them (`u_x` on x-faces, `u_y` on y-faces). The sliding law is still evaluated at the cell
  centre; its drag coefficient `beta = tau_b / |u_b|` is averaged to each face, giving the face
  tractions `tau_x = beta_face * u_x`, `tau_y = beta_face * u_y` (the way Yelmo builds `taub_acx/acy`
  from `beta_acx/acy`). The heat is then either formed on the faces or at Gauss points -- see
  `StaggeredFriction`.
"""
abstract type AbstractFrictionDiscretization end

"""$(TYPEDSIGNATURES)\n\nDefault: `tau_b * |u_b|` at cell centres. See [`AbstractFrictionDiscretization`](@ref)."""
struct CellCentredFriction <: AbstractFrictionDiscretization end

"""
$(TYPEDSIGNATURES)

Frictional heating from C-grid (staggered) basal velocities. `ux[i, j]` is the x-velocity on the
face between cells `i` and `i+1` (Yelmo's `acx` convention), `uy[i, j]` the y-velocity on the face
between cells `j` and `j+1` (`acy`), both [m/s], size `(Nx, Ny)`. See
[`AbstractFrictionDiscretization`](@ref).

- `quadrature = false` (faces): `tau_x u_x` is formed on each x-face and `tau_y u_y` on each y-face --
  both factors of each product live on that face -- and each is averaged from the cell's two faces to
  its centre: `Q = (P_x[i-1/2] + P_x[i+1/2])/2 + (P_y[j-1/2] + P_y[j+1/2])/2`, `P = beta_face u_face^2 >= 0`.
  The energy-consistent staggered form (the same construction as `FaceDissipation`).
- `quadrature = true`: Yelmo's own `qb_method = 2` (`calc_basal_heating_nodes`): the face velocities
  and tractions are interpolated bilinearly to the 4 points of a 2x2 Gauss quadrature in the cell,
  `|u||tau|` is formed there and averaged. Use this to reproduce the frictional heat Yelmo itself
  uses in its ice-temperature solve.

`beta` at a cell centre is `tau_b / max(|u_b|, u_floor)` (default `u_floor` 1e-3 m/yr, Yelmo's
`ub_sq_min`). Faces on the domain edge use the one cell they have.
"""
struct StaggeredFriction{M} <: AbstractFrictionDiscretization
    ux         ::M
    uy         ::M
    quadrature ::Bool
    u_floor    ::Float64
end
StaggeredFriction(ux::AbstractMatrix, uy::AbstractMatrix; quadrature::Bool = false, u_floor = perYear2perSecond(1e-3)) =
    StaggeredFriction(Matrix{Float64}(ux), Matrix{Float64}(uy), quadrature, Float64(u_floor))

"""
$(TYPEDSIGNATURES)

Flux-routing scheme: how each grounded cell's outflow `psi_out` is shared among its neighbours (see
`KazmierczakHydroModel`'s `routing_scheme` keyword). The options are the algorithms compared by Le
Brocq, Payne & Siegert (2006, Computers & Geosciences 32, 1780-1795, Sec. 3 and Table 1), named as there:

- [`Warner`](@ref) (default; Budd & Warner 1996, Eq. 8): 4 neighbours, shared in proportion to the
  potential drop to each downhill neighbour -- i.e. the gradient on each cell face.
- [`GDSWarner`](@ref) (the original K24/KORI scheme): direction from the Kamb-smoothed gradient
  components of the filled potential; outflow to at most 2 of the 4 neighbours, by component.
- [`Quinn`](@ref) (Quinn et al. 1991, Eq. 8): as Warner over all 8 neighbours; `Quinn(original = true)`
  uses Quinn et al.'s own slope x contour-length weights.
- [`Tarboton`](@ref) (D-infinity, Tarboton 1997): steepest direction over 8 triangular facets, split
  between the 2 neighbours bracketing it.
- [`ModifiedTarboton`](@ref): direction from the local 4-neighbour slope, split between 2 of 8 neighbours.
- [`GDSTarboton`](@ref): direction from the smoothed gradient components (as GDSWarner), split
  between 2 of 8 neighbours.

For Warner/Quinn/Tarboton/ModifiedTarboton the Kamb & Echelmeyer (1986) smoothing is applied to the
potential itself (then filled), rather than to its gradient components -- the same coupling, since
smoothing commutes with differentiation, but the routing then follows one surface. Warner and Quinn
only ever send water downhill on that surface, so their routing graph has no cycles and the routed
flux is conserved exactly (up to cells with no downhill neighbour, which a hollow fill removes -- use
`PriorityFloodFill` with them). The GDS schemes can send water uphill on the filled potential (Le
Brocq Sec. 3, Step 5), which is what creates routing cycles: on the Thwaites 2 km / GrIS 8 km / AIS
16 km test data GDSWarner loses 3% / 8% / 34% of the positive basal melt in them, Warner none. Schemes
other than `GDSWarner` need `psi_out_algorithm = TapedPsiOut()` (the default).

`fill_algorithm`, `q_conversion` and `dissipation_discretization` default to `nothing`, meaning "the
natural partner of `routing_scheme`": `PriorityFloodFill()`, `QFromFaceAverage()` and
`FaceDissipation()` for `Warner`; `JacobiFill()`, `QFromOutflow()` and `CellCentredDissipation()` for
the GDS schemes (so `routing_scheme = GDSWarner()` alone reproduces the original K24 set-up exactly);
`PriorityFloodFill()`, `QFromOutflow()` and `CellCentredDissipation()` for the other 8-neighbour
schemes. Pass any of them explicitly to override.
"""
abstract type AbstractRoutingScheme end
"""$(TYPEDSIGNATURES)

Original K24/KORI routing. See [`AbstractRoutingScheme`](@ref)."""
struct GDSWarner <: AbstractRoutingScheme end
"""$(TYPEDSIGNATURES)

Budd & Warner (1996) 4-neighbour potential-drop (face-gradient) routing; the default. See [`AbstractRoutingScheme`](@ref)."""
struct Warner <: AbstractRoutingScheme end
"""
$(TYPEDSIGNATURES)

Quinn et al. (1991) 8-neighbour multiple-flow-direction routing. See [`AbstractRoutingScheme`](@ref).

- `Quinn()` / `Quinn(original = false)`: as written in Le Brocq et al. (2006) Eq. 8 -- shares
  proportional to the potential drop to each downhill neighbour, with no distance or contour-length
  factor (so per unit drop a diagonal gets the same weight as a cardinal neighbour).
- `Quinn(original = true)`: Quinn et al.'s own weights, `tan(beta_d) * L_d` with
  `tan(beta_d) = (phi_c - phi_n) / delta_d` (`delta_d` = dx, dy or sqrt(dx^2 + dy^2)) and effective
  contour lengths `L = 0.5 * Delta` (cardinal) and `0.354 * Delta` (diagonal). For rectangular cells:
  `0.5 * dy` for x-neighbours, `0.5 * dx` for y-neighbours, `(sqrt(2)/4) * sqrt(dx * dy)` for diagonals.
"""
struct Quinn <: AbstractRoutingScheme
    original::Bool
end
Quinn(; original::Bool = false) = Quinn(original)
"""$(TYPEDSIGNATURES)

Tarboton (1997) D-infinity routing. See [`AbstractRoutingScheme`](@ref)."""
struct Tarboton <: AbstractRoutingScheme end
"""$(TYPEDSIGNATURES)

Le Brocq et al. (2006) modified Tarboton routing. See [`AbstractRoutingScheme`](@ref)."""
struct ModifiedTarboton <: AbstractRoutingScheme end
"""$(TYPEDSIGNATURES)

Le Brocq et al. (2006) GDS-Tarboton routing. See [`AbstractRoutingScheme`](@ref)."""
struct GDSTarboton <: AbstractRoutingScheme end

"""
$(TYPEDSIGNATURES)

Number of neighbour directions a routing scheme can send water to (4 or 8).
"""
n_directions(::Union{GDSWarner, Warner}) = 4
n_directions(::AbstractRoutingScheme) = 8

"""
$(TYPEDSIGNATURES)

How the routed flux `psi_out` [m3/s] becomes the cell-centred distributed flux `q` [m2/s] (see
`KazmierczakHydroModel`'s `q_conversion` keyword):
- [`QFromOutflow`](@ref) (default for all schemes except `Warner`): `q = psi_out / corfac`, the cell's outflow over the flux
  cross-section `dx|sin| + dy|cos|` of its flow direction (Le Brocq et al. 2006 Eq. 9). Works for every
  routing scheme; describes the flux at the cell's downstream side.
- [`QFromFaceAverage`](@ref): face fluxes `F` from the routing give face-normal `q_x = F/dy`,
  `q_y = F/dx`; each component is averaged from its two faces to the centre and `q = |(q_x, q_y)|`.
  Describes the flux at the centre (mean of inflow and outflow). 4-neighbour schemes only; the
  default with `Warner` routing.
"""
abstract type AbstractQConversion end
"""$(TYPEDSIGNATURES)

See [`AbstractQConversion`](@ref)."""
struct QFromOutflow <: AbstractQConversion end
"""$(TYPEDSIGNATURES)

See [`AbstractQConversion`](@ref)."""
struct QFromFaceAverage <: AbstractQConversion end

"""
$(TYPEDSIGNATURES)

How the dissipation melt `q . grad(phi0) / L_w` is discretised (see `KazmierczakHydroModel`'s
`dissipation_discretization` keyword). The result is a cell-centred melt rate either way:
- [`CellCentredDissipation`](@ref) (default for all schemes except `Warner`): `|q| * |grad(phi0)| / L_w` from the centred q and
  centred gradient of the true potential.
- [`FaceDissipation`](@ref) (default with `Warner` routing): the product is formed on each face, where both factors live on a
  staggered grid -- face flux times the true-potential drop across the face, the energy the water
  releases crossing it -- and averaged to the centre. Signed (water pushed up the true potential
  gives a negative term). 4-neighbour schemes only.
"""
abstract type AbstractDissipationDiscretization end
"""$(TYPEDSIGNATURES)

See [`AbstractDissipationDiscretization`](@ref)."""
struct CellCentredDissipation <: AbstractDissipationDiscretization end
"""$(TYPEDSIGNATURES)

See [`AbstractDissipationDiscretization`](@ref)."""
struct FaceDissipation <: AbstractDissipationDiscretization end

"""
$(TYPEDSIGNATURES)

How `potential_filling!` removes local minima ("pits") of the routing potential `phi0_filled`, so
water routed by `update_psi_out!` cannot get trapped in them (see `KazmierczakHydroModel`'s
`fill_algorithm` keyword):
- [`JacobiFill`](@ref) (default with the GDS routing schemes; the original K24/KORI filling):
  `fill_iters` passes of raising each strict minimum to the mean of its 4 neighbours. Converges slowly
  and does not guarantee a pit-free result.
- [`PriorityFloodFill`](@ref) (default with the other routing schemes, including the default
  `Warner`): raises every grounded pit/flat to its spill level plus a tiny slope,
  in one pass; guarantees every grounded cell drains to the grounding line/ice margin/domain edge.
"""
abstract type AbstractFillAlgorithm end

"""
$(TYPEDSIGNATURES)

The original iterative filling (`fill_iters` Jacobi passes of neighbour-mean raising). See
[`AbstractFillAlgorithm`](@ref).
"""
struct JacobiFill <: AbstractFillAlgorithm end

"""
$(TYPEDSIGNATURES)

Hollow filling as described by Le Brocq et al. (2006, Sec. 3.1): each strict local minimum is given
the value of its lowest neighbour, repeated for up to `fill_iters` passes. Leaves flats where
hollows were. See [`AbstractFillAlgorithm`](@ref).
"""
struct LowestNeighbourFill <: AbstractFillAlgorithm end

"""
$(TYPEDSIGNATURES)

Priority-Flood+epsilon filling of grounded cells (Barnes et al. 2014). `epsilon` [Pa] is the potential
step added per cell across a filled pit or flat so it keeps a (tiny) downhill direction; the default
1 Pa is ~0.1 mm of water head, negligible next to the ~1e3-1e5 Pa potential differences between
neighbouring cells. See [`AbstractFillAlgorithm`](@ref).
"""
struct PriorityFloodFill <: AbstractFillAlgorithm
    epsilon::Float64
end
PriorityFloodFill(; epsilon = 1.0) = PriorityFloodFill(epsilon)

"""
$(TYPEDSIGNATURES)

Recorded traversal used by [`TapedPsiOut`](@ref): operation `k` is `psi_out[dst_i[k], dst_j[k]] +=
psi_out[src_i[k], src_j[k]] * w[k]`, or, when `src_i[k] == 0`, the clamp `psi_out[dst] = max(0,
psi_out[dst])`. `valid` is cleared whenever the routing graph may have changed.
"""
mutable struct RoutingTape{T}
    dst_i ::Vector{Int32}
    dst_j ::Vector{Int32}
    src_i ::Vector{Int32}
    src_j ::Vector{Int32}
    w     ::Vector{T}
    valid ::Bool
    w8    ::Array{T, 3}   # w8[d, i, j]: fraction of cell (i, j)'s outflow sent in direction d (ROUTE_OFFSETS); used by non-default routing schemes and face fluxes
    Fx    ::Matrix{T}     # (Nx+1, Ny) net x-face volume fluxes [m3/s], for QFromFaceAverage/FaceDissipation
    Fy    ::Matrix{T}     # (Nx, Ny+1) net y-face volume fluxes [m3/s]
    diss  ::Matrix{T}     # face-assembled dissipation melt [kg/m2/s] (FaceDissipation)
end
RoutingTape{T}(Nx::Int, Ny::Int) where {T} = RoutingTape{T}(Int32[], Int32[], Int32[], Int32[], T[], false,
    zeros(T, 8, Nx, Ny), zeros(T, Nx + 1, Ny), zeros(T, Nx, Ny + 1), zeros(T, Nx, Ny))


"""
$(TYPEDSIGNATURES)

Trait selecting which closure `update_W!` uses to compute the reportable subglacial water thickness
`state.W` (see `KazmierczakHydroModel`'s `water_thickness_algorithm` keyword). Stored as a type
parameter, resolved by multiple dispatch at compile time -- the same pattern as
`AbstractPsiOutAlgorithm`/`AbstractDissipationMelt` above -- so only the selected closure's fields
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
  gradient by default (matching Kori-ULB's own `SubWaterFlux.m`) rather than the local one -- see its
  `gradient_convention` type parameter. Kept for direct comparison against Kori-ULB's `Wd`/SHAKTI's
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
MeanGradient()` by default (matching Kori-ULB's own `SubWaterFlux.m`). See
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

Trait controlling which opening terms `update_N_inf!` includes in Eq. (5b)/(6a) of Kazmierczak et al
2024, and which `Q_c` `update_H!` uses in Eq. (9) -- i.e. whether the model is left free to switch
between efficient and inefficient drainage (the paper's default behaviour) or is forced to produce
one or the other throughout, as in the paper's Sect. 3.2. Stored as a type parameter and resolved by
multiple dispatch, the same pattern as `AbstractDissipationMelt` and `AbstractPsiOutAlgorithm` above.

- `BothDrainage()` (default): both the sliding-over-obstacles opening term (||v_b||h_b, inefficient)
  and the melt-driven opening term (Q_w||grad(phi)||/(rho_i*L_w), efficient) are included in Eq.
  (5b), and `update_H!` uses the model's configured `Q_c` unchanged -- the paper's normal automatic
  switch between drainage types based on the local water flux.
- `EfficientOnly()`: the sliding opening term is dropped from Eq. (5b)/(6a), and `update_H!` is given
  `Q_c = 0` so that Eq. (9)'s exp(-Q/Q_c) -> 0 for any Q > 0, forcing the soft-bed conduit geometry to
  the canal/efficient case (H_soft -> H_0) everywhere.
- `InefficientOnly()`: the melt opening term is dropped from Eq. (5b), and `update_H!` is given
  `Q_c = Inf` so that exp(-Q/Q_c) -> 1, forcing the soft-bed conduit geometry to the inter-clastic-
  film/inefficient case (H_soft -> sqrt(S_inf)/F_till) everywhere.

Note the Q_c -> 0 / Q_c -> Inf assignment above is the *corrected* version of the paper's own Sect.
3.2 description (as published, it prescribes Q_c = Inf for the entirely-efficient case and Q_c = 0
for the entirely-inefficient case); the authors confirmed by email that this is a typo and the two
are swapped, which also matches Eq. (9) directly -- the exp(-Q/Q_c) limits above only work out this
way around.
"""
abstract type AbstractDrainageMode end
struct BothDrainage    <: AbstractDrainageMode end
struct EfficientOnly   <: AbstractDrainageMode end
struct InefficientOnly <: AbstractDrainageMode end


"""
$(TYPEDSIGNATURES)

Physical constants and solver configuration for `KazmierczakHydroModel` -- everything that is fixed
once the model is constructed and never touched again during a solve. Split out from
`KazmierczakWorkspace` (the mutable array buffers) so the two can be reasoned about and constructed
independently instead of interleaved as 40 flat fields on one struct; see `KazmierczakHydroModel`
for how the split is made transparent to callers.
"""
struct KazmierczakParams{T <: AbstractFloat, D <: AbstractDissipationMelt, L <: AbstractSlidingLaw, P <: AbstractPsiOutAlgorithm, WT <: AbstractWaterThicknessAlgorithm, M <: AbstractDrainageMode}

    rho_w           ::T    # Density of fresh water [kg/m3]
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
    Wmin            ::T    # Minimum subglacial water layer thickness [m]; only applied by the sheet-flow water_thickness_algorithm closures (DarcyWeisbachThickness, LaminarThickness). Defaults to 0.0 (no floor) -- pass 1e-8 for KORI-ULB's own Wdmin if you want that bound back
    Wmax            ::T    # Maximum subglacial water layer thickness [m]; only applied by the sheet-flow water_thickness_algorithm closures (DarcyWeisbachThickness, LaminarThickness). Defaults to Inf (no ceiling) -- pass 0.015 for KORI-ULB's own Wdmax if you want that bound back
    water_thickness_algorithm ::WT # ArealConduitThickness()/DarcyWeisbachThickness()/LaminarThickness(): which closure update_W! uses to compute state.W
    longcoupwater   ::T    # Longitudinal coupling factor for the stress-gradient coupling smoothing of the geometric potential gradients -- KORI-ULB's own internal parameter, unchanged in meaning. Set via the constructor's `coupling_length_kamb86` keyword (longcoupwater = coupling_length_kamb86 / 2), not directly -- see that keyword's docstring for why
    sigmat          ::T    # Effective pressure lower bound as fraction of overburden pressure. Defaults to 0.0 (no floor) -- pass 0.02 for KORI-ULB's own value if you want that bound back
    q_min           ::T    # Minimum allowed value for the distributed water flux
    q_max           ::T    # Maximum allowed value for the distributed water flux. Defaults to Inf (no ceiling) -- pass perYear2perSecond(1e5) for KORI-ULB's own SubWaterFlux.m numerical-stability cap if you want that bound back
    fill_iters      ::Int  # How many iterations to perform for the filling of local minima of the geometric potential phi0 (JacobiFill only)
    fill_algorithm  ::AbstractFillAlgorithm  # JacobiFill()/LowestNeighbourFill()/PriorityFloodFill(): how potential_filling! removes pits (dispatched once per update_q! call)
    routing_scheme  ::AbstractRoutingScheme  # Warner() (default), GDSWarner() (original K24) or another Le Brocq et al. (2006) scheme -- see AbstractRoutingScheme
    q_conversion    ::AbstractQConversion    # QFromOutflow() or QFromFaceAverage() (default follows routing_scheme) -- see AbstractQConversion
    dissipation_discretization ::AbstractDissipationDiscretization  # CellCentredDissipation() or FaceDissipation() (default follows routing_scheme) -- see AbstractDissipationDiscretization
    friction_discretization ::AbstractFrictionDiscretization  # CellCentredFriction() (default) or StaggeredFriction(ux, uy) -- see AbstractFrictionDiscretization
    max_psi_out_calls ::Int  # Safety cap on the number of accumulate_psi_out! calls in one update_psi_out! sweep, mirroring KORI-ULB's funcnt <= 5e4 cap in DpareaWarGds.m
    psi_out_algorithm ::P  # RecursivePsiOut() or IterativePsiOut(): which flow-routing implementation resolve_q! uses to compute psi_out each sweep
    max_dissipation_iters ::Int  # Safety cap on the number of Picard iterations for the dissipation melt term in update_q!
    dissipation_rtol       ::T    # Relative tolerance on q for the dissipation melt term's Picard iteration to be considered converged
    dissipation_melt        ::D    # DissipationMeltOn() or DissipationMeltOff(): whether update_q! includes the |q * grad(phi0)| / L_w term
    dissipation_verbose     ::Bool # Whether the dissipation melt term's Picard iteration logs its timing/convergence summary each call
    sliding_law             ::L    # AbstractSlidingLaw instance used to compute tau_b for the frictional-heating term tau_b*v_b in mdot
    max_qN_iters      ::Int  # Safety cap on the number of Picard iterations for the (q, N) loop when sliding_law is N-dependent
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
"""
struct KazmierczakWorkspace{A, R <: RoutingTape}

    # Geometric potential
    phi0                   ::A  # True geometric potential rho_i*g*h + rho_w*g*b [Pa]; used for N and for the local gradient magnitude
    phi0_filled            ::A  # phi0 with local minima filled; used only to route water (flow direction), never for N/S_inf/dissipation [Pa]
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
    G            ::A  # Geothermal heat flux into the bed [W/m2]
    q_T          ::A  # Conductive heat flux from the bed into the ice [W/m2]
    i_eb         ::A  # Water reaching the bed from above, not melted there (i_eb of Sommers et al 2018: drained englacial water, surface input) [kg/m2/s]
    Q_b          ::A  # Frictional heat tau_b . u_b [W/m2], from sliding_law and friction_discretization (0 with NoFrictionSlidingLaw)
    Q_diss       ::A  # Heat dissipated by the water flow |q . grad(phi0)| [W/m2] (0 when dissipation_melt is off)
    mdot_fixed   ::A  # Fixed part of the basal melt rate, (G - q_T)/L_w [kg/m2/s]; set by set_basal_terms!
    mdot_total   ::A  # Water source routed by the flux solver, mdot_fixed + (Q_b + Q_diss)/L_w + i_eb [kg/m2/s]
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
struct KazmierczakHydroModel{T <: AbstractFloat, A, D <: AbstractDissipationMelt, L <: AbstractSlidingLaw, P <: AbstractPsiOutAlgorithm, WT <: AbstractWaterThicknessAlgorithm, M <: AbstractDrainageMode} <: AbstractHydroModel
    params    ::KazmierczakParams{T, D, L, P, WT, M}
    workspace ::KazmierczakWorkspace{A, RoutingTape{T}}
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
    mdot_total = mdot_fixed + (Q_b + Q_diss) / L_w + i_eb   (water source routed)

`mdot_fixed + (Q_b + Q_diss)/L_w` is the basal melt rate; `Q_b` and `Q_diss` depend on the water flux
(through N and q), so they are recomputed every Picard sweep.

`G` and `q_T` are given (positional, [W/m^2]); `Q_b = tau_b . u_b` comes from `sliding_law` and
`friction_discretization` (zero for `NoFrictionSlidingLaw`, the default); `Q_diss = |q . grad(phi0)|`
is added when `dissipation_melt` is on; `i_eb` [kg/m^2/s] is water reaching the bed from above that
was not melted there (drained englacial water, surface input), zero by default. As in Sommers et al
2018, `i_eb` is a source of water but not part of the melt rate. Each term is kept as its own field
(`model.G`, `model.q_T`, `model.Q_b`, `model.Q_diss`, `model.i_eb`; `model.mdot_fixed` holds only the fixed
part `(G - q_T)/L_w`), so a
coupled model can exchange terms rather than a melt rate. Update the given terms between solves with
[`set_basal_terms!`](@ref).

See the `AbstractSlidingLaw` docstring in model.jl for the available laws and `resolve_q!` in
water_flux.jl for how N-dependent laws widen the existing dissipation-melt Picard loop into a joint
(q, N) fixed point.

Keyword names for the paper's symbols (the `KazmierczakParams` fields and `model.<field>` keep the paper symbol):
`glen_n` (n, Glen's flow law exponent), `flux_W_exponent` (alpha), `flux_grad_exponent` (beta),
`bed_bump_height` (h_b, typical bed obstacle height [m]), `H0_efficient` (H_0, canal thickness for soft bed
deformation [m]), `conduit_spacing` (l_c, distance between conduits [m]), `water_viscosity` (eta_w, dynamic
viscosity of water [Pa s]). The (q, N) Picard loop is controlled by `max_qN_iters`, `qN_rtol` and `qN_verbose`.

The `psi_out_algorithm` keyword (`TapedPsiOut()` by default) selects which flow-routing
implementation `resolve_q!` uses each sweep to compute psi_out -- see the `AbstractPsiOutAlgorithm`
docstring above for the `RecursivePsiOut`/`IterativePsiOut` speed-vs-stack-robustness trade-off.

The `water_thickness_algorithm` keyword (`DarcyWeisbachThickness()` by default) selects which closure
`update_W!` uses to compute the reportable `state.W` -- see the `AbstractWaterThicknessAlgorithm`
docstring above for the three available closures (`ArealConduitThickness`, `DarcyWeisbachThickness`,
`LaminarThickness`) and how they differ.

`Wmin`/`Wmax`/`sigmat`/`q_max` all default to no-op values (`0.0`/`Inf`/`0.0`/`Inf` respectively) --
i.e. nothing is clamped unless you ask for it. KORI-ULB's own values (`1e-8`/`0.015`/`0.02`/
`perYear2perSecond(1e5)`) are still available, just not silently applied: pass them explicitly if you
want that fidelity, or if you hit the numerical-stability edge cases they exist to guard against. Two
of the four are genuinely load-bearing for robustness, not just KORI-ULB fidelity, so removing them by
default is a real tradeoff, not a free one:
- `q_max = Inf` removes the cap Kori-ULB's own `SubWaterFlux.m` applies "for numerical stability"
  (per its own comment) -- without it, a sufficiently pathological cell (steep local gradient, small
  drainage area, high local melt) can drive `q` arbitrarily large, propagating into `S_inf`/`H`/`N_inf`.
- `sigmat = 0.0` removes `N_inf`'s floor at `sigmat*Po`; at a genuinely flat, zero-potential-gradient
  cell (`phi0 = 0` as well as `N_inf = 0`), `update_N!`'s `erf(sqrt(pi)*phi0/(2*N_inf))*N_inf` term
  hits a `0/0` inside the `erf` argument (`NaN`), whereas a nonzero `sigmat` floor keeps `N_inf` away
  from exactly zero.

`Wmin`/`Wmax` are lower risk to leave unclamped: `state.W` is a terminal diagnostic field (see
`update_W!`'s docstring in water_flux.jl) that feeds back into nothing else in the model, so an
unclamped value there can't itself destabilize the rest of a solve.

The constructor has no `longcoupwater` keyword directly -- pass `coupling_length_kamb86` instead.
`longcoupwater` itself (the field on `KazmierczakParams`, and the quantity `update_smoothed_potential_gradients!`
in water_flux.jl actually uses) keeps its original meaning unchanged, matching KORI-ULB's own internal
parameter of the same name; only the constructor's *public entry point* to it has moved. This is because
`longcoupwater` alone was never Kamb & Echelmeyer (1986)'s number directly: the kernel's true (2D
area-weighted) effective coupling length works out to `2 * longcoupwater * mean_ice_thickness`, not
`longcoupwater * mean_ice_thickness` -- see the derivation in `update_smoothed_potential_gradients!`
(water_flux.jl) and the field comment on `KazmierczakParams`. `coupling_length_kamb86` is defined so it
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
    Wmin          = 0.0,                          # Minimum subglacial water layer thickness [m]; no floor by default -- pass 1e-8 for KORI-ULB's own Wdmin
    Wmax          = Inf,                          # Maximum subglacial water layer thickness [m]; no ceiling by default -- pass 0.015 for KORI-ULB's own Wdmax
    water_thickness_algorithm = DarcyWeisbachThickness(), # ArealConduitThickness()/DarcyWeisbachThickness()/LaminarThickness(): which closure update_W! uses to compute state.W
    coupling_length_kamb86 = nothing,             # Kamb & Echelmeyer (1986) stress-gradient-coupling length, as a direct multiple of mean ice thickness (>= 0; 0 disables the smoothing); no safe no-op default, so leaving this unspecified defaults to 10.0 and emits a @warn explaining how to choose it
    sigmat        = 0.0,                          # Effective pressure lower bound as fraction of overburden pressure; no floor by default -- pass 0.02 for KORI-ULB's own value
    q_min         = 0.0,                          # Minimum allowed value for the distributed water flux
    q_max         = Inf,                          # Maximum allowed value for the distributed water flux; no ceiling by default -- pass perYear2perSecond(1e5) for KORI-ULB's own SubWaterFlux.m numerical-stability cap
    fill_iters    = 10,                           # How many iterations to perform for the filling of local minima of the geometric potential phi0 (JacobiFill only)
    fill_algorithm = nothing,                     # JacobiFill()/LowestNeighbourFill()/PriorityFloodFill(); nothing = the routing scheme's partner (PriorityFloodFill for Warner) -- see AbstractFillAlgorithm/AbstractRoutingScheme
    routing_scheme = Warner(),                    # Flux-routing scheme (Le Brocq et al. 2006): Warner() (default)/GDSWarner() (original K24)/Quinn()/Tarboton()/ModifiedTarboton()/GDSTarboton() -- see AbstractRoutingScheme
    q_conversion = nothing,                       # QFromOutflow()/QFromFaceAverage(); nothing = QFromFaceAverage for Warner, QFromOutflow otherwise -- see AbstractQConversion
    dissipation_discretization = nothing,         # CellCentredDissipation()/FaceDissipation(); nothing = FaceDissipation for Warner, CellCentredDissipation otherwise -- see AbstractDissipationDiscretization
    friction_discretization = CellCentredFriction(),       # CellCentredFriction()/StaggeredFriction(ux_acx, uy_acy; quadrature): frictional heat from C-grid velocities -- see AbstractFrictionDiscretization
    max_psi_out_calls = 100_000,                   # Safety cap on the number of accumulate_psi_out! calls in one update_psi_out! sweep, mirroring KORI-ULB's funcnt <= 5e4 cap
    psi_out_algorithm = TapedPsiOut(),            # TapedPsiOut()/RecursivePsiOut()/IterativePsiOut()/TopologicalPsiOut(): which flow-routing implementation resolve_q! uses to compute psi_out
    max_dissipation_iters = 20,                   # Safety cap on the number of Picard iterations for the dissipation melt term in update_q!
    dissipation_rtol       = 1e-12,                # Relative tolerance on q for the dissipation melt term's Picard iteration to be considered converged
    dissipation_melt        = true,                # Whether update_q! includes the |q * grad(phi0)| / L_w term
    dissipation_verbose     = true,                # Whether the dissipation melt term's Picard iteration logs its timing/convergence summary each call
    sliding_law         = NoFrictionSlidingLaw(),          # AbstractSlidingLaw instance used to compute tau_b for the frictional-heating term tau_b*v_b in mdot
    max_qN_iters  = 20,                      # Safety cap on the number of Picard iterations for the (q, N) loop when sliding_law is N-dependent
    qN_rtol       = 1e-8,                    # Relative tolerance on q and N for the (q, N) Picard iteration to be considered converged
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
    Wmin          = T(Wmin)
    Wmax          = T(Wmax)

    # coupling_length_kamb86 has no numerically-safe "no-op" default the way Wmin/Wmax/q_max/sigmat do
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

    sigmat        = T(sigmat)
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
    dissipation_rtol       = T(dissipation_rtol)
    dissipation_melt_trait = dissipation_melt ? DissipationMeltOn() : DissipationMeltOff()
    max_qN_iters  = Int(max_qN_iters)
    qN_rtol       = T(qN_rtol)
    sliding_law         = convert_sliding_law(T, sliding_law)

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

    # Water flux
    visited    = alloc_field(grid)
    h          = alloc_field(grid)
    G          = alloc_field(grid, G_in)
    q_T        = alloc_field(grid, q_T_in)
    i_eb = i_eb === nothing ? alloc_field(grid) : alloc_field(grid, i_eb)
    Q_b        = alloc_field(grid)
    Q_diss     = alloc_field(grid)
    mdot_fixed = alloc_field(grid)
    @. mdot_fixed = (G - q_T) / L_w
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
        rho_w, rho_i, g, L_w, n, h_b, alpha, beta, f, F_till, Q_c, drainage_mode, H_0, l_c, K, eta_w, Wmin, Wmax, water_thickness_algorithm, longcoupwater, sigmat, q_min, q_max, fill_iters, fill_algorithm, routing_scheme, q_conversion, dissipation_discretization, friction_discretization,
        max_psi_out_calls, psi_out_algorithm, max_dissipation_iters, dissipation_rtol, dissipation_melt_trait, dissipation_verbose,
        sliding_law, max_qN_iters, qN_rtol, qN_verbose
)

    workspace = KazmierczakWorkspace(
        phi0, phi0_filled, phi0_tmp, minus_grad_phi0_x, minus_grad_phi0_y,
        abs_grad_phi0, minus_grad_phi0_sx, minus_grad_phi0_sy, abs_grad_phi0_s,
        visited, h, G, q_T, i_eb, Q_b, Q_diss, mdot_fixed, mdot_total, psi_out, corfac, q, q_prev, tau_b, N_prev, RoutingTape{T}(grid.Nx, grid.Ny), Tuple{Int32, Int32}[], Int32[], zeros(Int32, grid.Nx, grid.Ny),
        Q, kappa, abs_v_b, A_visc, S_inf, H_hard, H_soft, H, N_inf, Po
    )

    return KazmierczakHydroModel(params, workspace)

end



"""
$(TYPEDSIGNATURES)

Update the given terms of the water source between solves: the geothermal heat flux `G` and the
conductive heat flux into the ice `q_T` [W/m^2], and the water from above `i_eb` [kg/m^2/s]. Any keyword left
`nothing` keeps its current value. Recomputes the fixed part of the source,
`model.mdot_fixed = (G - q_T)/L_w`; the frictional and dissipation heat are recomputed by the
solver itself. This is the only way to change the source: the melt rate is always built from terms.
"""
function set_basal_terms!(model::KazmierczakHydroModel; G = nothing, q_T = nothing, i_eb = nothing)
    G            === nothing || (model.G            .= G)
    q_T          === nothing || (model.q_T          .= q_T)
    i_eb === nothing || (model.i_eb .= i_eb)
    @. model.mdot_fixed = (model.G - model.q_T) / model.L_w
    return model
end
