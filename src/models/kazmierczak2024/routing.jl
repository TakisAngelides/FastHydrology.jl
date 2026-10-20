#################################
# Model: Kazmierczak et al 2024 #
#################################

# Flux-routing schemes of Le Brocq, Payne & Siegert (2006, Computers & Geosciences 32, 1780-1795,
# https://doi.org/10.1016/j.cageo.2006.05.003), selected by `model.routing_scheme` -- see
# `AbstractRoutingScheme` below. The default `GDSWarner()` keeps its original code path
# (`record_routing_tape_kernel!` reading the smoothed gradient directly); every other scheme is
# expressed as per-cell outflow fractions `W[d, i, j]` to the neighbour in direction `d`
# (`ROUTE_OFFSETS`), which the generic tape recorder below and the face-flux diagnostics use.


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

`fill_algorithm`, `q_conversion`, `dissipation_discretization` and `gradient_discretization` default
to `nothing`, meaning "the natural partner of `routing_scheme`": `PriorityFloodFill()`,
`QFromFaceAverage()`, `FaceDissipation()` and `FaceGradient()` for `Warner`; `JacobiFill()`,
`QFromOutflow()`, `CellCentredDissipation()` and `CentredGradient()` for the GDS schemes (so
`routing_scheme = GDSWarner()` alone reproduces the original K24 set-up exactly); `PriorityFloodFill()`,
`QFromOutflow()`, `CellCentredDissipation()` and `CentredGradient()` for the other 8-neighbour
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


const ROUTE_OFFSETS  = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))
const ROUTE_OPPOSITE = (2, 1, 4, 3, 8, 7, 6, 5)

# Tarboton (1997) triangular facets as (cardinal, diagonal) direction pairs of ROUTE_OFFSETS.
const TARBOTON_FACETS = ((2, 6), (2, 8), (4, 8), (4, 7), (1, 7), (1, 5), (3, 5), (3, 6))

@inline in_domain(i, j, Nx, Ny) = 1 <= i <= Nx && 1 <= j <= Ny


"""
$(TYPEDSIGNATURES)

Whether the solve needs face fluxes each sweep: for `QFromFaceAverage` or face-assembled
dissipation or sensible heat (`FaceDissipation` with either term on).
"""
needs_face_fluxes(model::AbstractKazmierczakHydroModel) =
    model.q_conversion isa QFromFaceAverage ||
    ((model.dissipation_melt isa DissipationMeltOn || model.sensible_heat isa SensibleHeatOn) &&
     model.dissipation_discretization isa FaceDissipation)


"""
$(TYPEDSIGNATURES)

Build the routing potential and flow directions for `model.routing_scheme`, called once per
`update_q!` after `update_phi0!`.

- GDS schemes (`GDSWarner`, `GDSTarboton`): the original K24 pipeline -- fill phi0, take its gradient,
  smooth the gradient components with the Kamb & Echelmeyer (1986) kernel, normalised by the weight
  `model.phi_weight` (see `update_smoothed_potential_gradients!`).
- Potential schemes (`Warner`, `Quinn`, `Tarboton`, `ModifiedTarboton`): smooth the potential itself
  with the same kernel, then fill it (so the filling acts on the surface the water actually follows).
  Smoothing commutes with differentiation, so this is the same stress-gradient coupling, but the
  result stays the gradient of one surface. `minus_grad_phi0_sx/sy` then hold the (unsmoothed)
  centred gradient of that surface, used for the psi -> q conversion direction (Le Brocq Eq. 9) and
  by `ModifiedTarboton`. The smoothing is normalised by the weight `model.phi_weight` (see
  `smooth_potential!`), so cells without a meaningful potential do not pull it down near the edge
  of the ice.

`abs_grad_phi0` (used by N and the dissipation melt) is always the gradient of the true potential.
Both magnitudes (and `abs_grad_phi0_s` wherever it is unsmoothed) use `model.gradient_discretization`'s stencil.
"""
function prepare_routing!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, scheme::Union{GDSWarner, GDSTarboton})
    potential_filling!(model, grid, state)
    update_potential_gradients!(model, grid)
    update_smoothed_potential_gradients!(model, grid, state)
    if !(scheme isa GDSWarner) || needs_face_fluxes(model)
        compute_routing_weights!(model, grid, state)
    end
    return nothing
end

function prepare_routing!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::Union{Warner, Quinn, Tarboton, ModifiedTarboton})
    if model.longcoupwater == 0.0
        model.phi0_filled .= model.phi0
    else
        kernel = coupling_kernel(model, grid, max(masked_mean(grid, state.h, state.mask), 10.0))
        smooth_potential!(model, grid, kernel)
    end
    potential_filling!(model, grid, state; from_true_potential = false)
    update_potential_gradients!(model, grid)
    model.minus_grad_phi0_sx .= model.minus_grad_phi0_x
    model.minus_grad_phi0_sy .= model.minus_grad_phi0_y
    update_unsmoothed_gradient_magnitude!(model, grid, model.gradient_discretization)
    compute_routing_weights!(model, grid, state)
    return nothing
end


"""
$(TYPEDSIGNATURES)

Smooth the true potential with the coupling `kernel` into `model.phi0_filled`, as the weighted
average `K * (phi0 w) / K * w` with `w = model.phi_weight` (set by `update_phi0!`). A plain `K * phi0`
averages in the cells outside the grounded ice, whose phi0 can be far below the potential under the
ice (0 for the `h = b = 0` padding around a rasterised mesh, against ~10 MPa just inside): every cell
within a kernel radius of that edge is pulled down, the more so the more of the kernel lies outside.
That creates spurious maxima of the smoothed potential -- drainage divides with no upstream water,
hence low `q` and high `N`: a band about one kernel radius inside the boundary for small kernels,
contracting to a single hot spot for kernels comparable to the domain. Normalising by `K * w`
averages real ice only. Cells outside `w` within reach of the kernel get the same weighted average
(an extrapolation, so water reaching the edge sees no artificial cliff); farther ones keep phi0.
"""
function smooth_potential!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, kernel)
    w, ws = model.phi_weight, model.phi_weight_s
    @. model.phi0_tmp = model.phi0 * w      # scratch; potential_filling! resets phi0_tmp afterwards
    convolve!(grid, model.phi0_filled, model.phi0_tmp, kernel)
    convolve!(grid, ws, w, kernel)
    tol = sqrt(eps(eltype(ws)))
    @. model.phi0_filled = ifelse(ws > tol, model.phi0_filled / max(ws, tol), model.phi0)
    return nothing
end


"""
$(TYPEDSIGNATURES)

Fill `model.routing_tape.w8[d, i, j]`, the fraction of grounded cell `(i, j)`'s outflow sent to its
neighbour in direction `d` (`ROUTE_OFFSETS`), for `model.routing_scheme`. Fractions of a cell sum to 1,
or to 0 for a cell with nowhere downhill to send water (a sink; see `AbstractRoutingScheme`). Water
sent to a non-grounded neighbour, or (GDS schemes) off the domain edge, leaves the system.
"""
function compute_routing_weights!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)
    routing_weights_kernel!(model.routing_tape.w8, model.routing_scheme, model.phi0_filled,
                            model.minus_grad_phi0_sx, model.minus_grad_phi0_sy, state.mask, grid.Nx, grid.Ny, grid.dx, grid.dy)
    return nothing
end

function routing_weights_kernel!(W, scheme, phi, sx, sy, mask, Nx, Ny, dx, dy)
    fill!(W, zero(eltype(W)))
    @inbounds for j in 1:Ny, i in 1:Nx
        mask[i, j] == 1.0 || continue
        cell_weights!(W, scheme, i, j, phi, sx, sy, Nx, Ny, dx, dy)
    end
    return nothing
end

# GDS-Warner (current K24/KORI scheme): the smoothed gradient's component towards each of the 4
# neighbours, i.e. at most two neighbours receive water. Identical to `routing_weight` as used by the
# recursive routing.
@inline function cell_weights!(W, ::GDSWarner, i, j, phi, sx, sy, Nx, Ny, dx, dy)
    epsT = eps(eltype(W))
    @inbounds for d in 1:4
        di, dj = ROUTE_OFFSETS[d]
        W[d, i, j] = max(zero(eltype(W)), routing_weight(sx[i, j], sy[i, j], di, dj, dx, dy, epsT))
    end
    return nothing
end

# Warner (Budd & Warner 1996; Le Brocq Eq. 8): outflow shared among the downhill 4-neighbours in
# proportion to the potential drop to each. The drop is weighted by (face length / distance), i.e.
# the face-normal gradient times the face length, which is Eq. 8 exactly on a square grid.
@inline function cell_weights!(W, ::Warner, i, j, phi, sx, sy, Nx, Ny, dx, dy)
    p = phi[i, j]
    tot = zero(eltype(W))
    @inbounds for d in 1:4
        di, dj = ROUTE_OFFSETS[d]
        ni, nj = i + di, j + dj
        in_domain(ni, nj, Nx, Ny) || continue
        drop = p - phi[ni, nj]
        if drop > 0
            v = drop * (d <= 2 ? dy / dx : dx / dy)
            W[d, i, j] = v
            tot += v
        end
    end
    if tot > 0
        @inbounds for d in 1:4
            W[d, i, j] /= tot
        end
    end
    return nothing
end

# Quinn (Quinn et al. 1991): as Warner, over all 8 neighbours. `original = false`: shares proportional
# to the potential drop, as written in Le Brocq Eq. 8. `original = true`: Quinn et al.'s own weights,
# slope times effective contour length, tan(beta_d) * L_d = (drop / delta_d) * L_d, with L = 0.5 Delta
# (cardinal) and sqrt(2)/4 Delta ~ 0.354 Delta (diagonal); for rectangular cells the cardinal contour
# is half the face the flow crosses and the diagonal one uses sqrt(dx dy).
@inline function cell_weights!(W, scheme::Quinn, i, j, phi, sx, sy, Nx, Ny, dx, dy)
    p = phi[i, j]
    tot = zero(eltype(W))
    ddiag = hypot(dx, dy)
    Ldiag = sqrt(2) / 4 * sqrt(dx * dy)
    @inbounds for d in 1:8
        di, dj = ROUTE_OFFSETS[d]
        ni, nj = i + di, j + dj
        in_domain(ni, nj, Nx, Ny) || continue
        drop = p - phi[ni, nj]
        if drop > 0
            v = if !scheme.original
                drop
            elseif d <= 2
                drop / dx * (dy / 2)
            elseif d <= 4
                drop / dy * (dx / 2)
            else
                drop / ddiag * Ldiag
            end
            W[d, i, j] = v
            tot += v
        end
    end
    if tot > 0
        @inbounds for d in 1:8
            W[d, i, j] /= tot
        end
    end
    return nothing
end

# Tarboton (1997) D-infinity: steepest downhill direction over the 8 triangular facets around the
# cell; the outflow goes to the facet's cardinal and diagonal neighbours in proportion to how close
# the flow angle is to each.
@inline function cell_weights!(W, ::Tarboton, i, j, phi, sx, sy, Nx, Ny, dx, dy)
    e0 = phi[i, j]
    best_s = zero(eltype(W)); best = 0; best_r = zero(eltype(W)); best_amax = one(eltype(W))
    @inbounds for f in 1:8
        c1, c2 = TARBOTON_FACETS[f]
        di1, dj1 = ROUTE_OFFSETS[c1]; di2, dj2 = ROUTE_OFFSETS[c2]
        (in_domain(i + di1, j + dj1, Nx, Ny) && in_domain(i + di2, j + dj2, Nx, Ny)) || continue
        e1 = phi[i + di1, j + dj1]; e2 = phi[i + di2, j + dj2]
        d1, d2 = c1 <= 2 ? (dx, dy) : (dy, dx)
        s1 = (e0 - e1) / d1
        s2 = (e1 - e2) / d2
        amax = atan(d2, d1)
        r = atan(s2, s1)
        s = hypot(s1, s2)
        if r < 0
            r = zero(r); s = s1
        elseif r > amax
            r = amax; s = (e0 - e2) / hypot(d1, d2)
        end
        if s > best_s
            best_s, best, best_r, best_amax = s, f, r, amax
        end
    end
    best == 0 && return nothing
    c1, c2 = TARBOTON_FACETS[best]
    @inbounds W[c2, i, j] = best_r / best_amax
    @inbounds W[c1, i, j] = 1 - best_r / best_amax
    return nothing
end

# Modified Tarboton / GDS-Tarboton (Le Brocq Sec. 3): a single flow direction -- from the local
# 4-neighbour slope of the routing potential (Modified) or from the smoothed gravitational driving
# stress components (GDS) -- split between the two of the 8 neighbours whose directions bracket it,
# in proportion to the angles. `sx/sy` hold the corresponding (minus) gradient for each scheme.
@inline cell_weights!(W, ::Union{ModifiedTarboton, GDSTarboton}, i, j, phi, sx, sy, Nx, Ny, dx, dy) =
    angle_split!(W, i, j, sx[i, j], sy[i, j], dx, dy)

@inline function angle_split!(W, i, j, fx, fy, dx, dy)
    (fx == 0 && fy == 0) && return nothing
    a = atan(dy, dx)
    pT = typeof(a)(pi)   # pi in the grid's float type: bare `pi`/`2pi`/`0.0` would promote a Float32 model to Float64
    angs = (zero(pT), a, pT / 2, pT - a, pT, pT + a, 3pT / 2, 2pT - a)
    dirs = (2, 8, 4, 7, 1, 5, 3, 6)
    theta = mod(atan(fy, fx), 2pT)
    k = 8
    @inbounds for m in 1:7
        if angs[m] <= theta < angs[m + 1]
            k = m
            break
        end
    end
    lo = angs[k]
    hi = k == 8 ? 2pT : angs[k + 1]
    frac = (theta - lo) / (hi - lo)
    @inbounds W[dirs[k], i, j] += 1 - frac
    @inbounds W[dirs[k == 8 ? 1 : k + 1], i, j] += frac
    return nothing
end


"""
$(TYPEDSIGNATURES)

Generic counterpart of `record_routing_tape_kernel!` for schemes expressed as outflow fractions `W`
(everything except the default `GDSWarner`): the same explicit-stack traversal, in the same
neighbour order, over the first `ndirs` directions of `ROUTE_OFFSETS`. Cell `c` receives
`psi_out[n] * W[opposite(d), n]` from its neighbour `n` in direction `d`.
"""
function record_routing_tape_weights_kernel!(tape::RoutingTape{T}, mask, W, ndirs, Nx, Ny, max_calls) where {T}

    for v in (tape.dst_i, tape.dst_j, tape.src_i, tape.src_j)
        empty!(v)
    end
    empty!(tape.w)

    visited = zeros(Bool, Nx, Ny)
    stack = NTuple{3, Int32}[]
    call_count = 0
    hit_cap = false
    done = Int32(ndirs + 1)

    @inbounds for j in 1:Ny, i in 1:Nx

        (mask[i, j] == 1.0 && !visited[i, j]) || continue

        push!(stack, (Int32(i), Int32(j), Int32(0)))

        while !isempty(stack)

            si, sj, sk = stack[end]

            if sk == 0

                visited[si, sj] = true
                call_count += 1
                if call_count > max_calls
                    hit_cap = true
                    push_tape_op!(tape, si, sj, Int32(0), Int32(0), zero(T))
                    pop!(stack)
                else
                    stack[end] = (si, sj, Int32(1))
                end

            elseif sk < done

                di, dj = ROUTE_OFFSETS[sk]
                ni, nj = si + di, sj + dj
                stack[end] = (si, sj, sk + Int32(1))

                in_domain(ni, nj, Nx, Ny) || continue

                w = W[ROUTE_OPPOSITE[sk], ni, nj]
                (w > 0 && mask[ni, nj] == 1.0) || continue

                if visited[ni, nj]
                    push_tape_op!(tape, si, sj, Int32(ni), Int32(nj), T(w))
                else
                    stack[end] = (si, sj, sk)
                    push!(stack, (Int32(ni), Int32(nj), Int32(0)))
                end

            else
                push_tape_op!(tape, si, sj, Int32(0), Int32(0), zero(T))
                pop!(stack)
            end
        end
    end

    return hit_cap

end


"""
$(TYPEDSIGNATURES)

Net volume flux through every cell face [m3/s] from the current `psi_out` and outflow fractions
(4-neighbour schemes only): `Fx[a, j]` is the flux through the face between cells `(a-1, j)` and
`(a, j)`, positive in +x; `Fy[i, b]` likewise in +y. Boundary faces carry what leaves the domain.

With `FaceDissipation` it also assembles the dissipation melt [kg/m2/s] into `tape.diss`: across
each face the water releases `F * (phi_upstream - phi_downstream)` [W] of the true potential, and
each cell gets half of the energy of each of its 4 faces (the face-to-centre average), divided by
its area and L_w. Boundary faces towards the domain edge have no neighbour potential and are
skipped, and so are faces to a cell outside `model.phi_weight` (see `update_phi0!`): the water leaves
the grounded system there, and the drop to that cell's phi0 is not energy released under the ice
(at the edge of a rasterised mesh it is the whole overburden). The result is signed: water pushed up
the true potential (out of a filled hollow, or uphill under the smoothed GDS direction) gives a
negative contribution.
"""
function update_face_fluxes!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)
    t = model.routing_tape
    face_fluxes_kernel!(t.Fx, t.Fy, t.diss, model.psi_out, t.w8, state.mask, model.phi_weight, model.phi0, grid.Nx, grid.Ny, grid.dx, grid.dy,
                        model.L_w, model.dissipation_discretization isa FaceDissipation)
    return nothing
end

function face_fluxes_kernel!(Fx, Fy, diss, psi, W, mask, w, phi, Nx, Ny, dx, dy, L_w, do_diss)
    z = zero(eltype(Fx))
    @inbounds for j in 1:Ny, a in 1:Nx+1
        f = z
        (a > 1 && mask[a - 1, j] == 1.0) && (f += psi[a - 1, j] * W[2, a - 1, j])
        (a <= Nx && mask[a, j] == 1.0) && (f -= psi[a, j] * W[1, a, j])
        Fx[a, j] = f
    end
    @inbounds for b in 1:Ny+1, i in 1:Nx
        f = z
        (b > 1 && mask[i, b - 1] == 1.0) && (f += psi[i, b - 1] * W[4, i, b - 1])
        (b <= Ny && mask[i, b] == 1.0) && (f -= psi[i, b] * W[3, i, b])
        Fy[i, b] = f
    end
    do_diss || return nothing
    @inbounds for j in 1:Ny, i in 1:Nx
        if mask[i, j] != 1.0
            diss[i, j] = z
            continue
        end
        P = z
        (i > 1  && w[i - 1, j] == 1) && (P += Fx[i, j]     * (phi[i - 1, j] - phi[i, j]))
        (i < Nx && w[i + 1, j] == 1) && (P += Fx[i + 1, j] * (phi[i, j] - phi[i + 1, j]))
        (j > 1  && w[i, j - 1] == 1) && (P += Fy[i, j]     * (phi[i, j - 1] - phi[i, j]))
        (j < Ny && w[i, j + 1] == 1) && (P += Fy[i, j + 1] * (phi[i, j] - phi[i, j + 1]))
        diss[i, j] = P / 2 / (dx * dy * L_w)
    end
    return nothing
end


"""
$(TYPEDSIGNATURES)

Convert routed flux to the distributed flux `q` [m2/s] (clamped to `[q_min, q_max]`) according to
`model.q_conversion`, updating the face fluxes first if any option needs them.
"""
function update_q_from_psi_out!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)
    needs_face_fluxes(model) && update_face_fluxes!(model, grid, state)
    q_from_psi_out!(model, grid, state, model.q_conversion)
    return nothing
end


"""
$(TYPEDSIGNATURES)

Convert the routed flux `psi_out` [m3/s] to the distributed flux `q` [m2/s] via `q = psi_out / corfac`,
clamped to `[q_min, q_max]`. `corfac` is exactly 0 at a cell whose (smoothed) potential gradient is
identically zero, so the bare division gave 0/0 = NaN (net-refreezing cell, `psi_out == 0`) or Inf
(`psi_out > 0`) there, and `max`/`min` propagate NaN. Adding `eps(T)` to the denominator, the same
guard every other division in this file uses, keeps q finite (0 for `psi_out == 0`). For `psi_out > 0`
on such a cell q is still huge, since the flow direction is genuinely undefined -- use `q_max` if you
need to bound it. `epsT` is a plain scalar computed outside the broadcast for the reason given in
`update_q!`.
"""
function update_q_from_psi_out!(model::AbstractKazmierczakHydroModel)

    epsT = eps(eltype(model.corfac))
    @. model.q = min(max(model.psi_out / (model.corfac + epsT), model.q_min), model.q_max)

    return nothing

end


# Le Brocq Eq. 9 / K24: q = psi_out / corfac, with corfac from the cell's flow direction.
q_from_psi_out!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::QFromOutflow) =
    update_q_from_psi_out!(model)

# Face fluxes -> face-normal q (Fx/dy, Fy/dx) -> averaged to the cell centre per component -> |q|.
function q_from_psi_out!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::QFromFaceAverage)
    t = model.routing_tape
    q_from_faces_kernel!(model.q, t.Fx, t.Fy, state.mask, grid.Nx, grid.Ny, grid.dx, grid.dy, model.q_min, model.q_max)
    return nothing
end

function q_from_faces_kernel!(q, Fx, Fy, mask, Nx, Ny, dx, dy, q_min, q_max)
    @inbounds for j in 1:Ny, i in 1:Nx
        v = zero(eltype(Fx))
        if mask[i, j] == 1.0
            qx = (Fx[i, j] + Fx[i + 1, j]) / (2 * dy)
            qy = (Fy[i, j] + Fy[i, j + 1]) / (2 * dx)
            v = hypot(qx, qy)
        end
        q[i, j] = min(max(v, q_min), q_max)
    end
    return nothing
end


"""
$(TYPEDSIGNATURES)

Freeze-on capacity [m/s ice equivalent] of every grounded cell after a solve, written into `C`
(`Nx x Ny`): the water that reaches the cell, routed in from upstream plus the water from above,
`C = (rho_w * Psi_in / (dx * dy) + i_eb) / rho_i`, with `Psi_in` the sum over neighbours of their
`psi_out` times the fraction of their outflow sent toward the cell and `i_eb` [kg/m^2/s] the water
reaching the bed from above (`model.i_eb`). The cell's own melt and dissipation are not included:
they are heat, which a host using the capacity basal boundary condition already counts in its
freezing demand. `i_eb` is water and in no heat balance, so it is part of the supply. Zero off the
grounded mask.

GDS-Warner routes with weights computed on the fly, so its routing weights are filled here first
when face fluxes have not already done so.
"""
function freeze_on_capacity!(C, model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)
    if model.routing_scheme isa GDSWarner && !needs_face_fluxes(model)
        compute_routing_weights!(model, grid, state)
    end
    freeze_on_capacity_kernel!(C, model.psi_out, model.routing_tape.w8, model.i_eb, state.mask, grid.Nx, grid.Ny,
                               grid.dx, grid.dy, model.rho_w, model.rho_i)
    return C
end

function freeze_on_capacity_kernel!(C, psi, W, i_eb, mask, Nx, Ny, dx, dy, rho_w, rho_i)
    T = eltype(C)
    @inbounds for j in 1:Ny, i in 1:Nx
        psi_in = zero(T)
        supply = zero(T)
        if mask[i, j] == 1.0
            for d in 1:8
                di, dj = ROUTE_OFFSETS[d]
                ni, nj = i + di, j + dj
                in_domain(ni, nj, Nx, Ny) || continue
                mask[ni, nj] == 1.0 || continue
                # the neighbour sends toward (i, j) in the direction opposite to d
                psi_in += psi[ni, nj] * W[ROUTE_OPPOSITE[d], ni, nj]
            end
            supply = rho_w * psi_in / (dx * dy) + i_eb[i, j]
        end
        C[i, j] = max(supply, zero(T)) / rho_i
    end
    return nothing
end
