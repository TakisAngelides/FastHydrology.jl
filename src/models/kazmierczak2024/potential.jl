#################################
# Model: Kazmierczak et al 2024 #
#################################

# The geometric potential phi0: filling its pits for routing, its gradients and their
# Kamb & Echelmeyer (1986) stress-gradient-coupling smoothing.


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
  in one pass; guarantees every grounded cell drains to an outlet (an `OCEAN` or `LAND` cell).
"""
abstract type AbstractFillAlgorithm end

"""
$(TYPEDSIGNATURES)

The original iterative filling (`fill_iters` Jacobi passes of neighbour-mean raising). See
[`AbstractFillAlgorithm`](@ref).

Like KORI-ULB's, it ignores the mask: every strict local minimum is filled, also in non-grounded
cells, whose filled values then enter the centred gradient of grounded neighbours. Use
[`PriorityFloodFill`](@ref) to fill grounded cells only.
"""
struct JacobiFill <: AbstractFillAlgorithm end

"""
$(TYPEDSIGNATURES)

Hollow filling as described by Le Brocq et al. (2006, Sec. 3.1): each strict local minimum is given
the value of its lowest neighbour, repeated for up to `fill_iters` passes. Leaves flats where
hollows were. See [`AbstractFillAlgorithm`](@ref).

Like KORI-ULB's, it ignores the mask: every strict local minimum is filled, also in non-grounded
cells, whose filled values then enter the centred gradient of grounded neighbours. Use
[`PriorityFloodFill`](@ref) to fill grounded cells only.
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

Which cells count as having a meaningful potential for the subglacial water, i.e. `model.phi_weight`
(see `KazmierczakHydroModel`'s `potential_weight` keyword and `update_phi0!`). Cells outside the
weight are kept out of the stress-gradient-coupling smoothing (`prepare_routing!`), the true gradient
magnitude (`update_potential_gradients!`) and the face dissipation (`update_face_fluxes!`). Across their
edge phi0 can jump by the whole overburden (e.g. the `h = b = 0` padding around a rasterised mesh,
where phi0 = 0), which otherwise creates spurious drainage divides (a high-N band about one kernel
radius inside the boundary, collapsing to a single hot spot for large kernels) and inflates N on the
boundary cells.
- [`GroundedIce`](@ref) (default): the `GROUNDED` cells, plus the `OTHER_BASIN` cells that carry
  grounded ice (`h > 0` and `rho_i h >= -rho_sw b`), i.e. the ice beyond a basin cut. The smoothing
  then feels the ice across the cut, as the stress coupling it stands for would. `OTHER_BASIN` cells
  without ice (no data, such as the padding around a mesh) or with floating ice are left out. (No water
  crosses into `OTHER_BASIN` either way: those faces are closed.)
- [`GroundedMask`](@ref): the `GROUNDED` cells only. Use it when the geometry outside the basin is not
  consistent with the inside, e.g. a transient basin run in which the outside ice is not evolved.

The two are the same whenever the mask has no `OTHER_BASIN` cells with grounded ice (whole ice
sheets, a coupled ice model's grounded mask, padding without data).
"""
abstract type AbstractPotentialWeight end
"""$(TYPEDSIGNATURES)

See [`AbstractPotentialWeight`](@ref)."""
struct GroundedIce <: AbstractPotentialWeight end
"""$(TYPEDSIGNATURES)

See [`AbstractPotentialWeight`](@ref)."""
struct GroundedMask <: AbstractPotentialWeight end


"""
$(TYPEDSIGNATURES)

Update the geometric potential phi0, and `model.phi_weight`: 1 where phi0 is a meaningful potential
for the subglacial water, 0 elsewhere, as chosen by `model.potential_weight` (see
`AbstractPotentialWeight`).
"""
function update_phi0!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)

    @. model.phi0 = model.rho_i * model.g * state.h + model.rho_w * model.g * state.b

    update_phi_weight!(model, state, model.potential_weight)

    return nothing

end

function update_phi_weight!(model::AbstractKazmierczakHydroModel, state::HydroState, ::GroundedIce)
    T = eltype(model.phi_weight)
    @. model.phi_weight = ifelse((state.mask == GROUNDED) |
                                 ((state.mask == OTHER_BASIN) & (state.h > 0) & (model.rho_i * state.h >= -model.rho_sw * state.b)), one(T), zero(T))
    return nothing
end

function update_phi_weight!(model::AbstractKazmierczakHydroModel, state::HydroState, ::GroundedMask)
    T = eltype(model.phi_weight)
    @. model.phi_weight = ifelse(state.mask == GROUNDED, one(T), zero(T))
    return nothing
end


"""
$(TYPEDSIGNATURES)

Fill the local minima of the geometric potential to avoid water getting stuck in them. The result goes
into `model.phi0_filled`, which is used only for routing (flow direction); `model.phi0` stays the true
potential, so the effective pressure N, `S_inf`/`N_inf` gradients and dissipation are not affected by
the filling. `model.h` receives the ice thickness consistent with the filled potential (only used for
the mean thickness in the smoothing kernel size).
"""
function potential_filling!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState; from_true_potential::Bool = true)

    phi0     = model.phi0_filled
    phi0_tmp = model.phi0_tmp

    # Start from the true potential, unless the caller has already put the surface to fill (e.g. the
    # smoothed potential of a non-GDS routing scheme, see prepare_routing!) into phi0_filled.
    from_true_potential && (phi0 .= model.phi0)
    outlet_potential_matters(model.routing_scheme, model.fill_algorithm) && set_outlet_potential!(model, state)
    phi0_tmp .= phi0

    fill_potential!(model, grid, state, model.fill_algorithm)

    # Ice thickness consistent with the filled potential; stored separately so it does not affect other calculations like effective pressure.
    @. model.h = (model.phi0_filled - model.rho_w * model.g * state.b) / (model.rho_i * model.g)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Set the routing potential `model.phi0_filled` of the outlet cells to the potential of their water:
`rho_w g b + rho_sw g max(-b, 0)` under the ocean (its hydrostatic pressure at the bed) and
`rho_w g b` on land (atmospheric pressure). Their geometric phi0 is not the potential of any
subglacial water (under an ice shelf it misses the water column below the ice, on padding without
data it is 0), but routing compares it with the grounded cells next to it, to decide whether water
drains into the outlet. At flotation the grounded phi0 equals the ocean value, so a grounded cell at
the grounding line drains into the ocean exactly when its ice is above flotation.
"""
function set_outlet_potential!(model::AbstractKazmierczakHydroModel, state::HydroState)
    rho_w_g, rho_sw_g = model.rho_w * model.g, model.rho_sw * model.g
    z = zero(eltype(state.b))
    @. model.phi0_filled = ifelse(state.mask == OCEAN, rho_w_g * state.b + rho_sw_g * max(-state.b, z),
                           ifelse(state.mask == LAND, rho_w_g * state.b, model.phi0_filled))
    return nothing
end

"""
$(TYPEDSIGNATURES)

`JacobiFill`: the original filling, `model.fill_iters` passes of raising every strict local minimum
of `model.phi0_filled` to the mean of its 4 neighbours.
"""
function fill_potential!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, alg::Union{JacobiFill, LowestNeighbourFill})

    lowest = alg isa LowestNeighbourFill

    phi0     = model.phi0_filled
    phi0_tmp = model.phi0_tmp
    Nx = grid.Nx
    Ny = grid.Ny

    # Jacobi-style passes: every strict local minimum of the current phi0 is raised to the mean of its
    # 4 neighbours (written to phi0_tmp, then copied back). Whether a cell is a strict minimum depends
    # only on itself and its 4 neighbours, so after the first full-grid pass only the cells filled in
    # the previous pass and their neighbours can be minima: later passes check just those, and stop
    # once a pass fills nothing. Same result as scanning the whole grid fill_iters times.
    filled = model.filled_cells
    candidates = model.fill_candidates
    fill!(model.fill_stamp, Int32(0))
    empty!(filled)
    if model.fill_iters > 0
        @inbounds for j in 1:Ny, i in 1:Nx
            fill_if_strict_minimum!(phi0, phi0_tmp, filled, i, j, Nx, Ny, lowest)
        end
    end
    for pass in 1:model.fill_iters
        isempty(filled) && break
        @inbounds for (i, j) in filled
            phi0[i, j] = phi0_tmp[i, j]
        end
        pass == model.fill_iters && break
        collect_fill_candidates!(candidates, model.fill_stamp, pass, filled, Nx, Ny)
        empty!(filled)
        @inbounds for k in candidates
            i, j = mod1(k, Nx), cld(k, Nx)
            fill_if_strict_minimum!(phi0, phi0_tmp, filled, i, j, Nx, Ny, lowest)
        end
    end

    return nothing

end


"""
$(TYPEDSIGNATURES)

`PriorityFloodFill`: Priority-Flood+epsilon depression filling (Barnes, Lehman & Mulla 2014,
https://doi.org/10.1016/j.cageo.2013.04.024), restricted to grounded cells.

Outlets (seeds) are the places water can leave the grounded hydrological system: every `OCEAN` or
`LAND` cell next to a `GROUNDED` one (grounding line, land margin), at its outlet potential (see
`set_outlet_potential!`). `OTHER_BASIN` cells and the edge of the grid are closed, so they are not
outlets. Starting from the lowest outlet, cells are visited in increasing order of (filled) potential;
each newly reached grounded neighbour that is not higher than the cell it was reached from is raised
to that cell's value plus `epsilon`. When the queue is empty every grounded cell has a path to an
outlet along which the filled potential strictly decreases, so no grounded pit or flat is left -- in a
single pass, O(n log n). Non-grounded cells keep their potential.

A grounded region with no outlet (enclosed by `OTHER_BASIN` cells and the grid edge) has nowhere for
its water to go, so it is an error.
"""
function fill_potential!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, alg::PriorityFloodFill)

    phi = model.phi0_filled
    mask = state.mask
    Nx, Ny = grid.Nx, grid.Ny
    T = eltype(phi)
    epsilon = T(alg.epsilon)

    closed = model.fill_stamp          # reused as the "already queued" flag (0/1)
    fill!(closed, Int32(0))
    # The heap lives in the workspace and keeps its capacity between calls, so a solve allocates nothing here.
    heap_p, heap_k = model.fill_heap_p, model.fill_heap_k
    empty!(heap_p); empty!(heap_k)

    grounded(i, j) = mask[i, j] == GROUNDED
    @inbounds for j in 1:Ny, i in 1:Nx
        seed = is_outlet(mask[i, j]) &&
            ((i > 1 && grounded(i - 1, j)) || (i < Nx && grounded(i + 1, j)) ||
             (j > 1 && grounded(i, j - 1)) || (j < Ny && grounded(i, j + 1)))
        if seed
            closed[i, j] = 1
            heap_push!(heap_p, heap_k, phi[i, j], Int32(i + (j - 1) * Nx))
        end
    end

    @inbounds while !isempty(heap_k)
        pc, k = heap_pop!(heap_p, heap_k)
        i, j = mod1(k, Nx), cld(k, Nx)
        for (ni, nj) in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
            (1 <= ni <= Nx && 1 <= nj <= Ny) || continue
            (closed[ni, nj] == 0 && grounded(ni, nj)) || continue
            closed[ni, nj] = 1
            if phi[ni, nj] <= pc
                phi[ni, nj] = pc + epsilon
            end
            heap_push!(heap_p, heap_k, phi[ni, nj], Int32(ni + (nj - 1) * Nx))
        end
    end

    @inbounds for j in 1:Ny, i in 1:Nx
        (grounded(i, j) && closed[i, j] == 0) || continue
        n = count(I -> grounded(I[1], I[2]) && closed[I] == 0, CartesianIndices(mask))
        error("PriorityFloodFill: $n grounded cell(s), e.g. ($i, $j), have no path to an outlet: they are enclosed by " *
              "OTHER_BASIN cells and the edge of the grid, which are closed. Water leaves only into OCEAN or LAND cells; " *
              "mark where it can drain in the mask.")
    end

    return nothing

end

# Minimal binary min-heap on (priority, linear index) kept in two parallel vectors.
function heap_push!(p::Vector, k::Vector{Int32}, pv, kv::Int32)
    push!(p, pv); push!(k, kv)
    c = length(p)
    @inbounds while c > 1
        par = c >> 1
        p[par] <= p[c] && break
        p[par], p[c] = p[c], p[par]
        k[par], k[c] = k[c], k[par]
        c = par
    end
    return nothing
end

function heap_pop!(p::Vector, k::Vector{Int32})
    @inbounds begin
        top_p, top_k = p[1], k[1]
        last_p, last_k = pop!(p), pop!(k)
        n = length(p)
        if n > 0
            c = 1
            while true
                l = 2c; r = l + 1
                l > n && break
                m = (r <= n && p[r] < p[l]) ? r : l
                p[m] < last_p || break
                p[c] = p[m]; k[c] = k[m]
                c = m
            end
            p[c] = last_p; k[c] = last_k
        end
    end
    return top_p, top_k
end


"""
$(TYPEDSIGNATURES)

If cell `(i, j)` of `phi0` is a strict local minimum, write the mean of its 4 neighbours to
`phi0_tmp[i, j]` (or, with `lowest`, the value of its lowest neighbour -- LowestNeighbourFill) and record the cell in `filled`. Domain edges are treated as zero-gradient
(edge-replicated) neighbours, so an edge cell is never a strict local minimum and is never filled.
"""
@inline function fill_if_strict_minimum!(phi0, phi0_tmp, filled, i, j, Nx, Ny, lowest::Bool = false)
    @inbounds begin
        p = phi0[i, j]
        im1, ip1 = max(i - 1, 1), min(i + 1, Nx)
        jm1, jp1 = max(j - 1, 1), min(j + 1, Ny)
        p1, p2 = phi0[ip1, j], phi0[im1, j]
        p3, p4 = phi0[i, jp1], phi0[i, jm1]
        if p < p1 && p < p2 && p < p3 && p < p4
            phi0_tmp[i, j] = lowest ? min(p1, p2, p3, p4) : (p1 + p2 + p3 + p4) / 4   # integer literal: 4.0 would promote a Float32 model to Float64
            push!(filled, (Int32(i), Int32(j)))
        end
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

The cells whose strict-minimum status can have changed after the cells in `filled` were updated:
those cells and their in-domain 4-neighbours, as duplicate-free column-major linear indices.
Duplicates are skipped by stamping each collected cell with the current pass number in `stamp` (so
`stamp` never needs clearing within a call; `potential_filling!` resets it once per call). The order
of `candidates` does not matter: each pass is a Jacobi update that reads only the previous pass's
values.
"""
function collect_fill_candidates!(candidates::Vector{Int32}, stamp::Matrix{Int32}, pass, filled, Nx, Ny)
    empty!(candidates)
    p = Int32(pass)
    @inbounds for (i, j) in filled
        k = i + (j - Int32(1)) * Int32(Nx)
        for (cond, kk) in ((true, k), (i > 1, k - Int32(1)), (i < Nx, k + Int32(1)),
                           (j > 1, k - Int32(Nx)), (j < Ny, k + Int32(Nx)))
            if cond && stamp[kk] != p
                stamp[kk] = p
                push!(candidates, kk)
            end
        end
    end
    return candidates
end


"""
$(TYPEDSIGNATURES)

Which stencil the gradient *magnitudes* of the potential are taken with (see
`KazmierczakHydroModel`'s `gradient_discretization` keyword). It sets `abs_grad_phi0` (the true
potential: `S_inf`, the melt-opening term `Q * |grad(phi0)|`, `CellCentredDissipation` and the
sheet-thickness closures), and `abs_grad_phi0_s` wherever that is the magnitude of an *unsmoothed*
gradient (the potential routing schemes, or the GDS schemes with `coupling_length_kamb86 = 0`). The
Kamb-smoothed GDS components are always centred. The gradient components (`minus_grad_phi0_x/y`,
flow direction) are the same under both, since the average of a cell's two face gradients is the
centred difference.
- [`CentredGradient`](@ref) (default for all schemes except `Warner`): the magnitude of the centred
  components, `sqrt(gx^2 + gy^2)` with `gx = (f[i+1] - f[i-1]) / 2dx`.
- [`FaceGradient`](@ref) (default with `Warner` routing): the face gradients are squared on each
  face and averaged to the centre (see [`face_gradient_magnitude!`](@ref)) -- the same staggered
  construction as `FaceDissipation`, so with Warner every term of the energy budget sees the
  potential on the faces the water actually crosses.

For `abs_grad_phi0`, both stencils use only neighbours inside `model.phi_weight` (one-sided where only
one is, see `update_phi0!`).
"""
abstract type AbstractGradientDiscretization end
"""$(TYPEDSIGNATURES)

See [`AbstractGradientDiscretization`](@ref)."""
struct CentredGradient <: AbstractGradientDiscretization end
"""$(TYPEDSIGNATURES)

See [`AbstractGradientDiscretization`](@ref)."""
struct FaceGradient <: AbstractGradientDiscretization end


"""
$(TYPEDSIGNATURES)

Compute the gradients of the geometric potential. `model.abs_grad_phi0` (used by `S_inf`, `N_inf` and the
dissipation term) is the magnitude of the gradient of the TRUE potential `model.phi0`. The components
`model.minus_grad_phi0_x/y` (negative gradient, i.e. the flow direction) are taken from the FILLED
potential `model.phi0_filled`, since they only feed the routing (via the smoothing and `psi_out`).
The magnitude's stencil follows `model.gradient_discretization` (see `AbstractGradientDiscretization`),
and uses only neighbours inside `model.phi_weight` (see `update_phi0!`), so a cell next to the edge of
the grounded ice does not see the jump in phi0 across it.
"""
function update_potential_gradients!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid)
    update_true_gradient_magnitude!(model, grid, model.gradient_discretization)
    minus_gradient_x!(grid, model.minus_grad_phi0_x, model.phi0_filled)
    minus_gradient_y!(grid, model.minus_grad_phi0_y, model.phi0_filled)
    return nothing
end

update_true_gradient_magnitude!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, ::CentredGradient) =
    centred_gradient_magnitude!(grid, model.abs_grad_phi0, model.phi0, model.phi_weight)

update_true_gradient_magnitude!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, ::FaceGradient) =
    face_gradient_magnitude!(grid, model.abs_grad_phi0, model.phi0, model.phi_weight)


"""
$(TYPEDSIGNATURES)

`model.abs_grad_phi0_s`, the (Euclidean) magnitude of the *unsmoothed* routing gradient, i.e. of
`model.phi0_filled`, with the stencil of `model.gradient_discretization`. Expects
`model.minus_grad_phi0_x/y` to hold that surface's centred components (`update_potential_gradients!`).
"""
update_unsmoothed_gradient_magnitude!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, ::CentredGradient) =
    (@. model.abs_grad_phi0_s = sqrt(model.minus_grad_phi0_x * model.minus_grad_phi0_x + model.minus_grad_phi0_y * model.minus_grad_phi0_y); nothing)

update_unsmoothed_gradient_magnitude!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, ::FaceGradient) =
    face_gradient_magnitude!(grid, model.abs_grad_phi0_s, model.phi0_filled)


"""
$(TYPEDSIGNATURES)

The normalised Kamb & Echelmeyer (1986) stress-gradient-coupling kernel (a cone) for mean grounded
ice thickness `h_avg`, used by `update_smoothed_potential_gradients!` (smoothing the gradient
components) and by `prepare_routing!` for the non-GDS routing schemes (smoothing the potential).

The last kernel is kept in `model.coupling_kernel_cache` with its `h_avg` and returned as is while
`h_avg` is unchanged (every solve on the same geometry). Treat it as read-only.
"""
function coupling_kernel(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, h_avg)
    cache = model.coupling_kernel_cache
    h = typeof(grid.dx)(h_avg)
    cache[][1] == h && return cache[][2]
    kernel = build_coupling_kernel(model, grid, h)
    cache[] = (h, kernel)
    return kernel
end

function build_coupling_kernel(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, h_avg)

    # Grid spacing in each direction. The kernel below computes each cell's distance from the
    # center using dx and dy separately, rather than collapsing both to a single isotropic
    # Delta = (dx+dy)/2 -- so its nonzero support is a genuine circle in physical space (an
    # ellipse in these grid-index coordinates whenever dx != dy), correct for any cell aspect
    # ratio rather than only for square cells.
    dx = grid.dx
    dy = grid.dy
    T = typeof(dx)   # the kernel has the grid's float type (integer literals below keep it from promoting)

    scale = T(h_avg) * model.longcoupwater * 2

    # Radius of the cone base (= 4 * h_avg * longcoupwater). The cone hits zero at this distance,
    # 2-5x the Kamb & Echelmeyer (1986) coupling length (4-10x ice thickness for ice sheets). The
    # true effective coupling length is the kernel's *area-weighted* mean distance from center --
    # i.e. the 2D mass centroid of the cone profile k(r) = max(0, 1 - r/width), weighting k(r) by
    # the r dr area element a 2D radial integral actually carries (there is more area, hence more
    # kernel mass, in an annulus further from the center) -- which works out to width/2
    # = 2 * h_avg * longcoupwater, giving ~10x ice thickness for longcoupwater=5: the *upper edge*
    # of Kamb & Echelmeyer's range, not comfortably in the middle of it. (Naively averaging k(r)
    # alone over r without that area weighting, as if it were a 1D profile, gives the smaller
    # width/3 = 4/3 * h_avg * longcoupwater ~ 6.7x instead -- but that number understates how much
    # of the kernel's weight actually sits at larger radius once the 2D area growth is accounted
    # for, so it is not the right quantity to compare against Kamb & Echelmeyer's ice-thickness
    # multiple.) At coarse resolution (16-32 km), the coupling length (6-15 km for 1500 m ice, from
    # Kamb & Echelmeyer's literal 4-10x range) is smaller than a grid cell, so set longcoupwater = 0.
    width = 2 * scale

    # A cone whose `width` is at most one cell (in either direction) gives a 1x1 kernel below,
    # i.e. no smoothing at all -- deliberately not widened: at those resolutions the coupling
    # length is not resolved, and a cone widened to span a cell would still be ~the identity.

    # Kernel size, independent per axis (frb_x, frb_y): cached_fft_convolve! (fft_convolution.jl)
    # supports a rectangular kernel array, so each axis is sized from its own spacing rather than
    # both being forced to the size the finer axis would need (which wasted array space and FFT
    # work on the coarser axis without changing the result).
    maxlevel_x = 2 * round(Int, width / dx - 0.5) + 1
    maxlevel_y = 2 * round(Int, width / dy - 0.5) + 1
    frb_x = Int((maxlevel_x - 1) / 2)
    frb_y = Int((maxlevel_y - 1) / 2)

    kernel = zeros(T, maxlevel_x, maxlevel_y)

    for nj in 1:maxlevel_y, ni in 1:maxlevel_x
        # True physical (Euclidean) distance from the kernel center, using dx and dy separately.
        dist = sqrt((dx * (ni - frb_x - 1))^2 +
                    (dy * (nj - frb_y - 1))^2) / scale

        kernel[ni, nj] = max(zero(T), 1 - dist / 2)
    end

    kernel ./= sum(kernel)

    return kernel

end


"""
$(TYPEDSIGNATURES)

Update the smoothed geometric potentials to incorporate the effects of the stress-gradient coupling. See also the description of the update_q! function.

The water flux at a given point is influenced by variations in ice thickness some distance away. To account for this we perform a convolution of the gradient of the potential
such that the influence of nearby points is now incorporated into the value of the gradient of the potential at that point.

The convolution is normalised by the weight `model.phi_weight` (see `AbstractPotentialWeight`), as
the potential smoothing of the other routing schemes is (`smooth_potential!`):
`s = K * (w g) / K * w`, with the components `g` of the filled potential taken one-sided next to a cell
outside `w`. KORI-ULB convolves the plain centred components, which on a boundary cell span the jump
to the cell outside (the whole overburden next to `h = b = 0` padding) and spread it a kernel radius
inward, creating the same spurious divides. Cells outside `w` within reach of the kernel get the
weighted average; farther ones keep the unsmoothed components.
"""
function update_smoothed_potential_gradients!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)

    if model.longcoupwater == 0.0
        model.minus_grad_phi0_sx .= model.minus_grad_phi0_x
        model.minus_grad_phi0_sy .= model.minus_grad_phi0_y
        update_unsmoothed_gradient_magnitude!(model, grid, model.gradient_discretization)
        return nothing
    end

    # Average grounded-ice thickness, of the *filled* surface (`model.h`), as in K24/KORI. The other
    # routing schemes (prepare_routing!) smooth the potential before filling, so they cannot use it
    # (the kernel would depend on its own output) and take `state.h` instead.
    kernel = coupling_kernel(model, grid, max(masked_mean(grid, model.h, state.mask), 10.0))

    w, ws = model.phi_weight, model.phi_weight_s
    convolve!(grid, ws, w, kernel)
    tol = sqrt(eps(eltype(ws)))
    for (axis, dest, raw) in ((1, model.minus_grad_phi0_sx, model.minus_grad_phi0_x), (2, model.minus_grad_phi0_sy, model.minus_grad_phi0_y))
        weighted_minus_gradient!(grid, model.phi0_tmp, model.phi0_filled, w, axis)   # phi0_tmp is free after the filling
        convolve!(grid, dest, model.phi0_tmp, kernel)
        @. dest = ifelse(ws > tol, dest / max(ws, tol), raw)
    end

    # Euclidean magnitude, like abs_grad_phi0 (Kori-ULB's gdsmag uses |sx| + |sy|, which depends on the
    # flow's orientation to the grid: up to sqrt(2) too large at 45 degrees).
    @. model.abs_grad_phi0_s = sqrt(model.minus_grad_phi0_sx * model.minus_grad_phi0_sx + model.minus_grad_phi0_sy * model.minus_grad_phi0_sy)

    return nothing

end
