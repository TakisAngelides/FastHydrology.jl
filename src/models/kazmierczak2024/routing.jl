#################################
# Model: Kazmierczak et al 2024 #
#################################

# Flux-routing schemes of Le Brocq, Payne & Siegert (2006, Computers & Geosciences 32, 1780-1795,
# https://doi.org/10.1016/j.cageo.2006.05.003), selected by `model.routing_scheme` -- see
# `AbstractRoutingScheme` in model.jl. The default `GDSWarner()` keeps its original code path
# (`record_routing_tape_kernel!` reading the smoothed gradient directly); every other scheme is
# expressed as per-cell outflow fractions `W[d, i, j]` to the neighbour in direction `d`
# (`ROUTE_OFFSETS`), which the generic tape recorder below and the face-flux diagnostics use.


const ROUTE_OFFSETS  = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))
const ROUTE_OPPOSITE = (2, 1, 4, 3, 8, 7, 6, 5)

# Tarboton (1997) triangular facets as (cardinal, diagonal) direction pairs of ROUTE_OFFSETS.
const TARBOTON_FACETS = ((2, 6), (2, 8), (4, 8), (4, 7), (1, 7), (1, 5), (3, 5), (3, 6))

@inline in_domain(i, j, Nx, Ny) = 1 <= i <= Nx && 1 <= j <= Ny


"""
$(TYPEDSIGNATURES)

Whether the solve needs face fluxes each sweep: for `QFromFaceAverage` or face-assembled
dissipation (`FaceDissipation` with the dissipation melt on).
"""
needs_face_fluxes(model::KazmierczakHydroModel) =
    model.q_conversion isa QFromFaceAverage ||
    (model.dissipation_melt isa DissipationMeltOn && model.dissipation_discretization isa FaceDissipation)


"""
$(TYPEDSIGNATURES)

Build the routing potential and flow directions for `model.routing_scheme`, called once per
`update_q!` after `update_phi0!`.

- GDS schemes (`GDSWarner`, `GDSTarboton`): the original K24 pipeline -- fill phi0, take its gradient,
  smooth the gradient components with the Kamb & Echelmeyer (1986) kernel.
- Potential schemes (`Warner`, `Quinn`, `Tarboton`, `ModifiedTarboton`): smooth the potential itself
  with the same kernel, then fill it (so the filling acts on the surface the water actually follows).
  Smoothing commutes with differentiation, so this is the same stress-gradient coupling, but the
  result stays the gradient of one surface. `minus_grad_phi0_sx/sy` then hold the (unsmoothed)
  centred gradient of that surface, used for the psi -> q conversion direction (Le Brocq Eq. 9) and
  by `ModifiedTarboton`.

`abs_grad_phi0` (used by N and the dissipation melt) is always the gradient of the true potential.
"""
function prepare_routing!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, scheme::Union{GDSWarner, GDSTarboton})
    potential_filling!(model, grid, state)
    update_potential_gradients!(model, grid)
    update_smoothed_potential_gradients!(model, grid, state)
    if !(scheme isa GDSWarner) || needs_face_fluxes(model)
        compute_routing_weights!(model, grid, state)
    end
    return nothing
end

function prepare_routing!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::Union{Warner, Quinn, Tarboton, ModifiedTarboton})
    if model.longcoupwater == 0.0
        model.phi0_filled .= model.phi0
    else
        kernel = coupling_kernel(model, grid, max(masked_mean(grid, state.h, state.mask), 10.0))
        convolve!(grid, model.phi0_filled, model.phi0, kernel)
    end
    fill_halo!(model.phi0_filled, grid)
    potential_filling!(model, grid, state; from_true_potential = false)
    update_potential_gradients!(model, grid)
    model.minus_grad_phi0_sx .= model.minus_grad_phi0_x
    model.minus_grad_phi0_sy .= model.minus_grad_phi0_y
    fill_halo!(model.minus_grad_phi0_sx, grid)
    fill_halo!(model.minus_grad_phi0_sy, grid)
    @. model.abs_grad_phi0_s = abs(model.minus_grad_phi0_sx) + abs(model.minus_grad_phi0_sy)
    compute_routing_weights!(model, grid, state)
    return nothing
end


"""
$(TYPEDSIGNATURES)

Fill `model.routing_tape.w8[d, i, j]`, the fraction of grounded cell `(i, j)`'s outflow sent to its
neighbour in direction `d` (`ROUTE_OFFSETS`), for `model.routing_scheme`. Fractions of a cell sum to 1,
or to 0 for a cell with nowhere downhill to send water (a sink; see `AbstractRoutingScheme`). Water
sent to a non-grounded neighbour, or (GDS schemes) off the domain edge, leaves the system.
"""
function compute_routing_weights!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)
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
    angs = (0.0, a, pi / 2, pi - a, pi * 1.0, pi + a, 3pi / 2, 2pi - a)
    dirs = (2, 8, 4, 7, 1, 5, 3, 6)
    theta = mod(atan(fy, fx), 2pi)
    k = 8
    @inbounds for m in 1:7
        if angs[m] <= theta < angs[m + 1]
            k = m
            break
        end
    end
    lo = angs[k]
    hi = k == 8 ? 2pi : angs[k + 1]
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
skipped. The result is signed: water pushed up the true potential (out of a filled hollow, or
uphill under the smoothed GDS direction) gives a negative contribution.
"""
function update_face_fluxes!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)
    t = model.routing_tape
    face_fluxes_kernel!(t.Fx, t.Fy, t.diss, model.psi_out, t.w8, state.mask, model.phi0, grid.Nx, grid.Ny, grid.dx, grid.dy,
                        model.L_w, model.dissipation_discretization isa FaceDissipation)
    return nothing
end

function face_fluxes_kernel!(Fx, Fy, diss, psi, W, mask, phi, Nx, Ny, dx, dy, L_w, do_diss)
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
        i > 1  && (P += Fx[i, j]     * (phi[i - 1, j] - phi[i, j]))
        i < Nx && (P += Fx[i + 1, j] * (phi[i, j] - phi[i + 1, j]))
        j > 1  && (P += Fy[i, j]     * (phi[i, j - 1] - phi[i, j]))
        j < Ny && (P += Fy[i, j + 1] * (phi[i, j] - phi[i, j + 1]))
        diss[i, j] = P / 2 / (dx * dy * L_w)
    end
    return nothing
end


"""
$(TYPEDSIGNATURES)

Convert routed flux to the distributed flux `q` [m2/s] (clamped to `[q_min, q_max]`) according to
`model.q_conversion`, updating the face fluxes first if any option needs them.
"""
function update_q_from_psi_out!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)
    needs_face_fluxes(model) && update_face_fluxes!(model, grid, state)
    q_from_psi_out!(model, grid, state, model.q_conversion)
    return nothing
end

# Le Brocq Eq. 9 / K24: q = psi_out / corfac, with corfac from the cell's flow direction.
q_from_psi_out!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::QFromOutflow) =
    update_q_from_psi_out!(model)

# Face fluxes -> face-normal q (Fx/dy, Fy/dx) -> averaged to the cell centre per component -> |q|.
function q_from_psi_out!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::QFromFaceAverage)
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


# Dissipation melt added to mdot_total, per `model.dissipation_discretization`.
function add_dissipation_term!(model::KazmierczakHydroModel, ::CellCentredDissipation)
    @. model.mdot_total += abs(model.q * model.abs_grad_phi0) / model.L_w
    return nothing
end

function add_dissipation_term!(model::KazmierczakHydroModel, ::FaceDissipation)
    diss = model.routing_tape.diss
    @inbounds for j in axes(diss, 2), i in axes(diss, 1)
        model.mdot_total[i, j] += diss[i, j]
    end
    return nothing
end
