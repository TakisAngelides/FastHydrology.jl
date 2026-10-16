#################################
# Model: Kazmierczak et al 2024 #
#################################

# Flow routing: the psi_out algorithms that accumulate each cell's upstream water flux, and
# the routing tape that records a traversal so it can be replayed.


"""
$(TYPEDSIGNATURES)

Trait selecting which flow-routing implementation `resolve_q!` uses to compute psi_out each sweep
(see `KazmierczakHydroModel`'s `psi_out_algorithm` keyword). Stored as a type parameter, resolved by
multiple dispatch at compile time via `route_psi_out!` below, the same pattern as
`AbstractDissipationMelt` (melt_sources.jl).

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

Computes psi_out for the current sweep, dispatching on `model.psi_out_algorithm`
(`RecursivePsiOut()`, `IterativePsiOut()`, or `TopologicalPsiOut()` -- see
`AbstractPsiOutAlgorithm`'s docstring above) to the recursive `update_psi_out!`, the
stack-based `update_psi_out_iterative!`, or the sorted single-pass `update_psi_out_topological!`.
Every `resolve_q!` method calls this instead of `update_psi_out!` directly, so the algorithm choice
applies uniformly regardless of which sliding law/dissipation-melt combination is active.
"""
route_psi_out!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState) =
    route_psi_out!(model, grid, state, model.psi_out_algorithm)

route_psi_out!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::RecursivePsiOut) =
    update_psi_out!(model, grid, state)

route_psi_out!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::IterativePsiOut) =
    update_psi_out_iterative!(model, grid, state)

route_psi_out!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, algorithm::TopologicalPsiOut) =
    update_psi_out_topological!(model, grid, state, algorithm.allow_cycles)

route_psi_out!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::TapedPsiOut) =
    update_psi_out_taped!(model, grid, state)


"""
$(TYPEDSIGNATURES)

Mark `model`'s recorded `TapedPsiOut` traversal as stale, so the next `route_psi_out!` re-records it.
"""
invalidate_routing_tape!(model::AbstractKazmierczakHydroModel) = (model.routing_tape.valid = false; nothing)


"""
$(TYPEDSIGNATURES)

`TapedPsiOut`'s routing: record the traversal if the tape is stale, then replay it with the current
`mdot_total`. See [`TapedPsiOut`](@ref).
"""
function update_psi_out_taped!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)
    tape = model.routing_tape
    tape.valid || record_routing_tape!(tape, model, grid, state)
    replay_routing_tape!(model.psi_out, model.mdot_total, state.mask, tape, grid.Nx, grid.Ny, grid.dx, grid.dy, model.rho_w)
    return nothing
end


"""
$(TYPEDSIGNATURES)

Record the operations `update_psi_out!`/`accumulate_psi_out!` would perform, in the same order, using
an explicit-stack traversal (frame `(i, j, k)`: `k == 0` not yet visited, `1 <= k <= 4` next
neighbour to fold in, `k == 5` done), as in `update_psi_out_iterative!`. Only the operations are
recorded; no psi_out values are computed. Every grounded cell's own source term is applied up front by
`replay_routing_tape!` rather than on first visit: a cell's psi_out is never read before its first
visit, so this does not change any value read.

Neighbours without grounded ice are skipped rather than recorded as `+= 0 * w`, which is a no-op.
"""
function record_routing_tape!(tape::RoutingTape, model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)
    T = eltype(model.abs_grad_phi0_s)
    max_calls = model.max_psi_out_calls
    hit_cap = if model.routing_scheme isa GDSWarner
        record_routing_tape_kernel!(tape, state.mask, model.minus_grad_phi0_sx, model.minus_grad_phi0_sy,
                                    grid.Nx, grid.Ny, grid.dx, grid.dy, eps(T), max_calls)
    else
        record_routing_tape_weights_kernel!(tape, state.mask, tape.w8, n_directions(model.routing_scheme), grid.Nx, grid.Ny, max_calls)
    end
    # Warned from here rather than inside the kernel: the logging macro's try/catch in the hot loop
    # stops the compiler from keeping the loop's locals in registers (~6x slower traversal).
    hit_cap && @warn "TapedPsiOut hit max_psi_out_calls = $max_calls cells in one sweep -- cutting the flow-routing traversal off early, matching accumulate_psi_out!'s own cutoff. Pass a larger `max_psi_out_calls` to KazmierczakHydroModel if this grid genuinely has more grounded cells than the default allows." maxlog=1
    tape.valid = true
    return nothing
end

@inline function push_tape_op!(tape::RoutingTape, ci, cj, ni, nj, w)
    push!(tape.dst_i, ci); push!(tape.dst_j, cj); push!(tape.src_i, ni); push!(tape.src_j, nj); push!(tape.w, w)
    return nothing
end

function record_routing_tape_kernel!(tape::RoutingTape{T}, mask, sx, sy, Nx, Ny, dx, dy, epsT, max_calls) where {T}

    for v in (tape.dst_i, tape.dst_j, tape.src_i, tape.src_j)
        empty!(v)
    end
    empty!(tape.w)

    visited = zeros(Bool, Nx, Ny)
    stack = NTuple{3, Int32}[]
    call_count = 0
    hit_cap = false

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

            elseif sk <= 4

                di = sk == 1 ? -1 : sk == 2 ? 1 : 0
                dj = sk == 3 ? -1 : sk == 4 ? 1 : 0
                ni, nj = si + di, sj + dj
                stack[end] = (si, sj, sk + Int32(1))   # overwritten below if the neighbour must be resolved first

                (1 <= ni <= Nx && 1 <= nj <= Ny) || continue

                w = routing_weight(sx[ni, nj], sy[ni, nj], -di, -dj, dx, dy, epsT)
                (w > 0 && mask[ni, nj] == 1.0) || continue

                if visited[ni, nj]
                    push_tape_op!(tape, si, sj, Int32(ni), Int32(nj), T(w))
                else
                    # Resolve the neighbour first, then come back to this same neighbour (sk unchanged).
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

Replay a recorded routing tape: set every grounded cell's psi_out to its own source term
`mdot_total * dx * dy / rho_w` (same expression and evaluation order as `accumulate_psi_out!`), then apply the recorded operations in
order. Kept as a separate function with the fields as arguments so the loop specializes on their
concrete types.
"""
function replay_routing_tape!(psi_out, mdot_total, mask, tape::RoutingTape, Nx, Ny, dx, dy, rho_w)

    z = zero(eltype(psi_out))
    @inbounds for j in 1:Ny, i in 1:Nx
        if mask[i, j] == 1.0
            psi_out[i, j] = mdot_total[i, j] * dx * dy / rho_w
        end
    end

    dst_i, dst_j, src_i, src_j, ws = tape.dst_i, tape.dst_j, tape.src_i, tape.src_j, tape.w
    @inbounds for k in eachindex(ws)
        ci, cj, ni = dst_i[k], dst_j[k], src_i[k]
        if ni == 0
            psi_out[ci, cj] = max(z, psi_out[ci, cj])
        else
            psi_out[ci, cj] += psi_out[ni, src_j[k]] * ws[k]
        end
    end

    return nothing

end


"""
$(TYPEDSIGNATURES)

Fraction of a cell's `psi_out` that leaves through the face toward the neighbour in direction `(di, dj)`
(`di`/`dj` in {-1, 0, 1}; one of them is 0), given that cell's smoothed negative-gradient components
`(sx, sy)`. The flux through an x-face is `q|cosθ|·dy` and through a y-face is `q|sinθ|·dx`, so the
share is `(sx·di·dy + sy·dj·dx) / (|sx|·dy + |sy|·dx)`, positive only when the flow leaves toward that
neighbour. The denominator is exactly `corfac·|s|`, so these weights are consistent with the
`q = psi_out/corfac` conversion, and reduce to `|s_dir|/(|sx|+|sy|)` when `dx == dy`.
"""
@inline function routing_weight(sx, sy, di, dj, dx, dy, epsT)
    return (sx * di * dy + sy * dj * dx) / (abs(sx) * dy + abs(sy) * dx + epsT)
end


"""
$(TYPEDSIGNATURES)

Helper function to the recursive function to calculate the psi_out for every grid cell that has grounded ice.

`call_count` is a per-sweep counter, shared by reference across the whole recursion tree started by
`update_psi_out!`, that caps the total number of cells this recursion is allowed to visit at
`model.max_psi_out_calls` -- mirroring KORI-ULB's own `funcnt <= 5e4` safety cap in
`DpareaWarGds.m` (https://github.com/FrankPat/Kori-ULB/blob/main/subroutines/DpareaWarGds.m).
Without it, a large or unusually convoluted flow-routing domain could recurse deep enough to hit
Julia's call stack limit and crash with a `StackOverflowError` instead of failing predictably. Once
the cap is hit, the current cell is treated as a terminal source (its own mdot_total contribution
only, no further upstream accumulation) so the sweep still terminates with a (locally truncated but
finite) result rather than crashing.
"""
function accumulate_psi_out!(model::AbstractKazmierczakHydroModel, i, j, grid::AbstractHydroGrid, state::HydroState, call_count::Base.RefValue{Int})

    # If the neighbour does not have grounded ice then return 0
    if state.mask[i, j] != 1.0
        # zero(eltype(...)), not a bare 0.0 literal: this function's other return paths
        # (model.psi_out[i, j]) already return in model's own float type, and a bare Float64
        # literal here would make the return type Union{Float32, Float64} for a Float32 model --
        # confirmed via @code_warntype -- taxing every recursive call in this, the default and
        # documented-"30-40x faster" RecursivePsiOut hot path, with a dynamic dispatch/union-split
        # that a Float64 model never pays (0.0 already being a Float64 there).
        return zero(eltype(model.psi_out))
    end

    # If the neighbour has been visited then the psi_out has already been calculated for that cell
    if model.visited[i, j] == 1.0
        return model.psi_out[i, j]
    end

    # Passing the above if statements means we are now visiting cell i, j
    model.visited[i, j] = 1.0

    dx = grid.dx
    dy = grid.dy
    T = eltype(model.abs_grad_phi0_s)

    model.psi_out[i, j] = model.mdot_total[i, j] * dx * dy / model.rho_w

    call_count[] += 1
    if call_count[] > model.max_psi_out_calls
        @warn "accumulate_psi_out! hit max_psi_out_calls = $(model.max_psi_out_calls) cells in one update_psi_out! sweep -- cutting the flow-routing recursion off early at cell ($i, $j) instead of risking a StackOverflowError. Pass a larger `max_psi_out_calls` to KazmierczakHydroModel if this grid genuinely has more grounded cells than the default allows." maxlog=1
        # Same net-refreezing clamp as the normal exit below: without it, a cell cut off here with a
        # negative local mdot_total would hand a negative flux to its downstream neighbour.
        model.psi_out[i, j] = max(zero(eltype(model.psi_out)), model.psi_out[i, j])
        return model.psi_out[i, j]
    end

    @inbounds for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))

        ni, nj = i + di, j + dj

        # Off the edge of the domain: no neighbouring cell, so no upstream contribution can cross
        # in (the no-flux divide condition, Eq. 2b of Kazmierczak et al. 2024's Γ_d boundary).
        (1 <= ni <= grid.Nx && 1 <= nj <= grid.Ny) || continue

        w = routing_weight(model.minus_grad_phi0_sx[ni, nj], model.minus_grad_phi0_sy[ni, nj], -di, -dj, dx, dy, eps(T))

        if w > 0
            model.psi_out[i, j] += accumulate_psi_out!(model, ni, nj, grid, state, call_count) * w
        end
    end

    # If the mdot is very negative that all the flux refreezes then we limit the flux to zero
    model.psi_out[i, j] = max(zero(T), model.psi_out[i, j])

    return model.psi_out[i, j]

end


"""
$(TYPEDSIGNATURES)

Recursive function to calculate the psi_out for every grid cell that has grounded ice.
We initialize psi_out to -1 since it is by definition positive semi-definite and hence we
know that if a grid point has negative psi_out, it is still unvisited.
"""
function update_psi_out!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)

    Nx = grid.Nx
    Ny = grid.Ny

    # Refresh visited cells field
    model.visited .= 0.0

    # Shared by reference across the whole sweep -- see accumulate_psi_out!'s docstring.
    call_count = Ref(0)

    @inbounds for j in 1:Ny, i in 1:Nx
        if state.mask[i, j] == 1.0
            accumulate_psi_out!(model, i, j, grid, state, call_count)
        end
    end

    return nothing

end


"""
$(TYPEDSIGNATURES)

Iterative counterpart to [`update_psi_out!`](@ref)/[`accumulate_psi_out!`](@ref): computes the exact
same psi_out field with an explicit stack standing in for the Julia call stack, instead of true
recursion. It exists purely to sidestep the recursive version's call-stack usage (and per-frame
dispatch/allocation overhead) on large domains, not to change the algorithm -- so it must reproduce
the recursive version's output cell-for-cell, `max_psi_out_calls` cutoff included.

Each stack entry is `(i, j, k)`: `k == 0` means cell `(i, j)` has not been visited yet (do the
first-visit work below); `1 <= k <= 4` means its first `k - 1` neighbours (in the same `dirs` order
`accumulate_psi_out!` iterates) have already had their contribution folded into `psi_out[i, j]`, and
neighbour `k` is next; `k == 5` means all four neighbours are done and `psi_out[i, j]` is ready to be
clamped and finalized. Critically, when a not-yet-visited neighbour is found, we push it *without*
advancing the current frame's `k` -- so once that child is popped (fully resolved, `visited = 1`), we
revisit the same `(i, j, k)` frame, and it takes the "already visited" branch to fold the now-ready
child value in and move on to `k + 1`. This is exactly how a return value flows back to a paused
caller in real recursion, just made explicit.
"""
function update_psi_out_iterative!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)

    Nx = grid.Nx
    Ny = grid.Ny
    dx = grid.dx
    dy = grid.dy
    T = eltype(model.abs_grad_phi0_s)

    # Refresh visited cells field
    model.visited .= 0.0

    # Local counter, playing the same role as the `Ref`-shared `call_count` in the recursive
    # version -- shared across the whole sweep since it lives outside the `stack` loop below.
    call_count = 0

    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    stack = Tuple{Int, Int, Int}[]

    @inbounds for j in 1:Ny, i in 1:Nx

        (state.mask[i, j] == 1.0 && model.visited[i, j] != 1.0) || continue

        push!(stack, (i, j, 0))

        while !isempty(stack)

            si, sj, sk = stack[end]

            if sk == 0

                model.visited[si, sj] = 1.0
                model.psi_out[si, sj] = model.mdot_total[si, sj] * dx * dy / model.rho_w

                call_count += 1
                if call_count > model.max_psi_out_calls
                    @warn "update_psi_out_iterative! hit max_psi_out_calls = $(model.max_psi_out_calls) cells in one sweep -- cutting the flow-routing traversal off early at cell ($si, $sj), matching accumulate_psi_out!'s own cutoff. Pass a larger `max_psi_out_calls` to KazmierczakHydroModel if this grid genuinely has more grounded cells than the default allows." maxlog=1
                    # Clamp before popping, matching accumulate_psi_out!'s own cap-trip branch: a cell
                    # cut off here with a negative local mdot_total (net refreezing) must not hand a
                    # negative flux to its downstream neighbour.
                    model.psi_out[si, sj] = max(zero(eltype(model.psi_out)), model.psi_out[si, sj])
                    pop!(stack)
                else
                    stack[end] = (si, sj, 1)
                end

            elseif sk <= 4

                di, dj = dirs[sk]
                ni, nj = si + di, sj + dj

                # Off the edge of the domain: no neighbouring cell, so no upstream contribution can cross
                # in (the no-flux divide condition, Eq. 2b of Kazmierczak et al. 2024's Γ_d boundary).
                if !(1 <= ni <= Nx && 1 <= nj <= Ny)
                    stack[end] = (si, sj, sk + 1)
                    continue
                end

                w = routing_weight(model.minus_grad_phi0_sx[ni, nj], model.minus_grad_phi0_sy[ni, nj], -di, -dj, dx, dy, eps(T))

                if w <= 0
                    stack[end] = (si, sj, sk + 1)
                    continue
                end

                # If the neighbour does not have grounded ice then it contributes 0, matching
                # accumulate_psi_out!'s own mask check (which runs inside the recursive call, i.e.
                # only once w > 0).
                if state.mask[ni, nj] != 1.0
                    stack[end] = (si, sj, sk + 1)
                    continue
                end

                if model.visited[ni, nj] == 1.0
                    # Neighbour already resolved (either earlier in this sweep, or as the child we
                    # just popped back from): fold its value in and move to the next neighbour.
                    model.psi_out[si, sj] += model.psi_out[ni, nj] * w
                    stack[end] = (si, sj, sk + 1)
                else
                    # Neighbour not resolved yet: resolve it first, without advancing sk -- see
                    # docstring.
                    push!(stack, (ni, nj, 0))
                end

            else
                # If the mdot is very negative that all the flux refreezes then we limit the flux to zero
                model.psi_out[si, sj] = max(zero(T), model.psi_out[si, sj])
                pop!(stack)
            end
        end
    end

    return nothing

end


"""
$(TYPEDSIGNATURES)

Single-pass counterpart to [`update_psi_out!`](@ref)/[`update_psi_out_iterative!`](@ref): builds the
exact same dependency graph `accumulate_psi_out!` traverses recursively -- cell A is upstream of
adjacent cell B (an edge A -> B) iff `w > 0` for that pair -- then processes it with Kahn's algorithm
(a BFS ordered by in-degree) instead of recursion. This is exact *only* when that graph is genuinely
acyclic; see `TopologicalPsiOut`'s docstring above for why real ice-sheet data isn't reliably
acyclic (confirmed on the real Thwaites-2km dataset at any `longcoupwater`, not just when smoothing is
on) and for the `allow_cycles` field this function's `allow_cycles::Bool` argument comes from.

`w` is evaluated once per cell here, from that cell's own gradient toward each neighbour it flows
into, rather than once per neighbour pair from the receiving cell's side as `accumulate_psi_out!`
does -- algebraically the identical test (`w(A->B)` using A's gradient dotted with the A->B direction
equals `accumulate_psi_out!`'s `w` using B's neighbour-indexed lookup of A's gradient dotted with the
B->A direction, since dotting with the negated direction just flips the sign twice), just rearranged
so each cell's outgoing edges can be built from its own data in one pass, without needing to iterate
every cell from every neighbour's perspective twice.

A cycle leaves `remaining_in_degree` unable to reach zero for the cells inside it, so they never get
enqueued. `processed < total_masked` detects this: with `allow_cycles = false` (the default) it
throws an error rather than proceeding with a wrong field -- confirmed on the real Thwaites-2km
dataset that a cycle-affected run's `q` can correlate at ~0 against `RecursivePsiOut`'s, i.e. this is
not a rare-edge-case-safe thing to silently degrade through. With `allow_cycles = true`, it instead
warns (once per sweep) and leaves cells inside a cycle with whatever partial upstream contributions
they received from outside it before it stalled (their own `mdot_total` source never added) -- the
same "locally truncated but finite" philosophy as `max_psi_out_calls`'s cutoff for the other two
algorithms.

Unlike `update_psi_out!`/`update_psi_out_iterative!`, this never needs `model.max_psi_out_calls`:
every grounded cell is enqueued at most once by construction, so there is no unbounded
recursion/traversal to cap in the first place.
"""
function update_psi_out_topological!(model::AbstractKazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, allow_cycles::Bool)

    Nx = grid.Nx
    Ny = grid.Ny
    dx = grid.dx
    dy = grid.dy
    T = eltype(model.phi0)

    model.psi_out .= 0.0 # accumulated via += below as upstream neighbours are processed, so must start clean

    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))

    # Build this cell's outgoing edges (up to 4: which of its masked neighbours it flows into, and
    # with what weight) and every masked cell's in-degree (how many masked neighbours flow into it),
    # in one O(N) pass.
    out_targets = Matrix{Vector{Tuple{Int, Int, T}}}(undef, Nx, Ny)
    in_degree = zeros(Int, Nx, Ny)
    total_masked = 0

    @inbounds for j in 1:Ny, i in 1:Nx
        out_targets[i, j] = Tuple{Int, Int, T}[]
    end

    @inbounds for j in 1:Ny, i in 1:Nx

        state.mask[i, j] == 1.0 || continue
        total_masked += 1

        for (di, dj) in dirs

            ni, nj = i + di, j + dj

            # Off the edge of the domain: no neighbouring cell, so no contribution can cross out
            # (the no-flux divide condition, Eq. 2b of Kazmierczak et al. 2024's Γ_d boundary).
            (1 <= ni <= Nx && 1 <= nj <= Ny) || continue
            state.mask[ni, nj] == 1.0 || continue

            w = routing_weight(model.minus_grad_phi0_sx[i, j], model.minus_grad_phi0_sy[i, j], di, dj, dx, dy, eps(T))

            if w > 0
                push!(out_targets[i, j], (ni, nj, w))
                in_degree[ni, nj] += 1
            end

        end

    end

    # Kahn's algorithm: a plain Vector used as an array-backed queue (push at the end, read via an
    # advancing `head` index) rather than `popfirst!`, which is O(N) per call on a Vector and would
    # make the whole sweep O(N^2).
    queue = Tuple{Int, Int}[]
    sizehint!(queue, total_masked)

    @inbounds for j in 1:Ny, i in 1:Nx
        if state.mask[i, j] == 1.0 && in_degree[i, j] == 0
            push!(queue, (i, j))
        end
    end

    remaining_in_degree = in_degree # no cells are re-enqueued, so it's safe to decrement in place
    processed = 0

    head = 1
    @inbounds while head <= length(queue)

        i, j = queue[head]
        head += 1
        processed += 1

        # All upstream contributions are already folded in via the += below, by construction (this
        # cell only reached the queue once every upstream neighbour had already been processed) --
        # so what's left is to add this cell's own source term and clamp.
        model.psi_out[i, j] = max(zero(T), model.psi_out[i, j] + model.mdot_total[i, j] * dx * dy / model.rho_w)

        for (ni, nj, w) in out_targets[i, j]
            model.psi_out[ni, nj] += model.psi_out[i, j] * w
            remaining_in_degree[ni, nj] -= 1
            if remaining_in_degree[ni, nj] == 0
                push!(queue, (ni, nj))
            end
        end

    end

    if processed < total_masked
        n_stuck = total_masked - processed
        if allow_cycles
            @warn "update_psi_out_topological! left $n_stuck grounded cell(s) unprocessed this sweep (a genuine cycle in the flow-direction graph) -- allow_cycles = true, so continuing with those cells left at only their partial upstream contributions (their own source term was never added). Pass RecursivePsiOut()/IterativePsiOut() instead if this matters for your domain." maxlog=1
        else
            error("update_psi_out_topological! found a cycle in the flow-direction graph: $n_stuck of $total_masked grounded cell(s) never reached in-degree zero. This is expected on real data, with or without coupling-length smoothing: the routing weight is a per-cell-normalised gradient that is not the gradient of any one surface (see TopologicalPsiOut's docstring above), and TopologicalPsiOut is only exact when this graph is acyclic. Either switch to TapedPsiOut()/RecursivePsiOut()/IterativePsiOut() (they do not fail on a cycle, but the result inside it is only approximate), or pass TopologicalPsiOut(allow_cycles = true) to accept a locally-truncated-but-finite result instead of this error.")
        end
    end

    return nothing

end
