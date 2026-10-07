#################################
# Model: Kazmierczak et al 2024 #
#################################


"""
$(TYPEDSIGNATURES)

For the Kazmierczak et al 2024 model, update the distributed water flux q [m2 s-1] via a recursive algorithm following Le Brocq's code for Le Brocq et al 2006 (https://doi.org/10.1016/j.cageo.2006.05.003).
First we update the geometric potential that depends on the icethickness h and bedrock elevation b. Then we fill up the local minima of this potential to avoid the
issue of water getting stuck. We then update the gradients of the potential in x, y. Further, we smooth these gradients with a convolution in order to incorporate
the effects of the stress-gradient coupling. For the latter concept, see for example Eq. (8.98) from Cuffey & Patterson 2010 book, Eq. (15) from Kamb et al 1986,
Gudmundsson 2002, and references therein. Finally, we calculate the scalar water flux out of each grid cell psi_out following the aforementioned recursive algorithm from Le Brocq.
To go from psi_out to q, we use a correction factor, here called corfac, that can be derived using the definition of psi_out given by Eq. (R4) of the referee reports of Kazmierczak et al 2024
(https://egusphere.copernicus.org/preprints/2024/egusphere-2024-466/egusphere-2024-466-AC1-supplement.pdf). Finally, q is clamped to 0 <= q <= perYear2perSecond(1e5) -- KORI-ULB's own SubWaterFlux.m
(https://github.com/FrankPat/Kori-ULB/blob/main/subroutines/SubWaterFlux.m) applies this same 1e5 m2/yr limit set by Frank Pattyn for numerical stability, but to its per-year-native `flw`; q here
is SI (m2/s), so the limit needs the same per-year -> per-second conversion as the other data-loader/unit fixes in this codebase, not the bare literal 1e5.

The water source that feeds the routing algorithm is not just the externally-supplied basal melt mdot, but also, if enabled, the dissipation melt rate ṁ_w = |q * grad(phi0)| / L_w generated
by the flux itself as it moves down the hydraulic gradient, and the frictional-heating term tau_b*v_b/L_w from `model.sliding_law` (Eq. (3), Sec. 2.2.1/2.2.2 of Kazmierczak et al 2024). The
paper drops the dissipation term as negligible; we keep it as an opt-out (`model.dissipation_melt`, set via the `dissipation_melt` keyword of the constructor) rather than always computing it,
since with it off (and no sliding law) `psi_out`/`q` reduce to a single pass with no feedback to resolve at all. When on, we use the unsmoothed gradient of phi0, not the smoothed one used for
flow routing below -- the smoothing is a numerical device for stress-gradient coupling, not part of the actual local driving gradient that does work on the water, and this matches how the
analogous conduit dissipation term (`model.Q * model.abs_grad_phi0`) is already written in `update_N_inf!`.

The dissipation term depends on q, which is itself the output of the routing algorithm, so we Picard-iterate: recompute the source from the current q, re-run the routing algorithm, and stop
once q stops changing (relative to its own peak magnitude) to within `model.dissipation_rtol`, capped at `model.max_dissipation_iters` sweeps as a safety net. The frictional-heating term
depends on the effective pressure N instead, which is itself downstream of q (via `update_N!`, called after `update_q!` in `update_steady_state!`) -- so an N-dependent `model.sliding_law`
(`PowerPlasticSlidingLaw`, `RegularizedCoulombSlidingLaw`) turns this into a second, coupled fixed point on (q, N). Rather than nesting a second Picard loop around the first (which would fully
reconverge q for a stale N every outer sweep), we widen the existing loop: each sweep recomputes tau_b from the current N, routes q, and then also updates W and N in place before the next
sweep, converging jointly. `model.sliding_law`'s type determines which `resolve_q!` method runs: `NoFrictionSlidingLaw`/`WeertmanSlidingLaw` do not depend on N (the latter contributes a fixed source
term, computed but not iterated on), so they fall back to the original q-only dispatch on `model.dissipation_melt`; `AbstractPressureDependentSlidingLaw` always takes the joint (q, N) loop,
using `model.max_qN_iters`/`model.qN_rtol` regardless of `model.dissipation_melt` (which only decides whether the dissipation term is added inside that loop, via `add_dissipation_term!`).

The frictional-heating term is always added to `mdot_total` (via `add_friction_term!`); it is zero
for `NoFrictionSlidingLaw`. The melt rate is only ever built from its terms, so there is no switch for
an "already included" friction term.
"""
function update_q!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)

    # Update phi0
    update_phi0!(model, grid, state)

    # Routing potential and flow directions for model.routing_scheme. For the default GDSWarner: fill
    # local minima of phi0, take its gradients, and smooth them (stress-gradient coupling). See
    # prepare_routing! in routing.jl.
    prepare_routing!(model, grid, state, model.routing_scheme)

    # Correction factor from psi_out to q; depends only on the (already updated) potential
    # gradients, so it stays fixed regardless of the dissipation-melt branch taken below.
    dx = grid.dx
    dy = grid.dy
    # eps(T) rather than a bare Float64 literal like 1e-15: keeps this (and every other
    # division-by-zero guard in this file) in the model's own float type, so a Float32 grid
    # broadcast doesn't get silently promoted to Float64 by a Float64 literal. Computed as a
    # plain scalar BEFORE the `@.` broadcast, not as a literal `eps(T)` call inside it:
    # Oceananigans' `@.` macro lifts every function call syntactically inside a Field-valued
    # broadcast into its own AbstractOperations framework, and `eps` isn't one of the operators
    # it supports -- writing `eps(T)` directly inside `@.` throws a MethodError deep inside
    # Oceananigans.AbstractOperations.broadcasted_to_abstract_operation, not from this file.
    T = eltype(model.minus_grad_phi0_sx)
    epsT = eps(T)
    # x*x rather than x^2.0: Float64^Float64 dispatches to libm's pow() per element (~17x slower
    # than a plain multiply, benchmarked), and x^2 (integer literal) hits the literal_pow issue
    # noted on DarcyWeisbachThickness below -- x*x is both the fast path and Oceananigans-safe.
    @. model.corfac = (abs(model.minus_grad_phi0_sx) * dy + abs(model.minus_grad_phi0_sy) * dx) /
                       (sqrt(model.minus_grad_phi0_sx * model.minus_grad_phi0_sx + model.minus_grad_phi0_sy * model.minus_grad_phi0_sy) + epsT)

    # The gradients (hence the routing graph) just changed, so any recorded TapedPsiOut traversal is stale.
    invalidate_routing_tape!(model)

    resolve_q!(model, grid, state, model.dissipation_melt, model.sliding_law)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Compute the dissipation heat `model.Q_diss` [W/m^2] and add `Q_diss/L_w` to `model.mdot_total` in place
(`Q_diss` is set to zero when the term is off). Dispatched on
`model.dissipation_melt` so the off case costs nothing; shared by every `resolve_q!` method so the
term is computed identically regardless of which sliding law is active.
"""
add_dissipation_term!(model::KazmierczakHydroModel, ::DissipationMeltOff) = (model.Q_diss .= 0; nothing)

add_dissipation_term!(model::KazmierczakHydroModel, ::DissipationMeltOn) =
    add_dissipation_term!(model, model.dissipation_discretization)


"""
$(TYPEDSIGNATURES)

Compute the frictional heat `model.Q_b` [W/m^2] from the current `model.tau_b` and add `Q_b/L_w` to
`model.mdot_total` in place, per `model.friction_discretization`. Shared by every `resolve_q!` method;
`model.tau_b` is always up to date by the time this is called (`update_tau_b!` runs first in every
`resolve_q!` method).
"""
add_friction_term!(model::KazmierczakHydroModel) = add_friction_term!(model, model.friction_discretization)

function add_friction_term!(model::KazmierczakHydroModel, ::CellCentredFriction)
    @. model.Q_b = model.tau_b * model.abs_v_b
    @. model.mdot_total += model.Q_b / model.L_w
    return nothing
end


"""
$(TYPEDSIGNATURES)

Computes psi_out for the current sweep, dispatching on `model.psi_out_algorithm`
(`RecursivePsiOut()`, `IterativePsiOut()`, or `TopologicalPsiOut()` -- see
`AbstractPsiOutAlgorithm`'s docstring in model.jl) to the recursive `update_psi_out!`, the
stack-based `update_psi_out_iterative!`, or the sorted single-pass `update_psi_out_topological!`.
Every `resolve_q!` method calls this instead of `update_psi_out!` directly, so the algorithm choice
applies uniformly regardless of which sliding law/dissipation-melt combination is active.
"""
route_psi_out!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState) =
    route_psi_out!(model, grid, state, model.psi_out_algorithm)

route_psi_out!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::RecursivePsiOut) =
    update_psi_out!(model, grid, state)

route_psi_out!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::IterativePsiOut) =
    update_psi_out_iterative!(model, grid, state)

route_psi_out!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, algorithm::TopologicalPsiOut) =
    update_psi_out_topological!(model, grid, state, algorithm.allow_cycles)

route_psi_out!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::TapedPsiOut) =
    update_psi_out_taped!(model, grid, state)


"""
$(TYPEDSIGNATURES)

Mark `model`'s recorded `TapedPsiOut` traversal as stale, so the next `route_psi_out!` re-records it.
"""
invalidate_routing_tape!(model::KazmierczakHydroModel) = (model.routing_tape.valid = false; nothing)


"""
$(TYPEDSIGNATURES)

`TapedPsiOut`'s routing: record the traversal if the tape is stale, then replay it with the current
`mdot_total`. See [`TapedPsiOut`](@ref).
"""
function update_psi_out_taped!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)
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
function record_routing_tape!(tape::RoutingTape, model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)
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

Convert the routed flux `psi_out` [m3/s] to the distributed flux `q` [m2/s] via `q = psi_out / corfac`,
clamped to `[q_min, q_max]`. `corfac` is exactly 0 at a cell whose (smoothed) potential gradient is
identically zero, so the bare division gave 0/0 = NaN (net-refreezing cell, `psi_out == 0`) or Inf
(`psi_out > 0`) there, and `max`/`min` propagate NaN. Adding `eps(T)` to the denominator, the same
guard every other division in this file uses, keeps q finite (0 for `psi_out == 0`). For `psi_out > 0`
on such a cell q is still huge, since the flow direction is genuinely undefined -- use `q_max` if you
need to bound it. `epsT` is a plain scalar computed outside the broadcast for the reason given in
`update_q!`.
"""
function update_q_from_psi_out!(model::KazmierczakHydroModel)

    epsT = eps(eltype(model.corfac))
    @. model.q = min(max(model.psi_out / (model.corfac + epsT), model.q_min), model.q_max)

    return nothing

end


"""
$(TYPEDSIGNATURES)

With the dissipation melt term off and an N-independent sliding law (`NoFrictionSlidingLaw`,
which contributes nothing; `PrescribedFieldSlidingLaw`, whose tau_b is a fixed externally-supplied
field; or `WeertmanSlidingLaw`, whose tau_b does not depend on N), the water source has no dependence
on q or N: a single pass through the routing algorithm already gives the exact answer.
The frictional heat tau_b*v_b is always part of the source (zero for `NoFrictionSlidingLaw`).
"""
function resolve_q!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState,
                     ::DissipationMeltOff, sliding_law::Union{NoFrictionSlidingLaw, PrescribedFieldSlidingLaw, WeertmanSlidingLaw})

    update_tau_b!(model, state, sliding_law)
    @. model.mdot_total = model.mdot_fixed + model.i_eb
    add_friction_term!(model)

    route_psi_out!(model, grid, state)

    update_q_from_psi_out!(model, grid, state)

    return nothing

end


"""
$(TYPEDSIGNATURES)

With the dissipation melt term on and an N-independent sliding law, mdot_total = mdot_fixed + tau_b*v_b/L_w
+ |q * grad(phi0)| / L_w depends on q (through the dissipation term only -- tau_b*v_b/L_w is fixed
for the whole loop since it does not depend on q or, for these laws, N), so we Picard-iterate:
recompute the source from the current q, re-run the routing algorithm, and stop once q stops
changing to within model.dissipation_rtol (relative to its own peak magnitude), capped at
model.max_dissipation_iters sweeps. If `model.dissipation_verbose` is set, logs (via @info) how long the loop
took, whether it converged, and after how many iterations.
"""
function resolve_q!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState,
                     ::DissipationMeltOn, sliding_law::Union{NoFrictionSlidingLaw, PrescribedFieldSlidingLaw, WeertmanSlidingLaw})

    start_time = time()
    converged  = false
    n_iters    = model.max_dissipation_iters

    update_tau_b!(model, state, sliding_law)

    for iter in 1:model.max_dissipation_iters

        model.q_prev .= model.q

        # Total water source: the fixed part mdot_fixed, the (fixed, for these laws) frictional-heating term, plus the
        # dissipation melt rate from the current estimate of q (zero on the first sweep, since
        # model.q carries over from the previous call and starts at zero). Routed through
        # add_dissipation_term! (rather than reimplementing the formula inline) so this stays the
        # single source of truth for it, matching the N-dependent resolve_q! method below.
        @. model.mdot_total = model.mdot_fixed + model.i_eb
        add_friction_term!(model)
        add_dissipation_term!(model, DissipationMeltOn())

        # Compute psi_out via whichever algorithm model.psi_out_algorithm selects.
        route_psi_out!(model, grid, state)

        update_q_from_psi_out!(model, grid, state)

        q_scale = max(masked_max_abs(grid, model.q, state.mask), eps(eltype(model.q)))
        if masked_max_abs_diff(grid, model.q, model.q_prev, state.mask) <= model.dissipation_rtol * q_scale
            converged = true
            n_iters   = iter
            break
        end

    end

    if model.dissipation_verbose
        elapsed = time() - start_time
        status  = converged ? "converged" : "did NOT converge (hit max_dissipation_iters)"
        @info "Water flux Picard loop: $status after $n_iters iteration(s) in $(round(elapsed, digits = 4)) s"
    end

    return nothing

end


"""
$(TYPEDSIGNATURES)

With an N-dependent sliding law (`PowerPlasticSlidingLaw`, `RegularizedCoulombSlidingLaw`), tau_b
depends on N, which is itself downstream of q -- so q and N form a joint fixed point regardless of
`model.dissipation_melt`. Each sweep: recompute tau_b from the current N, add its frictional heat to
the water source (plus the dissipation term, if `model.dissipation_melt` is on), route q, then update
N from the new q so the next sweep's tau_b uses a fresher N. Stops once both q and N stop
changing (each relative to its own peak magnitude) to within `model.qN_rtol`, capped at
`model.max_qN_iters` sweeps. If `model.qN_verbose` is set, logs (via @info) how long the loop
took, whether it converged, and after how many iterations.

N starts from whatever `state.N` already holds (zero on a fresh `HydroState`), so the first sweep's
tau_b is zero for these laws and ramps up as the loop proceeds -- an ordinary cold start for Picard
iteration, not a bug.
"""
function resolve_q!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState,
                     dissipation_melt::AbstractDissipationMelt, sliding_law::AbstractPressureDependentSlidingLaw)

    start_time = time()
    converged  = false
    n_iters    = model.max_qN_iters

    for iter in 1:model.max_qN_iters

        model.q_prev .= model.q
        model.N_prev .= state.N

        update_tau_b!(model, state, sliding_law)
        @. model.mdot_total = model.mdot_fixed + model.i_eb
        add_friction_term!(model)
        add_dissipation_term!(model, dissipation_melt)

        route_psi_out!(model, grid, state)
        update_q_from_psi_out!(model, grid, state)

        update_N!(model, grid, state)

        q_scale = max(masked_max_abs(grid, model.q, state.mask), eps(eltype(model.q)))
        N_scale = max(masked_max_abs(grid, state.N, state.mask), eps(eltype(state.N)))
        q_converged = masked_max_abs_diff(grid, model.q, model.q_prev, state.mask) <= model.qN_rtol * q_scale
        N_converged = masked_max_abs_diff(grid, state.N, model.N_prev, state.mask) <= model.qN_rtol * N_scale

        if q_converged && N_converged
            converged = true
            n_iters   = iter
            break
        end

    end

    if model.qN_verbose
        elapsed = time() - start_time
        status  = converged ? "converged" : "did NOT converge (hit max_qN_iters)"
        @info "Water flux/effective pressure coupling Picard loop: $status after $n_iters iteration(s) in $(round(elapsed, digits = 4)) s"
    end

    return nothing

end


"""
$(TYPEDSIGNATURES)

Update the water layer thickness `state.W`, dispatching on `model.water_thickness_algorithm`
(`ArealConduitThickness()`, `DarcyWeisbachThickness()`, or `LaminarThickness()` -- see
`AbstractWaterThicknessAlgorithm`'s docstring in model.jl) to whichever single closure is selected,
so only that closure's fields are touched each call.

Must run after `update_N!` (specifically after its `update_S_inf!` call), not before --
`ArealConduitThickness` reads `model.S_inf`, which `update_N!` is what keeps current.
`update_steady_state!` in run.jl calls `update_q!`, then `update_N!`, then `update_W!` in that order
for this reason; `state.W` itself feeds back into nothing else in the model, so it is safe to compute
last.
"""
function update_W!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)
    update_W!(model, grid, state, model.water_thickness_algorithm)
    return nothing
end


"""
$(TYPEDSIGNATURES)

`ArealConduitThickness`: report `model.S_inf / model.l_c` -- the conduit cross-section smeared over
the inter-conduit spacing -- as `state.W`, unclamped. See `AbstractWaterThicknessAlgorithm`'s
docstring in model.jl for the derivation and why `S_inf` (not `H`) is the right numerator.
"""
function update_W!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::ArealConduitThickness)
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
docstring in model.jl), or, with `gradient_convention = LocalGradient()`, the local, per-cell
gradient instead (matching the convention `update_S_inf!` uses). The turbulent analogue of
`LaminarThickness` below, and the closure consistent with K24's own turbulent-flow assumption for `q`
(unlike `LaminarThickness`) -- clamped to `[Wmin, Wmax]` since it represents the same kind of
thin-sheet quantity. The `+ eps(T)` guards degenerate cells where `abs_grad_phi0` is exactly zero
(e.g. flat cells outside the glacier extent). Uses `q*q` rather than `q^2.0`: `Float64^Float64`
dispatches to libm's `pow()` per element, ~17x slower (benchmarked) than a plain multiply for no
numerical difference, and `q^2` (integer literal) hits a separate issue -- `@.`'s `literal_pow`
rewrite isn't supported by Oceananigans' `AbstractOperation` broadcasting.
"""
function update_W!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, algorithm::DarcyWeisbachThickness)
    update_W_darcy_weisbach!(model, grid, state, algorithm.gradient_convention)
    return nothing
end

function update_W_darcy_weisbach!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::LocalGradient)
    # eps(T) computed as a plain scalar BEFORE the `@.` broadcast -- see update_q!'s own note on
    # why `eps(T)` cannot be written literally inside an Oceananigans `@.` broadcast expression.
    T = eltype(model.abs_grad_phi0)
    epsT = eps(T)
    @. state.W = min(model.Wmax, max(model.Wmin,
        (model.f * model.rho_w * model.q * model.q / (4 * model.abs_grad_phi0 + epsT))^(1/3)))
    return nothing
end

function update_W_darcy_weisbach!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::MeanGradient)
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
closure -- see `AbstractWaterThicknessAlgorithm`'s docstring in model.jl for why this closure is not
the default (`DarcyWeisbachThickness` is). Uses a single domain-mean smoothed gradient magnitude by
default (`gradient_convention = MeanGradient()`), faithfully matching Kori-ULB's own
`mean(gdsmag(...))` in `SubWaterFlux.m`, and matching `DarcyWeisbachThickness`'s own default too (kept
consistent between the package's two sheet-flow closures); pass `gradient_convention =
LocalGradient()` for the local-per-cell-gradient variant instead, e.g. to isolate how much of a
closure's spatial pattern comes from that domain-mean averaging versus from its own physics --
`DarcyWeisbachThickness`'s `gradient_convention` runs the identical comparison for the turbulent
closure.
"""
function update_W!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, algorithm::LaminarThickness)
    update_W_laminar!(model, grid, state, algorithm.gradient_convention)
    return nothing
end

function update_W_laminar!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::MeanGradient)
    abs_grad_phi0_s_mean = masked_mean(grid, model.abs_grad_phi0_s, state.mask)
    @. state.W = min(model.Wmax, max(model.Wmin, (12 * model.eta_w * model.q / abs_grad_phi0_s_mean)^(1/3)))
    return nothing
end

function update_W_laminar!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, ::LocalGradient)
    # eps(T) computed as a plain scalar BEFORE the `@.` broadcast -- see update_q!'s own note.
    T = eltype(model.abs_grad_phi0_s)
    epsT = eps(T)
    @. state.W = min(model.Wmax, max(model.Wmin, (12 * model.eta_w * model.q / (model.abs_grad_phi0_s + epsT))^(1/3)))
    return nothing
end


"""
$(TYPEDSIGNATURES)

Update the geometric potential phi0 and fill halo points.
"""
function update_phi0!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)

    @. model.phi0 = model.rho_i * model.g * state.h + model.rho_w * model.g * state.b
    fill_halo!(model.phi0, grid)

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
function potential_filling!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState; from_true_potential::Bool = true)

    phi0     = model.phi0_filled
    phi0_tmp = model.phi0_tmp

    # Start from the true potential, unless the caller has already put the surface to fill (e.g. the
    # smoothed potential of a non-GDS routing scheme, see prepare_routing!) into phi0_filled.
    from_true_potential && (phi0 .= model.phi0)
    fill_halo!(phi0, grid)
    phi0_tmp .= phi0
    fill_halo!(phi0_tmp, grid)

    fill_potential!(model, grid, state, model.fill_algorithm)

    # Ice thickness consistent with the filled potential; stored separately so it does not affect other calculations like effective pressure.
    @. model.h = (model.phi0_filled - model.rho_w * model.g * state.b) / (model.rho_i * model.g)

    return nothing

end


"""
$(TYPEDSIGNATURES)

`JacobiFill`: the original filling, `model.fill_iters` passes of raising every strict local minimum
of `model.phi0_filled` to the mean of its 4 neighbours.
"""
function fill_potential!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, alg::Union{JacobiFill, LowestNeighbourFill})

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
        fill_halo!(phi0, grid)
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

Outlets (seeds) are the places water can leave the grounded hydrological system: every non-grounded
cell next to a grounded one (grounding line, ice margin) and every grounded cell on the domain edge
(the routing treats the edge as a sink, as `JacobiFill` does by never filling edge cells). Starting from
the lowest outlet, cells are visited in increasing order of (filled) potential; each newly reached
grounded neighbour that is not higher than the cell it was reached from is raised to that cell's
value plus `epsilon`. When the queue is empty every grounded cell has a path to an outlet along which
the filled potential strictly decreases, so no grounded pit or flat is left -- in a single pass,
O(n log n). Non-grounded cells keep their true potential.
"""
function fill_potential!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, alg::PriorityFloodFill)

    phi = model.phi0_filled
    mask = state.mask
    Nx, Ny = grid.Nx, grid.Ny
    T = eltype(phi)
    epsilon = T(alg.epsilon)

    closed = model.fill_stamp          # reused as the "already queued" flag (0/1)
    fill!(closed, Int32(0))
    heap_p = T[]
    heap_k = Int32[]
    sizehint!(heap_p, Nx * Ny ÷ 4); sizehint!(heap_k, Nx * Ny ÷ 4)

    grounded(i, j) = mask[i, j] == 1.0
    @inbounds for j in 1:Ny, i in 1:Nx
        seed = if grounded(i, j)
            i == 1 || i == Nx || j == 1 || j == Ny
        else
            (i > 1 && grounded(i - 1, j)) || (i < Nx && grounded(i + 1, j)) ||
            (j > 1 && grounded(i, j - 1)) || (j < Ny && grounded(i, j + 1))
        end
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

    fill_halo!(phi, grid)
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
            phi0_tmp[i, j] = lowest ? min(p1, p2, p3, p4) : (p1 + p2 + p3 + p4) / 4.0
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

Compute the gradients of the geometric potential. `model.abs_grad_phi0` (used by `S_inf`, `N_inf` and the
dissipation term) is the magnitude of the gradient of the TRUE potential `model.phi0`. The components
`model.minus_grad_phi0_x/y` (negative gradient, i.e. the flow direction) are taken from the FILLED
potential `model.phi0_filled`, since they only feed the routing (via the smoothing and `psi_out`).
"""
function update_potential_gradients!(model::KazmierczakHydroModel, grid::AbstractHydroGrid)

    # True potential first: its components are used only to form the magnitude, and the buffers are
    # then reused for the filled potential's components below.
    minus_gradient_x!(grid, model.minus_grad_phi0_x, model.phi0)
    minus_gradient_y!(grid, model.minus_grad_phi0_y, model.phi0)
    fill_halo!(model.minus_grad_phi0_x, grid)
    fill_halo!(model.minus_grad_phi0_y, grid)

    # x*x rather than x^2.0 -- see the comment on the equivalent corfac computation in update_q!.
    @. model.abs_grad_phi0 = sqrt(model.minus_grad_phi0_x * model.minus_grad_phi0_x + model.minus_grad_phi0_y * model.minus_grad_phi0_y)
    fill_halo!(model.abs_grad_phi0, grid)

    minus_gradient_x!(grid, model.minus_grad_phi0_x, model.phi0_filled)
    minus_gradient_y!(grid, model.minus_grad_phi0_y, model.phi0_filled)
    fill_halo!(model.minus_grad_phi0_x, grid)
    fill_halo!(model.minus_grad_phi0_y, grid)

    return nothing

end


"""
$(TYPEDSIGNATURES)

The normalised Kamb & Echelmeyer (1986) stress-gradient-coupling kernel (a cone) for mean grounded
ice thickness `h_avg`, used by `update_smoothed_potential_gradients!` (smoothing the gradient
components) and by `prepare_routing!` for the non-GDS routing schemes (smoothing the potential).
"""
function coupling_kernel(model::KazmierczakHydroModel, grid::AbstractHydroGrid, h_avg)

    # Grid spacing in each direction. The kernel below computes each cell's distance from the
    # center using dx and dy separately, rather than collapsing both to a single isotropic
    # Delta = (dx+dy)/2 -- so its nonzero support is a genuine circle in physical space (an
    # ellipse in these grid-index coordinates whenever dx != dy), correct for any cell aspect
    # ratio rather than only for square cells.
    dx = grid.dx
    dy = grid.dy

    scale = h_avg * model.longcoupwater * 2.0

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
    width = 2.0 * scale

    # Below Delta_min (the finer of the two spacings), `width` wouldn't resolve to even one grid
    # cell in the tighter direction, so bump scale up to guarantee the kernel spans at least
    # ~1 cell there rather than degenerating to a single-cell no-op.
    Delta_min = min(dx, dy)

    if width <= Delta_min
        scale = Delta_min / 2.0 + 1.0
    end

    # Kernel size, independent per axis (frb_x, frb_y): cached_fft_convolve! (fft_convolution.jl)
    # supports a rectangular kernel array, so each axis is sized from its own spacing rather than
    # both being forced to the size the finer axis would need (which wasted array space and FFT
    # work on the coarser axis without changing the result).
    maxlevel_x = 2 * round(Int, width / dx - 0.5) + 1
    maxlevel_y = 2 * round(Int, width / dy - 0.5) + 1
    frb_x = Int((maxlevel_x - 1) / 2)
    frb_y = Int((maxlevel_y - 1) / 2)

    kernel = zeros(maxlevel_x, maxlevel_y)

    for nj in 1:maxlevel_y, ni in 1:maxlevel_x
        # True physical (Euclidean) distance from the kernel center, using dx and dy separately.
        dist = sqrt((dx * (ni - frb_x - 1))^2 +
                    (dy * (nj - frb_y - 1))^2) / scale

        kernel[ni, nj] = max(0.0, 1.0 - dist / 2.0)
    end

    kernel ./= sum(kernel)

    return kernel

end


"""
$(TYPEDSIGNATURES)

Update the smoothed geometric potentials to incorporate the effects of the stress-gradient coupling. See also the description of the update_q! function.

The water flux at a given point is influenced by variations in ice thickness some distance away. To account for this we perform a convolution of the gradient of the potential
such that the influence of nearby points is now incorporated into the value of the gradient of the potential at that point.
"""
function update_smoothed_potential_gradients!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)

    if model.longcoupwater == 0.0
        model.minus_grad_phi0_sx .= model.minus_grad_phi0_x
        model.minus_grad_phi0_sy .= model.minus_grad_phi0_y
        @. model.abs_grad_phi0_s = abs(model.minus_grad_phi0_x) + abs(model.minus_grad_phi0_y)
        return nothing
    end

    # Average grounded-ice thickness
    kernel = coupling_kernel(model, grid, max(masked_mean(grid, model.h, state.mask), 10.0))

    convolve!(grid, model.minus_grad_phi0_sx, model.minus_grad_phi0_x, kernel)
    convolve!(grid, model.minus_grad_phi0_sy, model.minus_grad_phi0_y, kernel)

    fill_halo!(model.minus_grad_phi0_sx, grid)
    fill_halo!(model.minus_grad_phi0_sy, grid)

    @. model.abs_grad_phi0_s = abs(model.minus_grad_phi0_sx) + abs(model.minus_grad_phi0_sy)

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
function accumulate_psi_out!(model::KazmierczakHydroModel, i, j, grid::AbstractHydroGrid, state::HydroState, call_count::Base.RefValue{Int})

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
    model.psi_out[i, j] = max(0.0, model.psi_out[i, j])

    return model.psi_out[i, j]

end


"""
$(TYPEDSIGNATURES)

Recursive function to calculate the psi_out for every grid cell that has grounded ice.
We initialize psi_out to -1 since it is by definition positive semi-definite and hence we
know that if a grid point has negative psi_out, it is still unvisited.
"""
function update_psi_out!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)

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
function update_psi_out_iterative!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)

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
                model.psi_out[si, sj] = max(0.0, model.psi_out[si, sj])
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
acyclic; see `TopologicalPsiOut`'s docstring in model.jl for why real ice-sheet data isn't reliably
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
function update_psi_out_topological!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState, allow_cycles::Bool)

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
        model.psi_out[i, j] = max(0.0, model.psi_out[i, j] + model.mdot_total[i, j] * dx * dy / model.rho_w)

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
            error("update_psi_out_topological! found a cycle in the flow-direction graph: $n_stuck of $total_masked grounded cell(s) never reached in-degree zero. This is expected whenever longcoupwater != 0 (the smoothed gradient the routing weight uses is not guaranteed to be a conservative field -- see TopologicalPsiOut's docstring in model.jl) and TopologicalPsiOut is only exact when this graph is acyclic. Either use longcoupwater = 0, switch to RecursivePsiOut()/IterativePsiOut(), or pass TopologicalPsiOut(allow_cycles = true) to accept a locally-truncated-but-finite result instead of this error.")
        end
    end

    return nothing

end