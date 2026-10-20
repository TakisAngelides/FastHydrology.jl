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

The sensible heat ct*cw*rho_w*(q . grad(p_w)) (`model.sensible_heat`, off by default) depends on q and, through the water pressure p_w = Po - N, on N. With it on, `resolve_q!`
takes the same joint (q, N) loop whatever the sliding law, adding the term every sweep via `add_sensible_term!`.

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
    # broadcast doesn't get silently promoted to Float64 by a Float64 literal. Computed once as a
    # plain scalar before the `@.` broadcast rather than per cell inside it.
    T = eltype(model.minus_grad_phi0_sx)
    epsT = eps(T)
    # x*x rather than x^2.0: Float64^Float64 dispatches to libm's pow() per element (~17x slower
    # than a plain multiply, benchmarked), and x^2 (integer literal) hits the literal_pow issue
    # noted on DarcyWeisbachThickness below -- x*x is the fast path.
    @. model.corfac = (abs(model.minus_grad_phi0_sx) * dy + abs(model.minus_grad_phi0_sy) * dx) /
                       (sqrt(model.minus_grad_phi0_sx * model.minus_grad_phi0_sx + model.minus_grad_phi0_sy * model.minus_grad_phi0_sy) + epsT)

    # The gradients (hence the routing graph) just changed, so any recorded TapedPsiOut traversal is stale.
    invalidate_routing_tape!(model)

    resolve_q!(model, grid, state, model.dissipation_melt, model.sliding_law, model.sensible_heat)

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
                     ::DissipationMeltOff, sliding_law::Union{NoFrictionSlidingLaw, PrescribedFieldSlidingLaw, WeertmanSlidingLaw},
                     ::SensibleHeatOff)

    update_tau_b!(model, state, sliding_law)
    @. model.mdot_total = model.mdot_fixed + model.i_eb
    add_friction_term!(model)
    model.Q_sens .= 0

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
                     ::DissipationMeltOn, sliding_law::Union{NoFrictionSlidingLaw, PrescribedFieldSlidingLaw, WeertmanSlidingLaw},
                     ::SensibleHeatOff)

    start_time = time()
    converged  = false
    n_iters    = model.max_dissipation_iters

    update_tau_b!(model, state, sliding_law)
    model.Q_sens .= 0

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
`model.dissipation_melt`. The sensible heat (`model.sensible_heat` on) couples q and N the same way,
through the water pressure p_w = Po - N, so this method also runs for it with any sliding law. Each
sweep: recompute tau_b from the current N, add its frictional heat to the water source (plus the
dissipation term, if `model.dissipation_melt` is on, and the sensible heat from the current q and N, if
`model.sensible_heat` is on), route q, then update N from the new q so the next sweep uses a fresher N. Stops once both q and N stop
changing (each relative to its own peak magnitude) to within `model.qN_rtol`, capped at
`model.max_qN_iters` sweeps. If `model.qN_verbose` is set, logs (via @info) how long the loop
took, whether it converged, and after how many iterations.

N starts from whatever `state.N` already holds (zero on a fresh `HydroState`), so the first sweep's
tau_b is zero for these laws and ramps up as the loop proceeds -- an ordinary cold start for Picard
iteration, not a bug.
"""
function resolve_q!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState,
                     dissipation_melt::AbstractDissipationMelt, sliding_law::AbstractSlidingLaw,
                     sensible_heat::AbstractSensibleHeat)

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
        add_sensible_term!(model, grid, state, sensible_heat)

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
