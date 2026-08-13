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
(https://egusphere.copernicus.org/preprints/2024/egusphere-2024-466/egusphere-2024-466-AC1-supplement.pdf). Finally, the 0 <= q <= 1e5 is calculated, with limits set by Frank Pattyn in KORI-ULB
for numerical stability.

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
sweep, converging jointly. `model.sliding_law`'s type determines which `resolve_q!` method runs: `NoSlidingLaw`/`WeertmanSlidingLaw` do not depend on N (the latter contributes a fixed source
term, computed but not iterated on), so they fall back to the original q-only dispatch on `model.dissipation_melt`; `AbstractPressureDependentSlidingLaw` always takes the joint (q, N) loop,
using `model.max_coupling_iters`/`model.coupling_rtol` regardless of `model.dissipation_melt` (which only decides whether the dissipation term is added inside that loop, via `add_dissipation_term!`).
"""
function update_q!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)

    # Update phi0
    update_phi0!(model, grid, state)

    # Fill local minima of phi0 to avoid water getting stuck.
    potential_filling!(model, grid, state)

    # Update the gradients of the geometric potential phi0
    update_potential_gradients!(model, grid)

    # Smoothen the gradients of phi0 to incorporate the concept of stress-gradient coupling.
    update_smoothed_potential_gradients!(model, grid, state)

    # Correction factor from psi_out to q; depends only on the (already updated) potential
    # gradients, so it stays fixed regardless of the dissipation-melt branch taken below.
    dx = grid.dx
    dy = grid.dy
    @. model.corfac = (abs(model.minus_grad_phi0_sx) * dy + abs(model.minus_grad_phi0_sy) * dx) /
                       (sqrt(model.minus_grad_phi0_sx^2.0 + model.minus_grad_phi0_sy^2.0) + 1e-15)

    resolve_q!(model, grid, state, model.dissipation_melt, model.sliding_law)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Add the dissipation melt term |q * grad(phi0)| / L_w to `model.mdot_total` in place. Dispatched on
`model.dissipation_melt` so the off case costs nothing; shared by every `resolve_q!` method so the
term is computed identically regardless of which sliding law is active.
"""
add_dissipation_term!(model::KazmierczakHydroModel, ::DissipationMeltOff) = nothing

function add_dissipation_term!(model::KazmierczakHydroModel, ::DissipationMeltOn)
    @. model.mdot_total += abs(model.q * model.abs_grad_phi0) / model.L_w
    return nothing
end


"""
$(TYPEDSIGNATURES)

With the dissipation melt term off and an N-independent sliding law (`NoSlidingLaw`, which
contributes nothing, or `WeertmanSlidingLaw`, whose tau_b does not depend on N), the water source
has no dependence on q or N: a single pass through the routing algorithm already gives the exact
answer.
"""
function resolve_q!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState,
                     ::DissipationMeltOff, sliding_law::Union{NoSlidingLaw, WeertmanSlidingLaw})

    update_tau_b!(model, state, sliding_law)
    @. model.mdot_total = model.mdot + model.tau_b * model.abs_v_b / model.L_w

    update_psi_out!(model, grid, state)

    @. model.q = min(max(model.psi_out / model.corfac, 0.0), 1e5)

    return nothing

end


"""
$(TYPEDSIGNATURES)

With the dissipation melt term on and an N-independent sliding law, mdot_total = mdot + tau_b*v_b/L_w
+ |q * grad(phi0)| / L_w depends on q (through the dissipation term only -- tau_b*v_b/L_w is fixed
for the whole loop since it does not depend on q or, for these two laws, N), so we Picard-iterate:
recompute the source from the current q, re-run the routing algorithm, and stop once q stops
changing to within model.dissipation_rtol (relative to its own peak magnitude), capped at
model.max_dissipation_iters sweeps. If `model.dissipation_verbose` is set, prints how long the loop
took, whether it converged, and after how many iterations.
"""
function resolve_q!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState,
                     ::DissipationMeltOn, sliding_law::Union{NoSlidingLaw, WeertmanSlidingLaw})

    start_time = time()
    converged  = false
    n_iters    = model.max_dissipation_iters

    update_tau_b!(model, state, sliding_law)

    for iter in 1:model.max_dissipation_iters

        model.q_prev .= model.q

        # Total water source: basal melt mdot, the (fixed, for these laws) frictional-heating term,
        # plus the dissipation melt rate from the current estimate of q (zero on the first sweep,
        # since model.q carries over from the previous call and starts at zero).
        @. model.mdot_total = model.mdot + model.tau_b * model.abs_v_b / model.L_w +
                               abs(model.q * model.abs_grad_phi0) / model.L_w

        # Compute psi_out via a recursive algorithm.
        update_psi_out!(model, grid, state)

        # Limits on q are heuristic and chosen by Frank Pattyn for numerical stability.
        @. model.q = min(max(model.psi_out / model.corfac, 0.0), 1e5)

        q_scale = max(masked_max_abs(grid, model.q, state.mask), 1e-15)
        if masked_max_abs_diff(grid, model.q, model.q_prev, state.mask) <= model.dissipation_rtol * q_scale
            converged = true
            n_iters   = iter
            break
        end

    end

    if model.dissipation_verbose
        elapsed = time() - start_time
        status  = converged ? "converged" : "did NOT converge (hit max_dissipation_iters)"
        println("Water flux Picard loop: $status after $n_iters iteration(s) in $(round(elapsed, digits = 4)) s")
    end

    return nothing

end


"""
$(TYPEDSIGNATURES)

With an N-dependent sliding law (`PowerPlasticSlidingLaw`, `RegularizedCoulombSlidingLaw`), tau_b
depends on N, which is itself downstream of q -- so q and N form a joint fixed point regardless of
`model.dissipation_melt`. Each sweep: recompute tau_b from the current N, add it (plus the
dissipation term, if `model.dissipation_melt` is on) to the water source, route q, then update W
and N from the new q so the next sweep's tau_b uses a fresher N. Stops once both q and N stop
changing (each relative to its own peak magnitude) to within `model.coupling_rtol`, capped at
`model.max_coupling_iters` sweeps. If `model.coupling_verbose` is set, prints how long the loop
took, whether it converged, and after how many iterations.

N starts from whatever `state.N` already holds (zero on a fresh `HydroState`), so the first sweep's
tau_b is zero for these laws and ramps up as the loop proceeds -- an ordinary cold start for Picard
iteration, not a bug.
"""
function resolve_q!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState,
                     dissipation_melt::AbstractDissipationMelt, sliding_law::AbstractPressureDependentSlidingLaw)

    start_time = time()
    converged  = false
    n_iters    = model.max_coupling_iters

    for iter in 1:model.max_coupling_iters

        model.q_prev .= model.q
        model.N_prev .= state.N

        update_tau_b!(model, state, sliding_law)
        @. model.mdot_total = model.mdot + model.tau_b * model.abs_v_b / model.L_w
        add_dissipation_term!(model, dissipation_melt)

        update_psi_out!(model, grid, state)
        @. model.q = min(max(model.psi_out / model.corfac, 0.0), 1e5)

        update_W!(model, grid, state)
        update_N!(model, grid, state)

        q_scale = max(masked_max_abs(grid, model.q, state.mask), 1e-15)
        N_scale = max(masked_max_abs(grid, state.N, state.mask), 1e-15)
        q_converged = masked_max_abs_diff(grid, model.q, model.q_prev, state.mask) <= model.coupling_rtol * q_scale
        N_converged = masked_max_abs_diff(grid, state.N, model.N_prev, state.mask) <= model.coupling_rtol * N_scale

        if q_converged && N_converged
            converged = true
            n_iters   = iter
            break
        end

    end

    if model.coupling_verbose
        elapsed = time() - start_time
        status  = converged ? "converged" : "did NOT converge (hit max_coupling_iters)"
        println("Water flux/effective pressure coupling Picard loop: $status after $n_iters iteration(s) in $(round(elapsed, digits = 4)) s")
    end

    return nothing

end


"""
$(TYPEDSIGNATURES)

Update the water layer thickness W that is part of the HydroState. See Eq. (8) from Kazmierczak et al 2022.
"""
function update_W!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)

    abs_grad_phi0_s_mean = masked_mean(grid, model.abs_grad_phi0_s, state.mask)
    @. state.W = min(model.Wmax, max(model.Wmin, (12 * model.eta_w * model.q / abs_grad_phi0_s_mean)^(1/3)))

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

Updates phi0 and consequently also updates h to reflect changes in phi0. It fills the local minima of phi0 to avoid water getting stuck in there.
"""
function potential_filling!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)

    phi0     = model.phi0
    phi0_tmp = model.phi0_tmp

    phi0_tmp .= phi0
    fill_halo!(phi0_tmp, grid)

    Nx = grid.Nx
    Ny = grid.Ny

    for _ in 1:model.fill_iters
        @inbounds for j in 1:Ny
            for i in 1:Nx
                p = phi0[i, j]
                # Domain edges are treated as zero-gradient (edge-replicated) neighbours, same
                # convention as minus_gradient_x!/minus_gradient_y! -- see grid.jl.
                im1, ip1 = max(i - 1, 1), min(i + 1, Nx)
                jm1, jp1 = max(j - 1, 1), min(j + 1, Ny)
                p1, p2 = phi0[ip1, j], phi0[im1, j]
                p3, p4 = phi0[i, jp1], phi0[i, jm1]
                if p < p1 && p < p2 && p < p3 && p < p4
                    phi0_tmp[i, j] = (p1 + p2 + p3 + p4) / 4.0
                end
            end
        end
        phi0 .= phi0_tmp
        fill_halo!(phi0, grid)
    end

    # Correction to h from potential filling; stored separately so it does not affect other calculations like effective pressure.
    @. model.h = (model.phi0 - model.rho_w * model.g * state.b) / (model.rho_i * model.g)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Compute (the negative of) the gradients of the geometric potential phi0 and its absolute value.
"""
function update_potential_gradients!(model::KazmierczakHydroModel, grid::AbstractHydroGrid)

    minus_gradient_x!(grid, model.minus_grad_phi0_x, model.phi0)
    minus_gradient_y!(grid, model.minus_grad_phi0_y, model.phi0)

    fill_halo!(model.minus_grad_phi0_x, grid)
    fill_halo!(model.minus_grad_phi0_y, grid)

    @. model.abs_grad_phi0 = sqrt(model.minus_grad_phi0_x^2.0 + model.minus_grad_phi0_y^2.0)
    fill_halo!(model.abs_grad_phi0, grid)

    return nothing

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
    h_avg = max(masked_mean(grid, model.h, state.mask), 10.0)

    # Radius of influence
    dx    = grid.dx
    dy    = grid.dy
    Delta = (dx + dy) / 2.0

    scale = h_avg * model.longcoupwater * 2.0

    # Radius of the cone base (= 4 * h_avg * longcoupwater). The cone hits zero at this distance.
    # Although this is 2-5x the Kamb & Echelmeyer (1986) coupling length (4-10x ice thickness for ice sheets),
    # the effective coupling length is the kernel's weighted mean distance from center = width/3
    # = 4/3 * h_avg * longcoupwater, which for longcoupwater=5 gives ~6.7x ice thickness —
    # consistent with Kamb & Echelmeyer. At coarse resolution (16-32 km), the coupling length
    # (6-15 km for 1500 m ice) is smaller than a grid cell, so set longcoupwater = 0.
    width = 2.0 * scale

    if width <= Delta
        scale = Delta / 2.0 + 1.0
    end

    # Kernel size
    maxlevel = 2 * round(Int, width / Delta - 0.5) + 1
    frb      = Int((maxlevel - 1) / 2)

    kernel = zeros(maxlevel, maxlevel)

    for nj in 1:maxlevel, ni in 1:maxlevel
        dist = sqrt((Delta * (ni - frb - 1))^2 +
                    (Delta * (nj - frb - 1))^2) / scale

        kernel[ni, nj] = max(0.0, 1.0 - dist / 2.0)
    end

    kernel ./= sum(kernel)

    convolve!(grid, model.minus_grad_phi0_sx, model.minus_grad_phi0_x, kernel)
    convolve!(grid, model.minus_grad_phi0_sy, model.minus_grad_phi0_y, kernel)

    fill_halo!(model.minus_grad_phi0_sx, grid)
    fill_halo!(model.minus_grad_phi0_sy, grid)

    @. model.abs_grad_phi0_s = abs(model.minus_grad_phi0_sx) + abs(model.minus_grad_phi0_sy)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Helper function to the recursive function to calculate the psi_out for every grid cell that has grounded ice.
"""
function accumulate_psi_out!(model::KazmierczakHydroModel, i, j, grid::AbstractHydroGrid, state::HydroState)

    # If the neighbour does not have grounded ice then return 0
    if state.mask[i, j] != 1.0
        return 0.0
    end

    # If the neighbour has been visited then the psi_out has already been calculated for that cell
    if model.visited[i, j] == 1.0
        return model.psi_out[i, j]
    end

    # Passing the above if statements means we are now visiting cell i, j
    model.visited[i, j] = 1.0

    dx = grid.dx
    dy = grid.dy

    model.psi_out[i, j] = model.mdot_total[i, j] * dx * dy / model.rho_w

    @inbounds for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))

        ni, nj = i + di, j + dj

        # Off the edge of the domain: no neighbouring cell, so no upstream contribution can cross
        # in (the no-flux divide condition, Eq. 2b of Kazmierczak et al. 2024's Γ_d boundary).
        (1 <= ni <= grid.Nx && 1 <= nj <= grid.Ny) || continue

        w = -(model.minus_grad_phi0_sx[ni, nj] * di + model.minus_grad_phi0_sy[ni, nj] * dj) / (model.abs_grad_phi0_s[ni, nj] + 1e-15)

        if w > 0
            model.psi_out[i, j] += accumulate_psi_out!(model, ni, nj, grid, state) * w
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

    @inbounds for j in 1:Ny, i in 1:Nx
        if state.mask[i, j] == 1.0
            accumulate_psi_out!(model, i, j, grid, state)
        end
    end
    
    return nothing

end