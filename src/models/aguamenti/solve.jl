###########################################
# Model: Aguamenti v1.0 (Gregov et al.)   #
###########################################

# The reduced problem (Sect. 2.8, Eqs. 26a-26f): routing, flux balance and regimes are algebraic in
# time and solved together by a Picard iteration on (q, N); the drainage state chi and the till
# reservoir N_t are advanced once per step.


# How a solve treats the two state fields.
struct SteadyMode end                       # till in equilibrium with N, chi in the steady state of Eq. (22)
struct StepMode{T, D}                       # advance N_t and chi over dt
    dt    ::T
    dH_dt ::D                               # rate of change of the ice thickness [m/s] (field), or nothing
end


"""
$(TYPEDSIGNATURES)

Build the routing graph of the engine for the current potential: the geometric `phi0`
(`GeometricRouting`), or `phi0 - N_route` (`HydraulicRouting`). The engine's
`abs_grad_phi0` is reset to the gradient of the true `phi0` afterwards (the elements use `Psi`, A2),
and the correction factor from `psi_out` to `q` and the routing tape are refreshed.
"""
function prepare_aguamenti_routing!(model::AguamentiHydroModel, grid::AbstractHydroGrid, state::HydroState, with_N::Bool)
    r = model.routing
    if with_N
        model.phi0_true .= r.phi0
        @. r.phi0 = ifelse(state.mask == GROUNDED, r.phi0 - model.N_route, r.phi0)
    end
    prepare_routing!(r, grid, state, r.routing_scheme)
    if with_N
        r.phi0 .= model.phi0_true
        update_true_gradient_magnitude!(r, grid, r.gradient_discretization)
    end
    epsT = eps(eltype(r.corfac))
    dx, dy = grid.dx, grid.dy
    @. r.corfac = (abs(r.minus_grad_phi0_sx) * dy + abs(r.minus_grad_phi0_sy) * dx) /
                  (sqrt(r.minus_grad_phi0_sx * r.minus_grad_phi0_sx + r.minus_grad_phi0_sy * r.minus_grad_phi0_sy) + epsT)
    invalidate_routing_tape!(r)
    return nothing
end


"""
$(TYPEDSIGNATURES)

Replay the engine's routing tape (see [`TapedPsiOut`](@ref)) with Aguamenti's sources and sinks. Each
grounded cell starts from its own supply `mdot_b/rho_w dx dy` and receives its upstream inflow, in the
tape's order (every cell after all the cells that drain into it); when the cell is complete, with
`W` [m3/s] the water available (Eq. 9a, `(mdot_b/rho_w + mdot_in) dx dy`):

- a cold cell refreezes everything it receives (`refreeze`) and passes nothing on;
- otherwise the till takes `uptake(i, j, W)` [m/s] (see [`till_uptake`](@ref)), which never exceeds
  `W/(dx dy)`; a freeze-on deficit (`W < 0`) is taken from the till down to `-release_max` (Eq. 9b), and
  what remains is `freeze_deficit`. A cell left without water is dry.

`psi_out` is the outflow [m3/s], `mdot_t` the exchange [m/s] (positive into the till), `dry` 1 for dry
or cold cells.
"""
function replay_aguamenti_tape!(uptake, psi_out, mdot_t, dry, refreeze, freeze_deficit, mdot_b, release_max, cold, mask,
                                tape::RoutingTape, Nx, Ny, dx, dy, rho_w)
    z = zero(eltype(psi_out))
    area = dx * dy
    @inbounds for j in 1:Ny, i in 1:Nx
        psi_out[i, j] = (mask[i, j] == GROUNDED && cold[i, j] == 0) ? mdot_b[i, j] / rho_w * area : z
        mdot_t[i, j] = z; refreeze[i, j] = z; dry[i, j] = z; freeze_deficit[i, j] = z
    end
    dst_i, dst_j, src_i, src_j, ws = tape.dst_i, tape.dst_j, tape.src_i, tape.src_j, tape.w
    @inbounds for k in eachindex(ws)
        ci, cj, ni = dst_i[k], dst_j[k], src_i[k]
        if ni == 0
            W = psi_out[ci, cj]
            if cold[ci, cj] != 0
                refreeze[ci, cj] = W / area
                dry[ci, cj] = one(z)
                psi_out[ci, cj] = z
                continue
            end
            u = uptake(ci, cj, W)
            if W < 0
                # freeze-on deficit: from the till, as far as it holds water above its residual content
                u_min = -release_max[ci, cj]
                u = max(min(u, W / area), u_min)
                freeze_deficit[ci, cj] = u - W / area
            end
            mdot_t[ci, cj] = u
            out = W - u * area
            if out <= 0
                dry[ci, cj] = one(z)
                psi_out[ci, cj] = z
            else
                psi_out[ci, cj] = out
            end
        else
            psi_out[ci, cj] += psi_out[ni, src_j[k]] * ws[k]
        end
    end
    return nothing
end


# No till exchange (steady state): the till takes nothing.
struct NoUptake end
(::NoUptake)(i, j, W) = zero(W)

"""
$(TYPEDSIGNATURES)

Till uptake of one cell over a step, solved together with the cell's own flux balance: the till relaxes
towards the `N` of the water that remains in the cell after its uptake, so the uptake `u` [m/s] solves
`u = demand(N(q(W - u dx dy)))`, where `q = outflow / corfac` (the routing's psi-to-q conversion of the
cell), `N(q)` is the flux balance with its regimes and grounding-line correction (Eqs. 21-24) and
`demand(N)` the exchange of the till relaxation over the step ([`till_relax`](@ref), Eq. 20). The
right-hand side decreases with `u` (less water, higher `N`, a smaller demand), so the root is unique
and found by bisection on `[-release_max, W/(dx dy)]`; at `u = W/(dx dy)` the till takes all the water
and the cell is dry (Eq. 9a). Solving the cell in the routing order makes the uptake consistent with
the inflow from upstream within one replay -- a till demand lagged by a sweep instead makes whole flow
lines flip between wet and dry.
"""
struct TillUptake{M, ST, TT, D}
    model ::M
    state ::ST
    dt    ::TT
    dH_dt ::D
    area  ::TT
end

function (tu::TillUptake)(i, j, W)
    m, state = tu.model, tu.state
    p, ws, r = m.params, m.workspace, m.routing
    z = zero(W)
    @inbounds begin
        (ws.kappa[i, j] == 0 || W <= 0) && return z
        p_i, kappa, Psi = ws.p_i[i, j], ws.kappa[i, j], ws.Psi[i, j]
        u_b, A, D_s, chi = r.abs_v_b[i, j], r.A_visc[i, j], ws.D_s[i, j], ws.chi[i, j]
        N_t, h_t, k_t, h, b = ws.N_t[i, j], ws.h_t[i, j], ws.k_t[i, j], state.h[i, j], state.b[i, j]
        corfac = r.corfac[i, j] + eps(z)
        dH = tu.dH_dt === nothing ? z : tu.dH_dt[i, j]
    end
    N0 = strength_bound(p, kappa, ws.c_0[i, j])
    q_on = kappa < 1 ? onset_flux(p, Psi, u_b) : oftype(z, Inf)
    q_sn = kappa < 1 ? first(saddle_node_flux(p, kappa, Psi, u_b, A, D_s)) : z
    function N_of(u)
        q = max(W - u * tu.area, z) / corfac
        N_star, _ = flux_balance_root(p, q, chi, kappa, Psi, u_b, A, D_s, p_i, q_sn; developing = q >= q_on)
        N_inf, _ = regime_pressure(p, N_star, p_i, N0)
        return grounding_line_correction(p, N_inf, h, b)
    end
    demand(u) = till_relax(p, N_t, N_of(u), tu.dt, h_t, k_t, p_i, dH)[2]
    lo, hi = -ws.release_max[i, j], W / tu.area
    demand(hi) >= hi && return hi                   # the till absorbs all the water: dry (Eq. 9a)
    demand(lo) <= lo && return lo
    for _ in 1:p.root_iters
        mid = (lo + hi) / 2
        demand(mid) > mid ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end


# Per-column kernels (threaded like K24's update_N!): the flux balance of every grounded cell.
function aguamenti_columns!(kernel!, Ny, args...)
    if Threads.nthreads() == 1
        for j in 1:Ny
            kernel!(j, args...)
        end
    else
        Threads.@threads for j in 1:Ny
            kernel!(j, args...)
        end
    end
    return nothing
end


"""
$(TYPEDSIGNATURES)

Flux balance of every grounded cell for the current `q` (Sect. 2.6): the onset and saddle-node fluxes,
the growth rate, the root `N*` on the active branch, the regime and `N_inf` (Eq. 23), the grounding-line
correction (Eq. 24) into `model.N_wet`, and the element fluxes at that pressure. With `steady_chi` the
channel indicator is first set to the steady state of Eq. (22) for this `q`.
"""
function update_flux_balance!(model::AguamentiHydroModel, grid::AbstractHydroGrid, state::HydroState, steady_chi::Bool)
    aguamenti_columns!(flux_balance_column!, grid.Ny, model, model.params, model.routing, state, grid.Nx, steady_chi)
    return nothing
end

function flux_balance_column!(j, model, p::AguamentiParams, r, state, Nx, steady_chi)
    ws = model.workspace
    z = zero(eltype(ws.N_wet))
    @inbounds for i in 1:Nx
        if state.mask[i, j] != GROUNDED
            ws.N_star[i, j] = z; ws.N_inf[i, j] = z; ws.N_wet[i, j] = z
            ws.regime[i, j] = AGUAMENTI_NOT_GROUNDED
            ws.q_on[i, j] = z; ws.q_sn[i, j] = z; ws.Lambda0[i, j] = z
            ws.q_h[i, j] = z; ws.q_ch[i, j] = z; ws.q_s[i, j] = z; ws.q_can[i, j] = z
            continue
        end
        q     = ws.q_in[i, j]
        kappa = ws.kappa[i, j]
        Psi   = ws.Psi[i, j]
        u_b   = r.abs_v_b[i, j]
        A     = r.A_visc[i, j]
        D_s   = ws.D_s[i, j]
        p_i   = ws.p_i[i, j]

        q_on = kappa < 1 ? onset_flux(p, Psi, u_b) : oftype(q, Inf)
        q_sn, _ = saddle_node_flux(p, kappa, Psi, u_b, A, D_s)
        Lambda0 = kappa < 1 ? channel_growth_rate(p, q, Psi, u_b) : z
        chi = ws.chi[i, j]
        if steady_chi && kappa < 1
            # from the indicator held before the solve, not the previous sweep's (path independence)
            chi = steady_channel_indicator(ws.chi_start[i, j], q, q_on, q_sn)
            ws.chi[i, j] = chi
        end

        N_star, channelized = flux_balance_root(p, q, chi, kappa, Psi, u_b, A, D_s, p_i, q_sn; developing = q >= q_on)
        N_inf, regime = regime_pressure(p, N_star, p_i, strength_bound(p, kappa, ws.c_0[i, j]))
        N = grounding_line_correction(p, N_inf, state.h[i, j], state.b[i, j])

        ws.N_star[i, j] = N_star; ws.N_inf[i, j] = N_inf; ws.N_wet[i, j] = N
        ws.regime[i, j] = regime
        ws.q_on[i, j] = q_on; ws.q_sn[i, j] = q_sn; ws.Lambda0[i, j] = Lambda0
        Ne = max(N, p.N_lo)
        ws.q_h[i, j]   = kappa < 1 ? sheet_flux(p, Ne, Psi, u_b, A) : z
        ws.q_ch[i, j]  = (kappa < 1 && channelized) ? chi * channel_discharge(p, Ne, Psi, A) / p.l_h : z
        ws.q_s[i, j]   = kappa > 0 ? film_flux(p, Ne, Psi, u_b, A) : z
        ws.q_can[i, j] = kappa > 0 ? canal_discharge(p, Ne, Psi, A, D_s) / p.l_s : z
    end
    return nothing
end


"""
$(TYPEDSIGNATURES)

`state.N` (the sliding pressure `N_slide` of Eq. 25b) from the flux-balance pressure `model.N_wet`:
`N` in wet cells, the dry value `min(N_t, p_i)` (`p_i` on a hard bed, `kappa = 0`) in dry or cold
cells, 0 off the grounded mask. With `equilibrate_till` (steady state) the till is first put in
equilibrium with the interface, `N_t = N_wet` clipped to the bounds of Eq. (19). Also sets `state.W`.
"""
function update_sliding_pressure!(model::AguamentiHydroModel, grid::AbstractHydroGrid, state::HydroState, equilibrate_till::Bool)
    p, ws, r = model.params, model.workspace, model.routing
    z = zero(eltype(state.N))
    @inbounds for j in 1:grid.Ny, i in 1:grid.Nx
        if state.mask[i, j] != GROUNDED
            state.N[i, j] = z; state.W[i, j] = z
            continue
        end
        p_i = ws.p_i[i, j]
        if equilibrate_till
            N_min, N_max = till_bounds(p, p_i)
            ws.N_t[i, j] = clamp(ws.N_wet[i, j], N_min, N_max)
        end
        kappa = ws.kappa[i, j]
        if ws.dry[i, j] != 0
            N = kappa > 0 ? min(ws.N_t[i, j], p_i) : p_i
            state.N[i, j] = N
            state.W[i, j] = z
            ws.regime[i, j] = ws.cold[i, j] != 0 ? AGUAMENTI_COLD : AGUAMENTI_DRY
        else
            N = ws.N_wet[i, j]
            state.N[i, j] = N
            Ne = max(N, p.N_lo)
            u_b, A = r.abs_v_b[i, j], r.A_visc[i, j]
            hs = kappa < 1 ? cavity_height(Ne, u_b, A, p.h_r, p.l_r, p.c_d, p.n) : z
            hf = kappa > 0 ? cavity_height(Ne, u_b, A, p.h_rs, p.l_rs, p.c_d, p.n) : z
            state.W[i, j] = (1 - kappa) * hs + kappa * hf
        end
    end
    return nothing
end


# Undrained till stress, storage and the most water the till can release this step, for every cell (cold
# and off-mask cells, and hard beds, exchange nothing). The uptake itself is solved in the replay.
function update_till_storage!(model::AguamentiHydroModel, grid::AbstractHydroGrid, state::HydroState, mode::StepMode)
    p, ws = model.params, model.workspace
    z = zero(eltype(ws.demand))
    @inbounds for j in 1:grid.Ny, i in 1:grid.Nx
        if state.mask[i, j] != GROUNDED || ws.cold[i, j] != 0 || ws.kappa[i, j] == 0
            ws.demand[i, j] = z
            ws.release_max[i, j] = z
            ws.N_u[i, j] = ws.N_t[i, j]
            ws.storage[i, j] = one(z)
            continue
        end
        dH = mode.dH_dt === nothing ? z : mode.dH_dt[i, j]
        _, demand, N_u, storage = till_relax(p, ws.N_t[i, j], ws.N_wet[i, j], mode.dt, ws.h_t[i, j], ws.k_t[i, j], ws.p_i[i, j], dH)
        ws.demand[i, j] = demand
        ws.release_max[i, j] = storage * (till_void_ratio(p, N_u) - till_void_ratio(p, till_bounds(p, ws.p_i[i, j])[2])) / mode.dt
        ws.N_u[i, j] = N_u
        ws.storage[i, j] = storage
    end
    return nothing
end

update_till_storage!(model::AguamentiHydroModel, grid::AbstractHydroGrid, state::HydroState, ::SteadyMode) =
    (model.demand .= 0; model.release_max .= 0; nothing)


"""
$(TYPEDSIGNATURES)

Solve the routing and the flux balance together (Eqs. 26a-26c, and the till exchange of 26e for a
step): a Picard iteration on `(q, N)`, since the supply (frictional heat through `N`, dissipation
through `q`), the till demand (through `N`) and, with [`HydraulicRouting`](@ref), the flow direction
depend on the solution. Stops when both `q` and `state.N` change by less than `model.rtol` (relative to
their maxima), or after `model.max_iters` sweeps.
"""
function solve_aguamenti!(model::AguamentiHydroModel, grid::AbstractHydroGrid, state::HydroState, mode)
    r = model.routing
    p = model.params

    # Supply: geometric potential, overburden and the gradient the elements use (A2).
    update_phi0!(r, grid, state)
    @. model.p_i = p.rho_i * p.g * state.h
    hydraulic = model.routing_potential isa HydraulicRouting
    prepare_aguamenti_routing!(model, grid, state, false)
    Psi_min = p.Psi_min
    @. model.Psi = max(r.abs_grad_phi0, Psi_min)

    mode isa SteadyMode && (model.chi_start .= model.chi)
    start_time = time()
    converged = false
    n_iters = model.max_iters
    info = model.info
    empty!(info.history)
    for iter in 1:model.max_iters
        model.q_prev .= r.q
        model.N_prev .= state.N

        if hydraulic
            w = model.routing_relaxation
            @. model.N_route += w * (model.N_wet - model.N_route)
            prepare_aguamenti_routing!(model, grid, state, true)
        end

        # Supply (Eq. 7): fixed melt, water from above, frictional heat at the current N, dissipation at the current q.
        update_tau_b!(r, state, r.sliding_law)
        @. r.mdot_total = r.mdot_fixed + r.i_eb
        add_friction_term!(r)
        add_dissipation_term!(r, r.dissipation_melt)
        r.Q_sens .= 0

        # Routing with the till exchange bounded by the water available (Eqs. 8-9).
        update_till_storage!(model, grid, state, mode)
        tape = r.routing_tape
        tape.valid || record_routing_tape!(tape, r, grid, state)
        uptake = mode isa StepMode ? TillUptake(model, state, mode.dt, mode.dH_dt, grid.dx * grid.dy) : NoUptake()
        replay_aguamenti_tape!(uptake, r.psi_out, model.mdot_t, model.dry, model.refreeze, model.freeze_deficit, r.mdot_total,
                               model.release_max, model.cold, state.mask, tape, grid.Nx, grid.Ny, grid.dx, grid.dy, p.rho_w)
        update_q_from_psi_out!(r, grid, state)
        # A dry or cold cell has no interfacial water (the face-averaged q would still see its inflow). The
        # flux balance keeps the routed flux, so the till of a dry cell keeps relaxing towards the pressure
        # of the water it absorbs rather than towards the q = 0 pressure (which would flip it wet again).
        model.q_in .= r.q
        z = zero(eltype(r.q))
        @. r.q = ifelse(model.dry != 0, z, r.q)

        # Flux balance, regimes and grounding-line correction; then the sliding pressure.
        update_flux_balance!(model, grid, state, mode isa SteadyMode)
        update_sliding_pressure!(model, grid, state, mode isa SteadyMode)

        q_scale = max(masked_max_abs(grid, r.q, state.mask), eps(eltype(r.q)))
        N_scale = max(masked_max_abs(grid, state.N, state.mask), eps(eltype(state.N)))
        res_N = masked_max_abs_diff(grid, state.N, model.N_prev, state.mask) / N_scale
        res_q = masked_max_abs_diff(grid, r.q, model.q_prev, state.mask) / q_scale
        push!(info.history, (res_q, res_N))
        if iter > 1 && res_q <= model.rtol && res_N <= model.rtol
            converged = true
            n_iters = iter
            break
        end
    end

    info.iters = n_iters
    info.converged = converged
    info.res_q, info.res_N = isempty(info.history) ? (NaN, NaN) : info.history[end]
    info.elapsed = time() - start_time
    if model.verbose
        status = converged ? "converged" : "did NOT converge (hit max_iters)"
        @info "Aguamenti (q, N) Picard loop: $status after $n_iters iteration(s) in $(round(time() - start_time, digits = 4)) s"
    end
    return converged
end


"""
$(TYPEDSIGNATURES)

Steady state of the Aguamenti model for the current geometry and forcing: the till in equilibrium with
the interface (`N_t = N`, no exchange) and the channel indicator in the steady state of Eq. (22),
keeping its current value in the bistable band `q_sn <= q < q_on` (so the result depends on the
history held in `model.chi`, the hysteresis of Sect. 2.6).
"""
function update_steady_state!(model::AguamentiHydroModel, grid::AbstractHydroGrid, state::HydroState)
    solve_aguamenti!(model, grid, state, SteadyMode()) ||
        @warn "Aguamenti steady solve did not converge in max_iters = $(model.max_iters) sweeps" maxlog = 10
    return nothing
end


"""
$(TYPEDSIGNATURES)

Advance an [`AguamentiHydroModel`](@ref) by `dt` [s] (Eqs. 26a-26f): solve the routing and flux balance
with the till exchange of this step, then advance the till effective stress `model.N_t` (Eq. 26e, with
the exchange bounded by the water available) and the channel indicator `model.chi` (Eq. 26d) over the
step. `dH_dt` (field [m/s], or `nothing` for zero) is the thinning/thickening rate of the ice for the
undrained response of the till. On return `state.N` is the sliding pressure of this step.
"""
function step!(model::AguamentiHydroModel, grid::AbstractHydroGrid, state::HydroState, dt::Real; dH_dt = nothing)
    dt > 0 || throw(ArgumentError("dt must be positive (got $dt)"))
    dH_dt === nothing || size(dH_dt)[1:2] == (grid.Nx, grid.Ny) ||
        throw(ArgumentError("dH_dt size $(size(dH_dt)) != grid size $((grid.Nx, grid.Ny))"))
    T = eltype(state.N)
    mode = StepMode(T(dt), dH_dt)
    solve_aguamenti!(model, grid, state, mode) ||
        @warn "Aguamenti step did not converge in max_iters = $(model.max_iters) sweeps" maxlog = 10

    p, ws, r = model.params, model.workspace, model.routing
    @inbounds for j in 1:grid.Ny, i in 1:grid.Nx
        state.mask[i, j] == GROUNDED || continue
        # Till: stored water changes by the bounded exchange (Eqs. 9, 20).
        if ws.cold[i, j] == 0 && ws.kappa[i, j] > 0
            ws.N_t[i, j] = till_from_exchange(p, ws.N_u[i, j], ws.mdot_t[i, j], mode.dt, ws.storage[i, j], ws.p_i[i, j])
            # a dry cell slides on the till at its end-of-step stress (Eq. 26f)
            ws.dry[i, j] != 0 && (state.N[i, j] = min(ws.N_t[i, j], ws.p_i[i, j]))
        end
        # Channels (Eq. 22), at the flux and pressure of this step.
        if ws.kappa[i, j] < 1
            ws.chi[i, j] = advance_channel_indicator(p, ws.chi[i, j], r.q[i, j], ws.q_on[i, j], ws.q_sn[i, j],
                                                     ws.Lambda0[i, j], state.N[i, j], r.A_visc[i, j], mode.dt)
        end
    end
    return nothing
end


"""
$(TYPEDSIGNATURES)

`state.N` for a new sliding speed `abs_v_b` [m/s], with the routing held (`q`, `Psi` and the till
exchange of the last solve): the flux balance alone, re-evaluated with the new cavity opening. See
[`N_responds_to_ub`](@ref).
"""
function N_from_ub!(model::AguamentiHydroModel, grid::AbstractHydroGrid, state::HydroState, abs_v_b)
    model.routing.abs_v_b .= abs_v_b
    update_flux_balance!(model, grid, state, false)
    update_sliding_pressure!(model, grid, state, false)
    return state.N
end
