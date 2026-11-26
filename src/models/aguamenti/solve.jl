###########################################
# Model: Aguamenti v1.0 (Gregov et al.)   #
###########################################

# The reduced problem (Eqs. 25a-25f) advanced as in Sect. 4: routing, flux balance and regimes are
# algebraic in time and solved together by a Picard iteration on (q, N); within a step the drainage
# state chi is advanced from its value at the start of the step at the routed flux of every sweep
# (Eq. 21), before the flux balance uses it, and the till reservoir N_t is advanced after the solve
# with the exchange applied in the routing (Eq. 28a).


# How a solve treats the two state fields.
struct SteadyMode end                       # till in equilibrium with N, chi in the steady state of Eq. (21)
struct StepMode{T, D}                       # advance N_t and chi over dt
    dt    ::T
    dH_dt ::D                               # rate of change of the ice thickness [m/s] (field), or nothing
end


"""
$(TYPEDSIGNATURES)

Build the routing graph of the engine for the potential `phi0 - corr` on the grounded cells (`phi0`
where `corr === nothing`): the geometric potential (`GeometricRouting`), `phi0 - N_route`
(`HydraulicRouting`) or `phi0 - C` (`WeightedRouting`). The engine's `abs_grad_phi0` is reset to the
gradient of the true `phi0` afterwards (the elements use `Psi`, A2), and the correction factor from
`psi_out` to `q` and the routing tape are refreshed. Counts the build in `model.info.routing_builds`.
"""
function prepare_aguamenti_routing!(model::AguamentiHydroModel, grid::AbstractHydroGrid, state::HydroState, corr)
    r = model.routing
    if corr !== nothing
        model.phi0_true .= r.phi0
        @. r.phi0 = ifelse(state.mask == GROUNDED, r.phi0 - corr, r.phi0)
    end
    prepare_routing!(r, grid, state, r.routing_scheme)
    if corr !== nothing
        r.phi0 .= model.phi0_true
        update_true_gradient_magnitude!(r, grid, r.gradient_discretization)
    end
    epsT = eps(eltype(r.corfac))
    dx, dy = grid.dx, grid.dy
    @. r.corfac = (abs(r.minus_grad_phi0_sx) * dy + abs(r.minus_grad_phi0_sy) * dx) /
                  (sqrt(r.minus_grad_phi0_sx * r.minus_grad_phi0_sx + r.minus_grad_phi0_sy * r.minus_grad_phi0_sy) + epsT)
    invalidate_routing_tape!(r)
    model.info.routing_builds += 1
    return nothing
end

# The gradient the elements use (A2), floored at Psi_min (Table C2), from the engine's abs_grad_phi0.
set_element_gradient!(model::AguamentiHydroModel) = (@. model.Psi = max(model.routing.abs_grad_phi0, model.params.Psi_min); nothing)


# Routing set-up of a solve, before the Picard loop: the graph when it is fixed over the solve, and Psi.
function setup_routing!(model::AguamentiHydroModel, ::AbstractRoutingPotential, grid::AbstractHydroGrid, state::HydroState, mode)
    prepare_aguamenti_routing!(model, grid, state, nothing)
    set_element_gradient!(model)
    return nothing
end

function setup_routing!(model::AguamentiHydroModel, rp::WeightedRouting, grid::AbstractHydroGrid, state::HydroState, mode)
    r = model.routing
    update_true_gradient_magnitude!(r, grid, r.gradient_discretization)   # Psi before the graph: the clip needs it
    set_element_gradient!(model)
    if mode isa StepMode
        # Eq. (26) at the state the step starts from: chi^(j) and N^(j-1); one graph for the whole step.
        weighted_routing_correction!(model, grid, state, rp.gamma0, model.chi_start, model.N_lag)
        prepare_aguamenti_routing!(model, grid, state, model.route_corr)
    end
    return nothing
end

# Routing update of every sweep (a graph that depends on the solution).
sweep_routing!(model::AguamentiHydroModel, ::AbstractRoutingPotential, grid::AbstractHydroGrid, state::HydroState, mode, iter) = nothing

function sweep_routing!(model::AguamentiHydroModel, ::HydraulicRouting, grid::AbstractHydroGrid, state::HydroState, mode, iter)
    w = model.routing_relaxation
    @. model.N_route += w * (model.N_wet - model.N_route)
    prepare_aguamenti_routing!(model, grid, state, model.N_route)
    return nothing
end

function sweep_routing!(model::AguamentiHydroModel, rp::WeightedRouting, grid::AbstractHydroGrid, state::HydroState, ::SteadyMode, iter)
    # Steady state: N and chi of Eq. (26) are those of the solve itself, N under-relaxed (from the
    # state.N the solve starts from) as for HydraulicRouting. The flow directions jump with N, so where
    # the feedback is strong the graph can cycle between two states however small the relaxation; if it
    # has not settled after half the sweeps, it is held for the rest of the solve so that (q, N)
    # converge on it (`model.info.res_route` then says how far N_route is from the final N).
    if iter == 1
        model.N_route .= state.N
    elseif iter <= max(model.max_iters ÷ 2, 1)
        w = model.routing_relaxation
        @. model.N_route += w * (state.N - model.N_route)
    else
        return nothing
    end
    weighted_routing_correction!(model, grid, state, rp.gamma0, model.chi, model.N_route)
    prepare_aguamenti_routing!(model, grid, state, model.route_corr)
    return nothing
end

# Relative distance between the N the routing graph was built with and the final N (WeightedRouting in
# a steady solve; NaN otherwise, where the graph is fixed over the solve or follows N_wet every sweep).
routing_residual(model::AguamentiHydroModel, ::AbstractRoutingPotential, grid, state, mode, N_scale) = oftype(N_scale, NaN)
routing_residual(model::AguamentiHydroModel, ::WeightedRouting, grid, state, ::SteadyMode, N_scale) =
    masked_max_abs_diff(grid, model.N_route, state.N, state.mask) / N_scale

# Whether that graph has settled: the next rebuild would move N_route by less than `rtol` (relative).
routing_settled(model::AguamentiHydroModel, rp::AbstractRoutingPotential, grid, state, mode, N_scale) =
    !(model.routing_relaxation * routing_residual(model, rp, grid, state, mode, N_scale) > model.rtol)


"""
$(TYPEDSIGNATURES)

The correction `C` of the weighted routing potential `phi0 - C`, Eq. (26), into `model.route_corr`:
`C = gamma0 (1 - chi) N` on the grounded cells (0 elsewhere), then clipped (Eq. 26c) to the largest
field below it whose difference across every face between two grounded cells is at most
`min(Psi_i, Psi_k) d / sqrt(2)` (`d = dx` or `dy`): the lower envelope
`C_i = min_k (C_k + dist(i, k))`, computed by Gauss-Seidel sweeps in the four raster orders until
nothing changes (a shortest-path relaxation, allocation-free). The result is a cell-value field, and
the `1/sqrt(2)` per face is what keeps the norm of its cell gradient vector, `|grad C| <= Psi` (the
literal Eq. 26c), under both of K24's gradient stencils (each centred component, or each face gradient,
is at most `Psi/sqrt(2)`); along a grid axis it caps the gradient at `Psi/sqrt(2)`, conservatively.
The envelope only lowers `C`, so it stays 0 where `gamma = 0` (channelized cells). `N` is `N^(j-1)`
(`N_slide` of the previous step, so `min(N_t, p_i)` or `p_i` on dry or cold cells) in a step, the
under-relaxed `N` of the solve in a steady solve; `Psi` is the elements' gradient floored at `Psi_min`.
Sets `model.info.G_gamma` (the largest ratio of an unclipped face difference to its bound:
`sqrt(2) max(|dC/dx|, |dC/dy|) / Psi`, between Table A2's `gamma |grad N| / Psi` and `sqrt(2)` times it)
and `model.info.clipped` (the number of cells lowered).
"""
function weighted_routing_correction!(model::AguamentiHydroModel, grid::AbstractHydroGrid, state::HydroState, gamma0, chi, N)
    C, Psi, mask = model.route_corr, model.Psi, state.mask
    T = eltype(C)
    Nx, Ny = grid.Nx, grid.Ny
    z = zero(T)
    g0 = T(gamma0)
    s = inv(sqrt(T(2)))
    wx, wy = s * grid.dx, s * grid.dy
    @inbounds for j in 1:Ny, i in 1:Nx
        C[i, j] = mask[i, j] == GROUNDED ? g0 * (1 - chi[i, j]) * max(N[i, j], z) : z
    end
    # G_gamma of the unclipped correction
    G = z
    @inbounds for j in 1:Ny, i in 1:Nx
        mask[i, j] == GROUNDED || continue
        if i < Nx && mask[i + 1, j] == GROUNDED
            G = max(G, abs(C[i + 1, j] - C[i, j]) / (min(Psi[i, j], Psi[i + 1, j]) * wx))
        end
        if j < Ny && mask[i, j + 1] == GROUNDED
            G = max(G, abs(C[i, j + 1] - C[i, j]) / (min(Psi[i, j], Psi[i, j + 1]) * wy))
        end
    end
    model.info.G_gamma = G
    if G > 1
        # Lower envelope: every pass is at least one Bellman-Ford round, so Nx Ny passes always suffice.
        for _ in 1:(Nx * Ny)
            changed = envelope_sweep!(C, Psi, mask, wx, wy, 1:Nx, 1:Ny)
            changed |= envelope_sweep!(C, Psi, mask, wx, wy, Nx:-1:1, 1:Ny)
            changed |= envelope_sweep!(C, Psi, mask, wx, wy, 1:Nx, Ny:-1:1)
            changed |= envelope_sweep!(C, Psi, mask, wx, wy, Nx:-1:1, Ny:-1:1)
            changed || break
        end
    end
    n = 0
    @inbounds for j in 1:Ny, i in 1:Nx
        mask[i, j] == GROUNDED || continue
        n += C[i, j] < g0 * (1 - chi[i, j]) * max(N[i, j], z)
    end
    model.info.clipped = n
    return nothing
end

# One Gauss-Seidel sweep of the lower envelope in the order (irange, jrange): each grounded cell takes
# the smallest of its own value and those of its grounded neighbours plus the bound of the face.
function envelope_sweep!(C, Psi, mask, wx, wy, irange, jrange)
    Nx, Ny = size(C, 1), size(C, 2)
    changed = false
    @inbounds for j in jrange, i in irange
        mask[i, j] == GROUNDED || continue
        c, P = C[i, j], Psi[i, j]
        if i > 1 && mask[i - 1, j] == GROUNDED
            c = min(c, C[i - 1, j] + min(P, Psi[i - 1, j]) * wx)
        end
        if i < Nx && mask[i + 1, j] == GROUNDED
            c = min(c, C[i + 1, j] + min(P, Psi[i + 1, j]) * wx)
        end
        if j > 1 && mask[i, j - 1] == GROUNDED
            c = min(c, C[i, j - 1] + min(P, Psi[i, j - 1]) * wy)
        end
        if j < Ny && mask[i, j + 1] == GROUNDED
            c = min(c, C[i, j + 1] + min(P, Psi[i, j + 1]) * wy)
        end
        if c < C[i, j]
            C[i, j] = c
            changed = true
        end
    end
    return changed
end


"""
$(TYPEDSIGNATURES)

Replay the engine's routing tape (see [`TapedPsiOut`](@ref)) with Aguamenti's sources and sinks. Each
grounded cell starts from its own supply `mdot_b/rho_w dx dy` and receives its upstream inflow, in the
tape's order (every cell after all the cells that drain into it); when the cell is complete, with
`W` [m3/s] the water available (Eq. 9a, `(mdot_b/rho_w + mdot_in) dx dy`):

- a cold cell refreezes everything it receives (`refreeze`) and passes nothing on;
- otherwise the till takes `uptake(i, j, W)` [m/s] (see [`till_uptake`](@ref)), which never exceeds
  `W/(dx dy)`; a freeze-on deficit (`W < 0`) is taken from the till down to `-release_max` (Eqs. 9b,
  27), and what remains is `freeze_deficit`. A cell left without water is dry.

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
                freeze_deficit[ci, cj] = max(u - W / area, z)    # 0 when a lagged release exceeds it
            end
            mdot_t[ci, cj] = u
            # the till takes all the water (or all of a freeze-on deficit comes from the till): exactly
            # dry. From W - u area the round-off (~1e-16 m3/s) would decide, and a cell at that edge
            # flipped between dry (N_slide) and wet at q ~ 0 (N near p_i) from sweep to sweep.
            out = u == W / area ? z : W - u * area
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

Till uptake of one cell over a step with [`ImplicitTillExchange`](@ref), solved together with the
cell's own flux balance: the till relaxes towards the `N` of the water that remains in the cell after
its uptake, so the uptake `u` [m/s] solves `u = demand(N(q(W - u dx dy)))`, where `q(out)` is the
routing's psi-to-q conversion of the cell at the outflow `out` (see [`uptake_flux_coefficients`](@ref)), `N(q)` is the flux balance with its regimes
and grounding-line correction (Eqs. 20-23) at the channel indicator `chi` the option of
[`AbstractUptakeChi`](@ref) selects (by default `chi^(j)`, the step's start), and `demand(N)` the
exchange of the till relaxation over the step ([`till_relax`](@ref), Eq. 28a without the undrained
term). The channel indicator is held within the root: advanced at `q(W - u dx dy)` it would let a
release open channels, raise `N` and release more, a second, spurious root. The right-hand side
decreases with `u` (less water, higher `N`, a smaller demand), so the root is unique and found by
Brent's method on `[-release_max, W/(dx dy)]`, to ~1e-12 of that bracket within `root_iters`
evaluations; at `u = W/(dx dy)` the till takes all the water and the cell is dry (Eq. 9a). Where `N(q)`
jumps (a switch of the active branch) and there is no root, the uptake is the largest `u` with
`u <= demand(u)`. Solving the cell in the routing order makes the uptake consistent with the inflow
from upstream within one replay -- a till demand lagged by a sweep (or by a step, Eq. 28b) instead
makes whole flow lines flip between wet and dry.
"""
struct TillUptake{M, ST, TT, D, C, Q}
    model ::M
    state ::ST
    chi   ::C     # channel indicator held within the root: model.chi_start or model.chi
    dt    ::TT
    dH_dt ::D
    area  ::TT
    dx    ::TT    # cell size along x [m] (dy = area / dx), for the face-normal fluxes
    q_conversion ::Q   # the routing's psi -> q conversion, which the uptake's q(W - u dx dy) follows
end

"""
$(TYPEDSIGNATURES)

The coefficients `(a_x, a_y, b_x, b_y)` of the interfacial flux of cell `(i, j)` as a function of its
outflow `out` [m3/s] during the routing replay, `q = |(a_x + b_x out, a_y + b_y out)|` clamped to
`[q_min, q_max]`: the same `q` the model then computes from the routed flux with its `q_conversion`, so
the till of [`TillUptake`](@ref) relaxes towards the `N` the cell gets.
- [`QFromOutflow`](@ref): `q = out / corfac` (`a = 0`).
- [`QFromFaceAverage`](@ref): the face-normal fluxes averaged to the centre. `a` holds the inflow of
  the cell's upstream neighbours across its faces, final when the replay reaches the cell (they come
  before it in the routing order), and `b` the cell's own routing weights. With `out / corfac` here
  instead (the inflow faces left out), the till relaxed towards an `N` the cell never has, and a
  steady state was not a fixed point of a step (AIS: 174 wet cells whose till drifted by >10%).
"""
@inline function uptake_flux_coefficients(::QFromOutflow, tu::TillUptake, i, j)
    r = tu.model.routing
    z = zero(tu.area)
    return z, z, @inbounds(one(z) / (r.corfac[i, j] + eps(z))), z
end

@inline function uptake_flux_coefficients(::QFromFaceAverage, tu::TillUptake, i, j)
    r, mask = tu.model.routing, tu.state.mask
    psi, w = r.psi_out, r.routing_tape.w8
    Nx, Ny = size(psi)
    z = zero(tu.area)
    dx = tu.dx
    dy = tu.area / dx
    ax = ay = z
    @inbounds begin
        (i > 1  && mask[i - 1, j] == GROUNDED) && (ax += psi[i - 1, j] * w[2, i - 1, j])
        (i < Nx && mask[i + 1, j] == GROUNDED) && (ax -= psi[i + 1, j] * w[1, i + 1, j])
        (j > 1  && mask[i, j - 1] == GROUNDED) && (ay += psi[i, j - 1] * w[4, i, j - 1])
        (j < Ny && mask[i, j + 1] == GROUNDED) && (ay -= psi[i, j + 1] * w[3, i, j + 1])
        bx = w[2, i, j] - w[1, i, j]
        by = w[4, i, j] - w[3, i, j]
    end
    return ax / (2dy), ay / (2dx), bx / (2dy), by / (2dx)
end

function (tu::TillUptake)(i, j, W)
    m, state = tu.model, tu.state
    p, ws, r = m.params, m.workspace, m.routing
    z = zero(W)
    @inbounds begin
        (ws.kappa[i, j] == 0 || W <= 0) && return z
        p_i, kappa, Psi = ws.p_i[i, j], ws.kappa[i, j], ws.Psi[i, j]
        u_b, A, D_s = r.abs_v_b[i, j], r.A_visc[i, j], ws.D_s[i, j]
        chi = tu.chi[i, j]
        N_t, h_t, k_t, h, b = ws.N_t[i, j], ws.h_t[i, j], ws.k_t[i, j], state.h[i, j], state.b[i, j]
        dH = tu.dH_dt === nothing ? z : tu.dH_dt[i, j]
    end
    N0 = strength_bound(p, kappa, ws.c_0[i, j])
    q_on, q_sn, N_sn = ws.q_on[i, j], ws.q_sn[i, j], ws.N_sn[i, j]      # per solve, update_channel_thresholds!
    dt = tu.dt
    ax, ay, bx, by = uptake_flux_coefficients(tu.q_conversion, tu, i, j)
    q_min, q_max = r.q_min, r.q_max
    function N_of(u)
        out = max(W - u * tu.area, z)
        q = min(max(hypot(ax + bx * out, ay + by * out), q_min), q_max)
        N_star, _ = flux_balance_root(p, q, chi, kappa, Psi, u_b, A, D_s, p_i, q_sn; developing = q >= q_on, N_sn)
        N_inf, _ = regime_pressure(p, N_star, p_i, N0)
        return grounding_line_correction(p, N_inf, h, b)
    end
    coeffs = till_relax_coefficients(p, N_t, dt, h_t, k_t, p_i, dH)
    demand(u) = till_relax_to(p, coeffs, N_of(u))[2]
    lo, hi = -ws.release_max[i, j], W / tu.area
    g_hi = demand(hi) - hi
    g_hi >= 0 && return hi                          # the till absorbs all the water: dry (Eq. 9a)
    g_lo = demand(lo) - lo
    g_lo <= 0 && return lo
    # demand(u) - u decreases strictly in u: Brent's method, to ~1e-12 of the bracket. Where N(q) jumps
    # (a switch of the active branch) there may be no root but a jump of demand(u) - u across zero: then
    # the largest uptake the demand still covers, u <= demand(u), the same whichever way it is approached.
    return brent_root(u -> demand(u) - u, lo, hi, g_lo, g_hi, (hi - lo) * eps(one(W))^(3 // 4), p.root_iters;
                      nonnegative = true)
end

"""
$(TYPEDSIGNATURES)

Till uptake of one cell over a step with [`LaggedTillExchange`](@ref), Eq. (27): the exchange of
Eq. (28b) at the `N` of the previous step (`model.demand`, from [`update_till_storage!`](@ref)),
bounded by the water available in the cell, `W/(dx dy)`, when it is positive, and by the water the till
holds above its residual content, `-release_max`, when it is negative.
"""
struct LaggedUptake{WS, TT}
    workspace ::WS
    area      ::TT
end

@inline function (lu::LaggedUptake)(i, j, W)
    ws = lu.workspace
    @inbounds m = ws.demand[i, j]
    return m > 0 ? min(m, W / lu.area) : @inbounds(max(m, -ws.release_max[i, j]))
end

"""
$(TYPEDSIGNATURES)

The till uptake functor `uptake(i, j, W)` the routing replay calls in every cell (see
[`replay_aguamenti_tape!`](@ref)): [`TillUptake`](@ref) for [`ImplicitTillExchange`](@ref),
[`LaggedUptake`](@ref) for [`LaggedTillExchange`](@ref), and none in a steady solve.
"""
till_uptake(te::ImplicitTillExchange, model::AguamentiHydroModel, state::HydroState, mode::StepMode, area, dx = sqrt(area)) =
    TillUptake(model, state, uptake_chi(te.chi, model.workspace), mode.dt, mode.dH_dt, area, oftype(area, dx), model.routing.q_conversion)

"The channel indicator array the uptake of [`ImplicitTillExchange`](@ref) holds, see [`AbstractUptakeChi`](@ref)."
uptake_chi(::StepStartChi, ws) = ws.chi_start
uptake_chi(::LatestChi, ws) = ws.chi
till_uptake(::LaggedTillExchange, model::AguamentiHydroModel, state::HydroState, ::StepMode, area, dx = sqrt(area)) =
    LaggedUptake(model.workspace, area)
till_uptake(::AbstractTillExchange, model::AguamentiHydroModel, state::HydroState, ::SteadyMode, area, dx = sqrt(area)) = NoUptake()


"""
$(TYPEDSIGNATURES)

Onset flux `q_on`, saddle-node flux `q_sn` and its pressure `N_sn` of every grounded cell (Sect. 3.6);
`q_on = Inf`, `q_sn = N_sn = 0` where no channel can form (`kappa = 1`, or `channels = false`).
They depend on the gradient, sliding speed, rate factor and bed only, not on `q` or `N`, so they are
computed once per solve (and again in [`N_from_ub!`](@ref), for the new sliding speed) rather than
in every sweep and every till-uptake evaluation.
"""
function update_channel_thresholds!(model::AguamentiHydroModel, grid::AbstractHydroGrid, state::HydroState)
    aguamenti_columns!(channel_thresholds_column!, grid.Ny, model.workspace, model.params, model.routing, state, grid.Nx)
    return nothing
end

function channel_thresholds_column!(j, ws, p::AguamentiParams, r, state, Nx)
    z = zero(eltype(ws.q_sn))
    @inbounds for i in 1:Nx
        kappa = ws.kappa[i, j]
        if state.mask[i, j] != GROUNDED || kappa >= 1 || !p.channels
            # no channel can form (soft bed, or channels = false): no onset, no channelized branch
            ws.q_on[i, j] = state.mask[i, j] == GROUNDED ? oftype(z, Inf) : z
            ws.q_sn[i, j] = z; ws.N_sn[i, j] = z
            continue
        end
        Psi, u_b = ws.Psi[i, j], r.abs_v_b[i, j]
        ws.q_on[i, j] = onset_flux(p, Psi, u_b)
        q_sn, N_sn = saddle_node_flux(p, kappa, Psi, u_b, r.A_visc[i, j], ws.D_s[i, j])
        ws.q_sn[i, j] = q_sn; ws.N_sn[i, j] = N_sn
    end
    return nothing
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

Flux balance of every grounded cell for the current `q` (Sect. 3.6): the growth rate, the root `N*` on
the active branch (Eq. 20), the regime and `N_inf` (Eq. 22), the grounding-line correction (Eq. 23)
into `model.N_wet`, and the element fluxes at that pressure. The channel indicator it uses depends on
`chi_mode`:

- `nothing`: the current `model.chi`, held;
- `SteadyMode()`: first set to the steady state of Eq. (21) for this `q`, from `model.chi_start`;
- a `StepMode`: first advanced over the step from `model.chi_start` (Eq. 21, integrated exactly with
  [`advance_channel_indicator`](@ref)) at the routed flux of this sweep (0 in dry cells), with the
  creep closure at `N^(j-1) = model.N_lag` -- step 3 of Sect. 4, before the flux balance (step 4).
"""
function update_flux_balance!(model::AguamentiHydroModel, grid::AbstractHydroGrid, state::HydroState, chi_mode)
    aguamenti_columns!(flux_balance_column!, grid.Ny, model, model.params, model.routing, state, grid.Nx, chi_mode)
    return nothing
end

# The channel indicator of a cell for the flux balance (see update_flux_balance!).
@inline balance_chi(::Nothing, p, ws, r, i, j, q, q_on, q_sn, Psi, u_b, A) = @inbounds ws.chi[i, j]

@inline function balance_chi(::SteadyMode, p, ws, r, i, j, q, q_on, q_sn, Psi, u_b, A)
    # from the indicator held before the solve, not the previous sweep's (path independence)
    @inbounds chi = ws.chi[i, j] = steady_channel_indicator(ws.chi_start[i, j], q, q_on, q_sn)
    return chi
end

@inline function balance_chi(mode::StepMode, p, ws, r, i, j, q, q_on, q_sn, Psi, u_b, A)
    @inbounds begin
        q_r = r.q[i, j]                     # the routed flux, 0 in a dry cell (no interfacial water)
        chi = ws.chi[i, j] = advance_channel_indicator(p, ws.chi_start[i, j], q_r, q_on, q_sn,
                                                       channel_growth_rate(p, q_r, Psi, u_b), ws.N_lag[i, j], A, mode.dt)
    end
    return chi
end

function flux_balance_column!(j, model, p::AguamentiParams, r, state, Nx, chi_mode)
    ws = model.workspace
    z = zero(eltype(ws.N_wet))
    @inbounds for i in 1:Nx
        if state.mask[i, j] != GROUNDED
            ws.N_star[i, j] = z; ws.N_inf[i, j] = z; ws.N_wet[i, j] = z
            ws.regime[i, j] = AGUAMENTI_NOT_GROUNDED
            ws.Lambda0[i, j] = z
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

        q_on, q_sn = ws.q_on[i, j], ws.q_sn[i, j]       # per solve, update_channel_thresholds!
        # Lambda0: a diagnostic, the opening rate at the flux of the balance (q_in). The indicator itself is
        # advanced at the routed flux r.q (balance_chi), which is q_in in a wet cell and 0 in a dry one.
        Lambda0 = (kappa < 1 && p.channels) ? channel_growth_rate(p, q, Psi, u_b) : z
        chi = kappa < 1 ? balance_chi(chi_mode, p, ws, r, i, j, q, q_on, q_sn, Psi, u_b, A) : ws.chi[i, j]

        N_star, channelized = flux_balance_root(p, q, chi, kappa, Psi, u_b, A, D_s, p_i, q_sn; developing = q >= q_on,
                                                N_sn = ws.N_sn[i, j])
        N_inf, regime = regime_pressure(p, N_star, p_i, strength_bound(p, kappa, ws.c_0[i, j]))
        N = grounding_line_correction(p, N_inf, state.h[i, j], state.b[i, j])

        ws.N_star[i, j] = N_star; ws.N_inf[i, j] = N_inf; ws.N_wet[i, j] = N
        ws.regime[i, j] = regime
        ws.Lambda0[i, j] = Lambda0
        Ne = max(N, p.N_lo)
        ws.q_h[i, j]   = kappa < 1 ? sheet_flux(p, Ne, Psi, u_b, A) : z
        # the channel flux: chi Q_h/l_h on the channelized branch, and the fraction chi of a closing channel
        ws.q_ch[i, j]  = (kappa < 1 && channelized) ? chi * channel_discharge(p, Ne, Psi, A) / p.l_h : z
        ws.q_s[i, j]   = kappa > 0 ? film_flux(p, Ne, Psi, u_b, A) : z
        ws.q_can[i, j] = kappa > 0 ? canal_discharge(p, Ne, Psi, A, D_s) / p.l_s : z
    end
    return nothing
end


"""
$(TYPEDSIGNATURES)

`state.N` (the sliding pressure `N_slide` of Eq. 24b) from the flux-balance pressure `model.N_wet`:
`N` in wet cells, the dry value `min(N_t, p_i)` in dry or cold cells, 0 off the grounded mask. A dry
or cold cell of a hard bed (`kappa = 0`) takes `p_i`: a user decision consistent with Eq. (24b), which
does not distinguish the beds -- with no till, `N_t` is undefined and taken as `p_i` (the bound
`N_max <= p_i` of a till at its densest state). This `state.N` is the `N^(j-1)` of the next step (the
creep of Eq. 21, the weighted routing of Eq. 26a, the lagged exchange of Eq. 28b), so water arriving
at a dry cell sees the pore pressure of its till. With `equilibrate_till` (steady state) the till is
first put in equilibrium with the interface, `N_t = N_wet` clipped to the bounds of Eq. (18). Also sets
`state.W`.
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
            # A temperate cell under freeze-on (a deficit the till would supply) drains its till to the
            # residual content, N_max, as a step does (Eqs. 9b, 27); it has no interfacial water to equilibrate with.
            ws.N_t[i, j] = (ws.cold[i, j] == 0 && ws.freeze_deficit[i, j] > 0) ? N_max : clamp(ws.N_wet[i, j], N_min, N_max)
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


"""
$(TYPEDSIGNATURES)

Per-cell till coefficients of a step (Eq. 28a): the undrained increment `dN_u`, the storage
`(1 - theta) h_t` at the stress the step starts from, the most water the till can release before it
reaches its densest state, `release_max = (1 - theta) h_t (e - e_min) / dt` (Eqs. 9b, 27 with the solids
conserved, Appendix D3: v2's `(theta - theta_min) h_t / dt` for a fixed `h_t` is `1/(1 + e_min)` times
this; `e_min` the void ratio at `N_max`), and the exchange the till asks for, `demand` (see
[`AbstractTillExchange`](@ref)).
Cold and off-mask cells, and hard beds, exchange nothing. A steady solve has no exchange.
"""
function update_till_storage!(model::AguamentiHydroModel, grid::AbstractHydroGrid, state::HydroState, mode::StepMode)
    p, ws, te = model.params, model.workspace, model.till_exchange
    z = zero(eltype(ws.demand))
    @inbounds for j in 1:grid.Ny, i in 1:grid.Nx
        if state.mask[i, j] != GROUNDED || ws.cold[i, j] != 0 || ws.kappa[i, j] == 0
            ws.demand[i, j] = z
            ws.release_max[i, j] = z
            ws.dN_u[i, j] = z
            ws.storage[i, j] = one(z)
            continue
        end
        dH = mode.dH_dt === nothing ? z : mode.dH_dt[i, j]
        c = till_relax_coefficients(p, ws.N_t[i, j], mode.dt, ws.h_t[i, j], ws.k_t[i, j], ws.p_i[i, j], dH)
        ws.demand[i, j] = till_demand(te, p, c, ws, i, j)
        ws.release_max[i, j] = c.storage * max(c.e_0 - till_void_ratio(p, c.N_max), z) / mode.dt
        ws.dN_u[i, j] = c.dN_u
        ws.storage[i, j] = c.storage
    end
    return nothing
end

update_till_storage!(model::AguamentiHydroModel, grid::AbstractHydroGrid, state::HydroState, ::SteadyMode) =
    (model.demand .= 0; model.release_max .= 0; model.dN_u .= 0; nothing)

# The exchange the till asks for before the bounds: the relaxation's towards this sweep's N_wet (a
# diagnostic: the implicit uptake is solved in the replay), or Eq. (28b) at N^(j-1).
@inline till_demand(::ImplicitTillExchange, p, c, ws, i, j) = @inbounds till_relax_to(p, c, ws.N_wet[i, j])[2]
@inline till_demand(::LaggedTillExchange, p, c, ws, i, j) =
    @inbounds till_lagged_exchange(p, c, ws.N_lag[i, j])


"""
$(TYPEDSIGNATURES)

Solve the routing and the flux balance together (Eqs. 25a-25c; for a step, with the channel indicator
of Eq. 25d advanced and the till exchange of Eq. 25e bounded within the routing, Sect. 4): a Picard
iteration on `(q, N)`, since the supply (dissipation through `q`, and frictional heat through `N`
unless `model.lagged_friction`), the channel indicator and the till uptake (through `q` and `N`) and,
with [`HydraulicRouting`](@ref) (or [`WeightedRouting`](@ref) in a steady solve), the flow direction
depend on the solution. Every sweep goes through steps 2-4 of Sect. 4 in order: routing with the
bounded till exchange, channel indicator, flux balance. Stops when both `q` and `state.N` change by less
than `model.rtol` (relative to their maxima), or after `model.max_iters` sweeps. With
[`WeightedRouting`](@ref) in a steady solve, the graph (rebuilt every sweep from an under-relaxed `N`)
must also have settled, `routing_relaxation * res_route <= rtol`; a graph still moving after
`max_iters ÷ 2` sweeps is held so that `(q, N)` converge on it, and the solve returns `false`
(`model.info.converged = false`, `model.info.res_route` the remaining gap), since the result is then
not a fixed point of Eq. (26).

`chi^(j)` (`model.chi_start`) and `N^(j-1)` (`model.N_lag`) are the channel indicator and `state.N` the
solve starts from: the sliding pressure `N_slide` of the previous solve (Eq. 24b, see
[`update_sliding_pressure!`](@ref): `min(N_t, p_i)` on a dry or cold cell, `p_i` on a hard bed); on
the first step, whatever `state.N` holds (the initial condition; 0 for a fresh `HydroState`, so no creep
closure, no routing correction, and a lagged till that loosens: call [`update_steady_state!`](@ref)
first). In a steady solve it is not used except as the starting `N` of the routing (the fixed point
`N^(j-1) = N`).
"""
function solve_aguamenti!(model::AguamentiHydroModel, grid::AbstractHydroGrid, state::HydroState, mode)
    r = model.routing
    p = model.params
    info = model.info
    info.routing_builds = 0
    info.G_gamma = NaN
    info.clipped = 0
    info.res_route = NaN

    # The state the solve starts from: chi^(j) and N^(j-1).
    model.chi_start .= model.chi
    model.N_lag .= state.N

    # Supply: geometric potential, overburden, the routing graph when it is fixed over the solve, and the
    # gradient the elements use (A2).
    update_phi0!(r, grid, state)
    @. model.p_i = p.rho_i * p.g * state.h
    setup_routing!(model, model.routing_potential, grid, state, mode)
    update_channel_thresholds!(model, grid, state)

    # Basal stress at the N the solve starts from (lagged friction), or at every sweep's N below.
    model.lagged_friction && update_tau_b!(r, state, r.sliding_law)
    info.dissipation_held = false
    info.res_diss = NaN
    start_time = time()
    converged = false
    n_iters = model.max_iters
    empty!(info.history)
    area = grid.dx * grid.dy
    for iter in 1:model.max_iters
        model.q_prev .= r.q
        model.N_prev .= state.N

        sweep_routing!(model, model.routing_potential, grid, state, mode, iter)

        # Supply (Eq. 7): fixed melt, water from above, frictional heat, and the dissipation of the routed
        # flux q_in -- also in a cell the till leaves dry, which the water still crosses. (From the q with
        # dry cells zeroed, a cell at the wet/dry edge would lose its dissipation melt one sweep after
        # turning dry and regain it one sweep after turning wet: a period-2 flip.)
        model.lagged_friction || update_tau_b!(r, state, r.sliding_law)
        @. r.mdot_total = r.mdot_fixed + r.i_eb
        add_friction_term!(r)
        dissipation_flux!(model.dissipation_rule, model, iter)
        add_dissipation_term!(r, r.dissipation_melt)
        r.Q_sens .= 0

        # Routing with the till exchange bounded by the water available (Eqs. 8-9, 27).
        update_till_storage!(model, grid, state, mode)
        tape = r.routing_tape
        tape.valid || record_routing_tape!(tape, r, grid, state)
        uptake = till_uptake(model.till_exchange, model, state, mode, area, grid.dx)
        replay_aguamenti_tape!(uptake, r.psi_out, model.mdot_t, model.dry, model.refreeze, model.freeze_deficit, r.mdot_total,
                               model.release_max, model.cold, state.mask, tape, grid.Nx, grid.Ny, grid.dx, grid.dy, p.rho_w)
        update_q_from_psi_out!(r, grid, state)
        # A dry or cold cell has no interfacial water (the face-averaged q would still see its inflow). The
        # flux balance keeps the routed flux, so the till of a dry cell keeps relaxing towards the pressure
        # of the water it absorbs rather than towards the q = 0 pressure (which would flip it wet again).
        model.q_in_prev .= model.q_in
        model.q_in .= r.q
        z = zero(eltype(r.q))
        @. r.q = ifelse(model.dry != 0, z, r.q)

        # Channel indicator (steady state, or advanced over the step), flux balance, regimes and
        # grounding-line correction; then the sliding pressure.
        update_flux_balance!(model, grid, state, mode)
        update_sliding_pressure!(model, grid, state, mode isa SteadyMode)

        q_scale = max(masked_max_abs(grid, r.q, state.mask), eps(eltype(r.q)))
        N_scale = max(masked_max_abs(grid, state.N, state.mask), eps(eltype(state.N)))
        res_N = masked_max_abs_diff(grid, state.N, model.N_prev, state.mask) / N_scale
        res_q = masked_max_abs_diff(grid, r.q, model.q_prev, state.mask) / q_scale
        push!(info.history, (res_q, res_N))
        # A graph rebuilt every sweep from an under-relaxed N (WeightedRouting in a steady solve) must
        # have settled too: the next rebuild would move N_route by `routing_relaxation * res_route`. Once
        # the graph is held, (q, N) converge on it, but the solve only counts as converged if it had
        # settled (`routing_settled` is true where the graph does not follow the solution this way).
        route_ok = routing_settled(model, model.routing_potential, grid, state, mode, N_scale)
        if iter > 1 && res_q <= model.rtol && res_N <= model.rtol && (route_ok || iter > max(model.max_iters ÷ 2, 1))
            converged = route_ok
            n_iters = iter
            break
        end
    end

    info.iters = n_iters
    info.converged = converged
    info.res_q, info.res_N = isempty(info.history) ? (NaN, NaN) : info.history[end]
    N_scale = max(masked_max_abs(grid, state.N, state.mask), eps(eltype(state.N)))
    info.res_route = routing_residual(model, model.routing_potential, grid, state, mode, N_scale)
    if info.dissipation_held
        q_scale = max(masked_max_abs(grid, model.q_in, state.mask), eps(eltype(r.q)))
        info.res_diss = masked_max_abs_diff(grid, model.q_diss, model.q_in, state.mask) / q_scale
    end
    info.elapsed = time() - start_time
    if model.verbose
        status = converged ? "converged" : "did NOT converge (hit max_iters)"
        @info "Aguamenti (q, N) Picard loop: $status after $n_iters iteration(s) in $(round(time() - start_time, digits = 4)) s"
    end
    return converged
end


# The flux the dissipation melt of this sweep is evaluated from, left in r.q (see AbstractDissipationRule).
dissipation_flux!(::FreeDissipation, model, iter) = (model.routing.q .= model.q_in; nothing)

function dissipation_flux!(rule::HeldDissipation, model, iter)
    if iter == rule.after
        @. model.q_diss = (model.q_in + model.q_in_prev) / 2     # the centre of a period-2 cycle
        model.info.dissipation_held = true
    end
    model.routing.q .= iter >= rule.after ? model.q_diss : model.q_in
    return nothing
end

"""
$(TYPEDSIGNATURES)

Steady state of the Aguamenti model for the current geometry and forcing: the till in equilibrium with
the interface (`N_t = N`, no exchange) and the channel indicator in the steady state of Eq. (21),
keeping its current value in the bistable band `q_sn <= q < q_on` (so the result depends on the
history held in `model.chi`, the hysteresis of Sect. 3.6).
"""
function update_steady_state!(model::AguamentiHydroModel, grid::AbstractHydroGrid, state::HydroState)
    solve_aguamenti!(model, grid, state, SteadyMode()) ||
        @warn "Aguamenti steady solve did not converge in max_iters = $(model.max_iters) sweeps" maxlog = 10
    return nothing
end


"""
$(TYPEDSIGNATURES)

Advance an [`AguamentiHydroModel`](@ref) by `dt` [s] (Sect. 4, Eqs. 25a-25f): solve the routing, the
channel indicator `model.chi` (Eq. 21, advanced from its value at the start of the step before the flux
balance uses it) and the flux balance with the till exchange of this step, then advance the till
effective stress `model.N_t` (Eq. 28a): the stored water changes by the exchange applied in the routing
(bounded by the water available, Eq. 27), `N_t` follows from the storage law, and the undrained
response to the thinning/thickening `dH_dt` (field [m/s], or `nothing` for zero) is added. On return
`state.N` is the sliding pressure of this step.
"""
function step!(model::AguamentiHydroModel, grid::AbstractHydroGrid, state::HydroState, dt::Real; dH_dt = nothing)
    dt > 0 || throw(ArgumentError("dt must be positive (got $dt)"))
    dH_dt === nothing || size(dH_dt)[1:2] == (grid.Nx, grid.Ny) ||
        throw(ArgumentError("dH_dt size $(size(dH_dt)) != grid size $((grid.Nx, grid.Ny))"))
    T = eltype(state.N)
    mode = StepMode(T(dt), dH_dt)
    solve_aguamenti!(model, grid, state, mode) ||
        @warn "Aguamenti step did not converge in max_iters = $(model.max_iters) sweeps" maxlog = 10

    p, ws = model.params, model.workspace
    @inbounds for j in 1:grid.Ny, i in 1:grid.Nx
        state.mask[i, j] == GROUNDED || continue
        # Till: stored water changes by the bounded exchange, then the undrained increment (Eqs. 27, 28a).
        if ws.cold[i, j] == 0 && ws.kappa[i, j] > 0
            ws.N_t[i, j] = till_from_exchange(p, ws.N_t[i, j], ws.mdot_t[i, j], mode.dt, ws.storage[i, j], ws.p_i[i, j],
                                              ws.dN_u[i, j])
            # a dry cell slides on the till at its end-of-step stress (Eq. 24b)
            ws.dry[i, j] != 0 && (state.N[i, j] = min(ws.N_t[i, j], ws.p_i[i, j]))
        end
    end
    return nothing
end


"""
$(TYPEDSIGNATURES)

`state.N` for a new sliding speed `abs_v_b` [m/s], with the routing held (`q`, `Psi`, the dry cells
and the till exchange of the last solve): the channel thresholds and the flux balance alone,
re-evaluated with the new cavity opening, and the overburden from the current `state.h` (a host may
refresh the thickness in place between solves). See [`N_responds_to_ub`](@ref).
"""
function N_from_ub!(model::AguamentiHydroModel, grid::AbstractHydroGrid, state::HydroState, abs_v_b)
    p = model.params
    model.routing.abs_v_b .= abs_v_b
    @. model.p_i = p.rho_i * p.g * state.h
    update_channel_thresholds!(model, grid, state)
    update_flux_balance!(model, grid, state, nothing)
    update_sliding_pressure!(model, grid, state, false)
    return state.N
end
