#################################
# Model: Kazmierczak et al 2024 #
#################################


"""
$(TYPEDSIGNATURES)

Basal shear stress tau_b [Pa] from `law`, given the current effective pressure `N` [Pa] and the
magnitude of the basal sliding velocity `abs_v_b` [m/s]. Feeds the frictional-heating term
tau_b*v_b/L_w of the melt rate (Eq. 3, Sec. 2.2.1 of Kazmierczak et al 2024). See the
`AbstractSlidingLaw` docstring in model.jl for the physics/provenance of each law. A plain scalar
function (used for tests/diagnostics on ordinary numbers) -- `update_tau_b!` below is the version
actually used inside the model's field broadcasts.
"""
calc_tau_b(::PrescribedFrictionSlidingLaw, N, abs_v_b) = zero(abs_v_b)

calc_tau_b(law::WeertmanSlidingLaw, N, abs_v_b) = law.C * abs_v_b^law.q

calc_tau_b(law::PowerPlasticSlidingLaw, N, abs_v_b) = law.c_till * N * (abs_v_b / law.u0)^law.q

calc_tau_b(law::RegularizedCoulombSlidingLaw, N, abs_v_b) = law.c_till * N * (abs_v_b / (abs_v_b + law.u0))^law.q


"""
$(TYPEDSIGNATURES)

Update `model.tau_b` in place from `sliding_law` and the current `state.N`/`model.abs_v_b`.
Duplicates the `calc_tau_b` formulas above with the law's parameters pulled out as plain scalar
locals first, rather than calling `calc_tau_b` from inside the `@.` broadcast, because
`OGRectHydroGrid`'s `Field` broadcasting goes through Oceananigans' `AbstractOperations` machinery,
which only recognises registered arithmetic/operators inside a broadcast -- not arbitrary
multi-argument user functions taking a struct argument. Writing the formula with plain arithmetic
(`+`, `*`, `/`, `^`) on fields/arrays and scalar locals, exactly as the rest of this model already
does (e.g. `update_N_inf!` in effective_pressure.jl), works uniformly for both grid backends.
"""
function update_tau_b!(model::KazmierczakHydroModel, state::HydroState, ::PrescribedFrictionSlidingLaw)
    model.tau_b .= 0.0
    return nothing
end

function update_tau_b!(model::KazmierczakHydroModel, state::HydroState, law::PrescribedFieldSlidingLaw)
    model.tau_b .= law.tau_b
    return nothing
end

function update_tau_b!(model::KazmierczakHydroModel, state::HydroState, law::WeertmanSlidingLaw)
    C, q = law.C, law.q
    @. model.tau_b = C * model.abs_v_b^q
    return nothing
end

function update_tau_b!(model::KazmierczakHydroModel, state::HydroState, law::PowerPlasticSlidingLaw)
    c_till, q, u0 = law.c_till, law.q, law.u0
    @. model.tau_b = c_till * state.N * (model.abs_v_b / u0)^q
    return nothing
end

# RegularizedCoulombSlidingLaw (scalar c_till) and RegularizedCoulombFieldSlidingLaw (per-cell
# c_till) share this one method: the formula is identical, and broadcasting already handles a
# scalar or a Field for c_till transparently, so there is nothing that actually differs per type.
function update_tau_b!(model::KazmierczakHydroModel, state::HydroState, law::Union{RegularizedCoulombSlidingLaw, RegularizedCoulombFieldSlidingLaw})
    c_till, q, u0 = law.c_till, law.q, law.u0
    @. model.tau_b = c_till * state.N * (model.abs_v_b / (model.abs_v_b + u0))^q
    return nothing
end

function update_tau_b!(model::KazmierczakHydroModel, state::HydroState, law::ShaktiRegularizedCoulombSlidingLaw)
    C, n, inv_n, lambda = law.C, law.n, law.inv_n, law.lambda
    @. model.tau_b = C * state.N * (model.abs_v_b / (model.abs_v_b + abs(state.N)^n * lambda))^inv_n
    return nothing
end


"""
$(TYPEDSIGNATURES)

`StaggeredFriction`: add the C-grid frictional heat `/ L_w` to `model.mdot_total`, from the current
cell-centred `model.tau_b` (through `beta = tau_b / |u_b|`) and the face velocities. See
[`StaggeredFriction`](@ref).
"""
function add_friction_term!(model::KazmierczakHydroModel, fd::StaggeredFriction)
    staggered_friction_kernel!(model.mdot_total, model.tau_b, model.abs_v_b, fd.ux, fd.uy, size(fd.ux)...,
                               model.L_w, fd.u_floor, fd.quadrature)
    return nothing
end

@inline stagger_beta(tau_b, abs_v_b, u_floor, i, j) = tau_b[i, j] / max(abs_v_b[i, j], u_floor)

# 2x2 Gauss quadrature points in the reference cell [-1, 1]^2 and the bilinear shape functions of the
# SW, SE, NE, NW corners there (Yelmo's gq2D).
const GQ_S3 = 1 / sqrt(3)
const GQ_PTS = ((-GQ_S3, -GQ_S3), (GQ_S3, -GQ_S3), (GQ_S3, GQ_S3), (-GQ_S3, GQ_S3))
@inline gq_interp(c, p) = ((1 - p[1]) * (1 - p[2]) * c[1] + (1 + p[1]) * (1 - p[2]) * c[2] +
                           (1 + p[1]) * (1 + p[2]) * c[3] + (1 - p[1]) * (1 + p[2]) * c[4]) / 4
# corner (ab-node) values of an acx / acy field around cell (i, j), SW, SE, NE, NW
@inline acx_corners(F, i, j, im1, jm1, jp1) = ((F[im1, jm1] + F[im1, j]) / 2, (F[i, jm1] + F[i, j]) / 2,
                                               (F[i, j] + F[i, jp1]) / 2, (F[im1, j] + F[im1, jp1]) / 2)
@inline acy_corners(F, i, j, im1, ip1, jm1) = ((F[im1, jm1] + F[i, jm1]) / 2, (F[i, jm1] + F[ip1, jm1]) / 2,
                                               (F[i, j] + F[ip1, j]) / 2, (F[im1, j] + F[i, j]) / 2)

function staggered_friction_kernel!(mdot_total, tau_b, abs_v_b, ux, uy, Nx, Ny, L_w, u_floor, quadrature)
    # face drag coefficients and tractions: beta on the face between i and i+1 (acx) / j and j+1 (acy)
    tx = similar(ux); ty = similar(uy)
    @inbounds for j in 1:Ny, i in 1:Nx
        ip1, jp1 = min(i + 1, Nx), min(j + 1, Ny)
        bc = stagger_beta(tau_b, abs_v_b, u_floor, i, j)
        tx[i, j] = (bc + stagger_beta(tau_b, abs_v_b, u_floor, ip1, j)) / 2 * ux[i, j]
        ty[i, j] = (bc + stagger_beta(tau_b, abs_v_b, u_floor, i, jp1)) / 2 * uy[i, j]
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        im1, ip1 = max(i - 1, 1), min(i + 1, Nx)
        jm1, jp1 = max(j - 1, 1), min(j + 1, Ny)
        Q = if quadrature
            cux, cuy = acx_corners(ux, i, j, im1, jm1, jp1), acy_corners(uy, i, j, im1, ip1, jm1)
            ctx, cty = acx_corners(tx, i, j, im1, jm1, jp1), acy_corners(ty, i, j, im1, ip1, jm1)
            acc = zero(eltype(mdot_total))
            for p in GQ_PTS
                acc += hypot(gq_interp(cux, p), gq_interp(cuy, p)) * hypot(gq_interp(ctx, p), gq_interp(cty, p))
            end
            acc / 4
        else
            (tx[im1, j] * ux[im1, j] + tx[i, j] * ux[i, j]) / 2 + (ty[i, jm1] * uy[i, jm1] + ty[i, j] * uy[i, j]) / 2
        end
        mdot_total[i, j] += Q / L_w
    end
    return nothing
end
