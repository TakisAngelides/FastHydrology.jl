#################################
# Model: Kazmierczak et al 2024 #
#################################

# The water source mdot_total: the fixed basal terms (G, q_T, i_eb), the dissipation melt
# |q . grad(phi0)| / L_w and the frictional heat tau_b * |u_b| / L_w, with their discretisations.


"""
$(TYPEDSIGNATURES)

Update the given terms of the water source between solves: the geothermal heat flux `G` and the
conductive heat flux into the ice `q_T` [W/m^2], and the water from above `i_eb` [kg/m^2/s]. Any keyword left
`nothing` keeps its current value. Recomputes the fixed part of the source,
`model.mdot_fixed = (G - q_T)/L_w`; the frictional and dissipation heat are recomputed by the
solver itself. This is the only way to change the source: the melt rate is always built from terms.
"""
function set_basal_terms!(model::AbstractKazmierczakHydroModel; G = nothing, q_T = nothing, i_eb = nothing)
    G            === nothing || (model.G            .= G)
    q_T          === nothing || (model.q_T          .= q_T)
    i_eb === nothing || (model.i_eb .= i_eb)
    @. model.mdot_fixed = (model.G - model.q_T) / model.L_w
    return model
end


"""
$(TYPEDSIGNATURES)

Trait controlling whether `update_q!` includes the dissipation melt rate |q * grad(phi0)| / L_w in the water source
(see `KazmierczakHydroModel`'s `dissipation_melt` keyword). Stored as a type parameter so the on/off choice is
resolved by multiple dispatch at compile time -- `update_q!` calls a helper on `model.dissipation_melt` that has one
method for `DissipationMeltOn` and one for `DissipationMeltOff`, rather than branching on a `Bool` field at runtime.
"""
abstract type AbstractDissipationMelt end
struct DissipationMeltOn  <: AbstractDissipationMelt end
struct DissipationMeltOff <: AbstractDissipationMelt end


"""
$(TYPEDSIGNATURES)

How the dissipation melt `q . grad(phi0) / L_w` is discretised (see `KazmierczakHydroModel`'s
`dissipation_discretization` keyword). The result is a cell-centred melt rate either way:
- [`CellCentredDissipation`](@ref) (default for all schemes except `Warner`): `|q| * |grad(phi0)| / L_w` from the centred q and
  centred gradient of the true potential.
- [`FaceDissipation`](@ref) (default with `Warner` routing): the product is formed on each face, where both factors live on a
  staggered grid -- face flux times the true-potential drop across the face, the energy the water
  releases crossing it -- and averaged to the centre. Signed (water pushed up the true potential
  gives a negative term). 4-neighbour schemes only.
"""
abstract type AbstractDissipationDiscretization end
"""$(TYPEDSIGNATURES)

See [`AbstractDissipationDiscretization`](@ref)."""
struct CellCentredDissipation <: AbstractDissipationDiscretization end
"""$(TYPEDSIGNATURES)

See [`AbstractDissipationDiscretization`](@ref)."""
struct FaceDissipation <: AbstractDissipationDiscretization end


"""
$(TYPEDSIGNATURES)

Compute the dissipation heat `model.Q_diss` [W/m^2] and add `Q_diss/L_w` to `model.mdot_total` in place
(`Q_diss` is set to zero when the term is off). Dispatched on
`model.dissipation_melt` so the off case costs nothing; shared by every `resolve_q!` method so the
term is computed identically regardless of which sliding law is active.
"""
add_dissipation_term!(model::AbstractKazmierczakHydroModel, ::DissipationMeltOff) = (model.Q_diss .= 0; nothing)

add_dissipation_term!(model::AbstractKazmierczakHydroModel, ::DissipationMeltOn) =
    add_dissipation_term!(model, model.dissipation_discretization)


# Dissipation heat Q_diss [W/m2] and its melt Q_diss/L_w added to mdot_total, per
# `model.dissipation_discretization`.
function add_dissipation_term!(model::AbstractKazmierczakHydroModel, ::CellCentredDissipation)
    @. model.Q_diss = abs(model.q * model.abs_grad_phi0)
    @. model.mdot_total += model.Q_diss / model.L_w
    return nothing
end

function add_dissipation_term!(model::AbstractKazmierczakHydroModel, ::FaceDissipation)
    diss = model.routing_tape.diss      # face-assembled dissipation melt [kg/m2/s]
    @inbounds for j in axes(diss, 2), i in axes(diss, 1)
        model.Q_diss[i, j]      = diss[i, j] * model.L_w
        model.mdot_total[i, j] += diss[i, j]
    end
    return nothing
end


"""
$(TYPEDSIGNATURES)

How the frictional-heating term `tau_b . u_b / L_w` of the melt rate is discretised (see
`KazmierczakHydroModel`'s `friction_discretization` keyword). Either way it is a cell-centred melt
rate, recomputed every Picard sweep from the current `model.tau_b` (so it follows an N-dependent
sliding law through the `(q, N)` loop):

- [`CellCentredFriction`](@ref) (default): `tau_b * |u_b|` from the cell-centred `model.tau_b` and
  `model.abs_v_b`.
- [`StaggeredFriction`](@ref): for velocities on an Arakawa C-grid, as an ice model such as Yelmo
  provides them (`u_x` on x-faces, `u_y` on y-faces). The sliding law is still evaluated at the cell
  centre; its drag coefficient `beta = tau_b / |u_b|` is averaged to each face, giving the face
  tractions `tau_x = beta_face * u_x`, `tau_y = beta_face * u_y` (the way Yelmo builds `taub_acx/acy`
  from `beta_acx/acy`). The heat is then either formed on the faces or at Gauss points -- see
  `StaggeredFriction`.
"""
abstract type AbstractFrictionDiscretization end

"""$(TYPEDSIGNATURES)\n\nDefault: `tau_b * |u_b|` at cell centres. See [`AbstractFrictionDiscretization`](@ref)."""
struct CellCentredFriction <: AbstractFrictionDiscretization end

"""
$(TYPEDSIGNATURES)

Frictional heating from C-grid (staggered) basal velocities. `ux[i, j]` is the x-velocity on the
face between cells `i` and `i+1` (Yelmo's `acx` convention), `uy[i, j]` the y-velocity on the face
between cells `j` and `j+1` (`acy`), both [m/s], size `(Nx, Ny)`. See
[`AbstractFrictionDiscretization`](@ref).

- `quadrature = false` (faces): `tau_x u_x` is formed on each x-face and `tau_y u_y` on each y-face --
  both factors of each product live on that face -- and each is averaged from the cell's two faces to
  its centre: `Q = (P_x[i-1/2] + P_x[i+1/2])/2 + (P_y[j-1/2] + P_y[j+1/2])/2`, `P = beta_face u_face^2 >= 0`.
  The energy-consistent staggered form (the same construction as `FaceDissipation`).
- `quadrature = true`: Yelmo's own `qb_method = 2` (`calc_basal_heating_nodes`): the face velocities
  and tractions are interpolated bilinearly to the 4 points of a 2x2 Gauss quadrature in the cell,
  `|u||tau|` is formed there and averaged. Use this to reproduce the frictional heat Yelmo itself
  uses in its ice-temperature solve.

`beta` at a cell centre is `tau_b / max(|u_b|, u_floor)` (default `u_floor` 1e-3 m/yr, Yelmo's
`ub_sq_min`). Faces on the domain edge use the one cell they have.
"""
struct StaggeredFriction{T <: AbstractFloat} <: AbstractFrictionDiscretization
    ux         ::Matrix{T}
    uy         ::Matrix{T}
    tx         ::Matrix{T}   # scratch: face tractions, overwritten every sweep (allocated once here, not per call)
    ty         ::Matrix{T}
    quadrature ::Bool
    u_floor    ::T
end
function StaggeredFriction(ux::AbstractMatrix, uy::AbstractMatrix; quadrature::Bool = false, u_floor = perYear2perSecond(1e-3))
    T = float(promote_type(eltype(ux), eltype(uy)))
    return StaggeredFriction{T}(Matrix{T}(ux), Matrix{T}(uy), similar(Matrix{T}(ux)), similar(Matrix{T}(uy)), quadrature, T(u_floor))
end

# The model converts the velocity arrays to its own float type `T` (a copy, with its own scratch), the way
# `convert_sliding_law` does for the sliding law; update the velocities through `model.friction_discretization`.
convert_friction_discretization(::Type{T}, fd::AbstractFrictionDiscretization) where {T <: AbstractFloat} = fd
convert_friction_discretization(::Type{T}, fd::StaggeredFriction{T}) where {T <: AbstractFloat} = fd
convert_friction_discretization(::Type{T}, fd::StaggeredFriction) where {T <: AbstractFloat} =
    StaggeredFriction{T}(T.(fd.ux), T.(fd.uy), similar(T.(fd.ux)), similar(T.(fd.uy)), fd.quadrature, T(fd.u_floor))


"""
$(TYPEDSIGNATURES)

Compute the frictional heat `model.Q_b` [W/m^2] from the current `model.tau_b` and add `Q_b/L_w` to
`model.mdot_total` in place, per `model.friction_discretization`. Shared by every `resolve_q!` method;
`model.tau_b` is always up to date by the time this is called (`update_tau_b!` runs first in every
`resolve_q!` method).
"""
add_friction_term!(model::AbstractKazmierczakHydroModel) = add_friction_term!(model, model.friction_discretization)

function add_friction_term!(model::AbstractKazmierczakHydroModel, ::CellCentredFriction)
    @. model.Q_b = model.tau_b * model.abs_v_b
    @. model.mdot_total += model.Q_b / model.L_w
    return nothing
end


"""
$(TYPEDSIGNATURES)

`StaggeredFriction`: the C-grid frictional heat `model.Q_b` [W/m^2], from the current cell-centred
`model.tau_b` (through `beta = tau_b / |u_b|`) and the face velocities, added as `Q_b / L_w` to
`model.mdot_total`. See [`StaggeredFriction`](@ref).
"""
function add_friction_term!(model::AbstractKazmierczakHydroModel, fd::StaggeredFriction)
    staggered_friction_kernel!(model.Q_b, model.tau_b, model.abs_v_b, fd.ux, fd.uy, fd.tx, fd.ty, size(fd.ux)...,
                               fd.u_floor, fd.quadrature)
    @. model.mdot_total += model.Q_b / model.L_w
    return nothing
end

@inline stagger_beta(tau_b, abs_v_b, u_floor, i, j) = tau_b[i, j] / max(abs_v_b[i, j], u_floor)

# 2x2 Gauss quadrature points in the reference cell [-1, 1]^2 and the bilinear shape functions of the
# SW, SE, NE, NW corners there (Yelmo's gq2D).
const GQ_S3 = 1 / sqrt(3)
const GQ_PTS = ((-GQ_S3, -GQ_S3), (GQ_S3, -GQ_S3), (GQ_S3, GQ_S3), (-GQ_S3, GQ_S3))
# the same points in the model's float type, so the quadrature does not promote a Float32 model to Float64
@inline gq_points(::Type{T}) where {T} = (s = one(T) / sqrt(T(3)); ((-s, -s), (s, -s), (s, s), (-s, s)))
@inline gq_interp(c, p) = ((1 - p[1]) * (1 - p[2]) * c[1] + (1 + p[1]) * (1 - p[2]) * c[2] +
                           (1 + p[1]) * (1 + p[2]) * c[3] + (1 - p[1]) * (1 + p[2]) * c[4]) / 4
# corner (ab-node) values of an acx / acy field around cell (i, j), SW, SE, NE, NW
@inline acx_corners(F, i, j, im1, jm1, jp1) = ((F[im1, jm1] + F[im1, j]) / 2, (F[i, jm1] + F[i, j]) / 2,
                                               (F[i, j] + F[i, jp1]) / 2, (F[im1, j] + F[im1, jp1]) / 2)
@inline acy_corners(F, i, j, im1, ip1, jm1) = ((F[im1, jm1] + F[i, jm1]) / 2, (F[i, jm1] + F[ip1, jm1]) / 2,
                                               (F[i, j] + F[ip1, j]) / 2, (F[im1, j] + F[i, j]) / 2)

# Convenience method that allocates the face-traction scratch itself (the model passes its preallocated one).
staggered_friction_kernel!(Q_b, tau_b, abs_v_b, ux, uy, Nx, Ny, u_floor, quadrature) =
    staggered_friction_kernel!(Q_b, tau_b, abs_v_b, ux, uy, similar(ux), similar(uy), Nx, Ny, u_floor, quadrature)

function staggered_friction_kernel!(Q_b, tau_b, abs_v_b, ux, uy, tx, ty, Nx, Ny, u_floor, quadrature)
    # face drag coefficients and tractions: beta on the face between i and i+1 (acx) / j and j+1 (acy),
    # written into the scratch arrays tx, ty
    gq = gq_points(eltype(Q_b))
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
            acc = zero(eltype(Q_b))
            for p in gq
                acc += hypot(gq_interp(cux, p), gq_interp(cuy, p)) * hypot(gq_interp(ctx, p), gq_interp(cty, p))
            end
            acc / 4
        else
            (tx[im1, j] * ux[im1, j] + tx[i, j] * ux[i, j]) / 2 + (ty[i, jm1] * uy[i, jm1] + ty[i, j] * uy[i, j]) / 2
        end
        Q_b[i, j] = Q
    end
    return nothing
end
