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

    # Compute psi_out via a recursive algorithm.
    update_psi_out!(model, grid, state)

    # Correction factor from psi_out to q.
    Dx = grid_dx(grid)
    Dy = grid_dy(grid)
    @. model.corfac.data = (abs(model.minus_grad_phi0_sx.data) * Dy + abs(model.minus_grad_phi0_sy.data) * Dx) /
                           (sqrt(model.minus_grad_phi0_sx.data^2 + model.minus_grad_phi0_sy.data^2) + 1e-15)

    # Limits on q are heuristic and chosen by Frank Pattyn for numerical stability.
    @. model.q.data = min(max(model.psi_out.data / model.corfac.data, 0), 1e5)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Update the water layer thickness W that is part of the HydroState. See Eq. (8) from Kazmierczak et al 2022.
"""
function update_W!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)

    abs_grad_phi0_s_active = @views interior(model.abs_grad_phi0_s, :, :, 1)[state.mask .== 1]
    abs_grad_phi0_s_mean   = mean(abs_grad_phi0_s_active)
    @. state.W.data = min(model.Wmax, max(model.Wmin, (12 * model.eta_w * model.q.data / abs_grad_phi0_s_mean)^(1/3)))

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

    Nx = grid_Nx(grid)
    Ny = grid_Ny(grid)

    for _ in 1:model.fill_iters
        @inbounds for j in 1:Ny
            for i in 1:Nx
                p = phi0[i, j]
                p1, p2 = phi0[i+1, j], phi0[i-1, j]
                p3, p4 = phi0[i, j+1], phi0[i, j-1]
                if p < p1 && p < p2 && p < p3 && p < p4
                    phi0_tmp[i, j] = (p1 + p2 + p3 + p4) / 4.0
                end
            end
        end
        phi0 .= phi0_tmp
        fill_halo!(phi0, grid)
    end

    # Correction to h from potential filling; stored separately so it does not
    # affect other calculations like effective pressure.
    @. model.h = (model.phi0 - model.rho_w * model.g * state.b) / (model.rho_i * model.g)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Compute (the negative of) the gradients of the geometric potential phi0 and its absolute value.
"""
function update_potential_gradients!(model::KazmierczakHydroModel, grid::AbstractHydroGrid)

    model.minus_grad_phi0_x .= -∂x(model.phi0)
    model.minus_grad_phi0_y .= -∂y(model.phi0)
    fill_halo!(model.minus_grad_phi0_x, grid)
    fill_halo!(model.minus_grad_phi0_y, grid)

    @. model.abs_grad_phi0.data = sqrt(model.minus_grad_phi0_x.data^2 + model.minus_grad_phi0_y.data^2)
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
        model.minus_grad_phi0_sx.data .= model.minus_grad_phi0_x.data
        model.minus_grad_phi0_sy.data .= model.minus_grad_phi0_y.data
        @. model.abs_grad_phi0_s.data  = abs(model.minus_grad_phi0_x.data) + abs(model.minus_grad_phi0_y.data)
        return nothing
    end

    # Average grounded-ice thickness
    h_active = @views interior(model.h, :, :, 1)[state.mask .== 1]
    h_avg    = max(mean(h_active), 10.0)

    # Radius of influence
    Dx    = grid_dx(grid)
    Dy    = grid_dy(grid)
    Delta = (Dx + Dy) / 2.0
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

    kernel = zeros(maxlevel, maxlevel, 1)
    for nj in 1:maxlevel, ni in 1:maxlevel
        dist = sqrt((Delta * (ni - frb - 1))^2 + (Delta * (nj - frb - 1))^2) / scale
        kernel[ni, nj, 1] = max(0.0, 1.0 - dist / 2.0)
    end
    kernel ./= sum(kernel)

    imfilter!(model.minus_grad_phi0_sx, model.minus_grad_phi0_x, centered(kernel))
    imfilter!(model.minus_grad_phi0_sy, model.minus_grad_phi0_y, centered(kernel))
    fill_halo!(model.minus_grad_phi0_sx, grid)
    fill_halo!(model.minus_grad_phi0_sy, grid)

    @. model.abs_grad_phi0_s.data = abs(model.minus_grad_phi0_sx.data) + abs(model.minus_grad_phi0_sy.data)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Helper function to the recursive function to calculate the psi_out for every grid cell that has grounded ice.
"""
function accumulate_psi_out!(model::KazmierczakHydroModel, i, j, grid::AbstractHydroGrid, state::HydroState)

    if model.psi_out[i, j] >= 0.0
        return model.psi_out[i, j]
    end

    Dx = grid_dx(grid)
    Dy = grid_dy(grid)
    model.psi_out[i, j] = max(0.0, model.mdot_rho_w[i, j]) * Dx * Dy

    @inbounds for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
        ni, nj = i + di, j + dj
        w = -(model.minus_grad_phi0_sx[ni, nj] * di + model.minus_grad_phi0_sy[ni, nj] * dj) /
            (model.abs_grad_phi0_s[ni, nj] + 1e-15)
        if w > 0
            model.psi_out[i, j] += accumulate_psi_out!(model, ni, nj, grid, state) * w
        end
    end

    return model.psi_out[i, j]

end


"""
$(TYPEDSIGNATURES)

Recursive function to calculate the psi_out for every grid cell that has grounded ice.
We initialize psi_out to -1 since it is by definition positive semi-definite and hence we
know that if a grid point has negative psi_out, it is still unvisited.
"""
function update_psi_out!(model::KazmierczakHydroModel, grid::AbstractHydroGrid, state::HydroState)

    Nx = grid_Nx(grid)
    Ny = grid_Ny(grid)

    @inbounds for j in 1:Ny, i in 1:Nx
        if state.mask[i, j] == 1
            model.psi_out[i, j] = -1.0
        end
    end

    @inbounds for j in 1:Ny, i in 1:Nx
        if state.mask[i, j] == 1
            accumulate_psi_out!(model, i, j, grid, state)
        end
    end

    return nothing

end
