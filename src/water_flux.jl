#################################
# Model: Kazmierczak et al 2024 #
#################################


"""
$(TYPEDSIGNATURES)

For the Kazmierczak et al 2024 model, update the distributed water flux q [m² s⁻¹] via a recursive algorithm following Le Brocq's code for Le Brocq et al 2006 (https://doi.org/10.1016/j.cageo.2006.05.003).
First we update the geometric potential that depends on the icethickness h and bedrock elevation b. Then we fill up the local minima of this potential to avoid the 
issue of water getting stuck. We then update the gradients of the potential in x, y. Further, we smooth these gradients with a convolution in order to incorporate
the effects of the stress-gradient coupling. For the latter concept, see for example Eq. (8.98) from Cuffey & Patterson 2010 book, Eq. (15) from Kamb et al 1986, 
Gudmundsson 2002, and references therein. Finally, we calculate the scalar water flux out of each grid cell ψ_out following the aforementioned recursive algorithm from Le Brocq.
To go from ψ_out to q, we use a correction factor, here called corfac, that can be derived using the definition of ψ_out given by Eq. (R4) of the referee reports of Kazmierczak et al 2024
(https://egusphere.copernicus.org/preprints/2024/egusphere-2024-466/egusphere-2024-466-AC1-supplement.pdf). Finally, the 0 ≤ q ≤ 1e5 is calculated, with limits set by Frank Pattyn in KORI-ULB
for numerical stability.

"""
function update_q!(model::KazmierczakHydroModel, grid::OGRectHydroGrid, state::HydroState)

    # Update ϕ₀
    update_ϕ₀!(model, grid, state)
    
    # Update ϕ₀ and consequently also updates h to reflect changes in ϕ₀. It fills the local minima of ϕ₀ to avoid water getting stuck in there.
    potential_filling!(model, grid, state) 

    # Update the gradients of the geometric potential ϕ₀
    update_potential_gradients!(model, grid)

    # Smoothen the gradients of ϕ₀ to incorporate the concept of stress-gradient coupling.
    update_smoothed_potential_gradients!(model, grid, state) 

    # Compute ψ_out via a recursive algorithm that traverses upsteam to compute ψ_out at the highest ϕ₀ and then come back down again, for each unvisited point on the grounded ice grid.
    update_ψ_out!(model, grid, state)
    
    # This is the factor to go from ψ_out to q. It can be derived using the definition of ψ_out = integral of q ⋅ n with n the outward normal 
    # and the line integral is over the whole sides of the grid cell that water leaves the cell, which can be 1 side for when q points to any of the 4 cardinal neighbours's center or 2 sides.
    @. model.corfac.data = (abs(model.minus_∇ϕ₀_smoothed_x.data) * grid.grid.Δyᵃᶜᵃ + abs(model.minus_∇ϕ₀_smoothed_y.data) * grid.grid.Δxᶜᵃᵃ) / (sqrt(model.minus_∇ϕ₀_smoothed_x.data^2 + model.minus_∇ϕ₀_smoothed_y.data^2) + 1e-15) # 1e-15 for division by zero safety

    # Limits on q are heuristic and chosen by Frank Pattyn for numerical stability.
    @. model.q = model.ψ_out / model.corfac
    @. model.q.data = min(max(model.ψ_out.data / model.corfac.data, 0), 1e5)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Update the water layer thickness W that is part of the HydroState. See Eq. (8) from Kazmierczak et al 2022.
"""
function update_W!(model::KazmierczakHydroModel, grid::OGRectHydroGrid, state::HydroState)

    abs_∇ϕ₀_smoothed_active = @views interior(model.abs_∇ϕ₀_smoothed, :, :, 1)[state.mask .== 1] # this allocates memory and we can avoid that with a for loop but the current method is faster
    abs_∇ϕ₀_smoothed_active_mean = mean(abs_∇ϕ₀_smoothed_active)
    @. state.W.data = min(model.Wmax, max(model.Wmin, (12 * model.η_w * model.q.data / abs_∇ϕ₀_smoothed_active_mean)^(1/3)))
    # fill_halo!(state.W, grid)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Update the geometric potential ϕ₀ and set the values for the halo points (ghost points that handle boundary conditions automatically with Oceananigans).
"""
function update_ϕ₀!(model::KazmierczakHydroModel, grid::OGRectHydroGrid, state::HydroState)
    
    @. model.ϕ₀ = model.ρ_i * model.g * state.h + model.ρ_w * model.g * state.b
    fill_halo!(model.ϕ₀, grid)

    return nothing
    
end


"""
$(TYPEDSIGNATURES)

Updates ϕ₀ and consequently also updates h to reflect changes in ϕ₀. It fills the local minima of ϕ₀ to avoid water getting stuck in there.
"""
function potential_filling!(model::KazmierczakHydroModel, grid::OGRectHydroGrid, state::HydroState)

    ϕ₀ = model.ϕ₀
    ϕ₀_tmp = model.ϕ₀_tmp

    ϕ₀_tmp .= ϕ₀ # update ϕ₀_tmp to match ϕ₀ before we start the potential filling
    fill_halo!(ϕ₀_tmp, grid)

    for _ in 1:model.fill_iters # one iteration might fill a small hole, but many can ensure even deep basins can be filled 
        @inbounds for j in 1:grid.grid.Ny
            for i in 1:grid.grid.Nx

                p = ϕ₀[i, j]

                # Cardinal neighbours of i, j. ϕ₀ is an offset array and we use a 1, 1 halo so indexing here is safe and thus we can use @inbounds
                # concretely, the indices of ϕ₀ range from 0 to Nx+1 and from 0 to Ny+1 for the x and y dimensions
                p1, p2 = ϕ₀[i+1, j], ϕ₀[i-1, j]
                p3, p4 = ϕ₀[i, j+1], ϕ₀[i, j-1]
                
                if p < p1 && p < p2 && p < p3 && p < p4 # identify a local minimum where the gradient is effectively pointing inward from all sides
                    ϕ₀_tmp[i, j] = (p1 + p2 + p3 + p4) / 4.0 # Laplacian smoothing step, replacing the pit with the average height of its neighbouring points effectively filling the hole
                end
            end
        end
        ϕ₀ .= ϕ₀_tmp
        fill_halo!(ϕ₀, grid)
    end

    # correction to h from potential filling stored in the ice thickness field of the model rather than the hydrology state since we don't want to affect the other calculations like effective pressure
    # this is essentially only used in the update_smoothed_potential_gradients! function to calculate the average grounded ice thickness
    @. model.h = (model.ϕ₀ - model.ρ_w * model.g * state.b)/(model.ρ_i * model.g) 

    return nothing

end


"""
$(TYPEDSIGNATURES)

Compute (the negative of) the gradients of the geometric potential ϕ₀ and its absolute value.
"""
function update_potential_gradients!(model::KazmierczakHydroModel, grid::OGRectHydroGrid)

    model.minus_∇ϕ₀_x .= -∂x(model.ϕ₀)
    model.minus_∇ϕ₀_y .= -∂y(model.ϕ₀)
    fill_halo!(model.minus_∇ϕ₀_x, grid) # halo used in accumulate_ψ_out so we have to update them here in case longcoupwater = 0 which is relevant for the if statement in update_smoothed_potential_gradients
    fill_halo!(model.minus_∇ϕ₀_y, grid)

    @. model.abs_∇ϕ₀.data = sqrt(model.minus_∇ϕ₀_x.data^2 + model.minus_∇ϕ₀_y.data^2)
    fill_halo!(model.abs_∇ϕ₀, grid)

    return nothing

end


"""
$(TYPEDSIGNATURES)

Update the smoothed geometric potentials to incorporate the effects of the stress-gradient coupling. See also the description of the update_q! function.

The water flux at a given point is influenced by variations in ice thickness some distance away. To account for this we perform a convolution of the gradient of the potential
such that the influence of nearby points is now incorporated into the value of the gradient of the potential at that point.
"""
function update_smoothed_potential_gradients!(model::KazmierczakHydroModel, grid::OGRectHydroGrid, state::HydroState)

    # If the longcoupwater parameter is set to zero we do not need to perform the smoothening here
    if model.longcoupwater == 0.0
        model.minus_∇ϕ₀_smoothed_x.data .= model.minus_∇ϕ₀_x.data
        model.minus_∇ϕ₀_smoothed_y.data .= model.minus_∇ϕ₀_y.data
        @. model.abs_∇ϕ₀_smoothed.data = abs(model.minus_∇ϕ₀_x.data) + abs(model.minus_∇ϕ₀_y.data)
        # fill_halo!(model.abs_∇ϕ₀_smoothed, grid) # used in accumulate_ψ_out! but this is commented out because we use .data in the line above which automatically sets the halo points as well
        return nothing
    else
        # Average grounded-ice ice thickness
        h_active = @views interior(model.h, :, :, 1)[state.mask .== 1] # this allocates memory and we can avoid that with a for loop but the current method is faster
        h_avg = max(mean(h_active), 10.0) # see Cuffey & Paterson 2010 end of sec. 8.7.2 for the maximum value
        
        # Radius of influence
        scale = h_avg * model.longcoupwater * 2.0 # represents half the radius of the influence zone of the ice thickness on water
        width = 2.0 * scale # radius of influence, after which the cone ends and the weighting of any points beyond that range is 0
        Δ = (grid.grid.Δxᶜᵃᵃ + grid.grid.Δyᵃᶜᵃ)/2.0 # # assumes that in each dimension the grid cell center spacing is uniform 
        if width <= Δ
            scale = Δ / 2.0 + 1.0
        end
        
        # Size of the kernel
        maxlevel = 2 * round(Int, width / Δ - 0.5) + 1 # e.g. 9

        # Filter radius boundary is the radius of the kernel, it calculates how many pixels extend from the center of that kernel window to its edge, for example for 9 x 9 the frb = 4
        frb = Int((maxlevel - 1) / 2) 

        # Filtering kernel
        kernel = zeros(maxlevel, maxlevel, 1)
        for nj in 1:maxlevel, ni in 1:maxlevel
            dist = sqrt((Δ * (ni - frb - 1))^2 + (Δ * (nj - frb - 1))^2) / scale
            kernel[ni, nj, 1] = max(0.0, 1.0 - dist / 2.0)
        end
        kernel ./= sum(kernel) # normalization is important as it ensures that the smoothing doesn't change the total amount of potential but only redistributes it

        # This slides the cone kernel over every pixel of the gradient and each pixel's new value becomes a weighted average of its neighbors and itself, 
        # at the boundaries we reflect the array when the kernel is outside, imfilter! allocates quite a bit of memory e.g. see discussion here https://discourse.julialang.org/t/how-to-accelerate-the-imfiter-operation/102965/3
        imfilter!(model.minus_∇ϕ₀_smoothed_x, model.minus_∇ϕ₀_x, centered(kernel))
        imfilter!(model.minus_∇ϕ₀_smoothed_y, model.minus_∇ϕ₀_y, centered(kernel))
        
        # Halo used in the accumulate_ψ_out! so we update them according to the BC of the field
        fill_halo!(model.minus_∇ϕ₀_smoothed_x, grid)
        fill_halo!(model.minus_∇ϕ₀_smoothed_y, grid) 

        # Update the absolute value of the gradient of the potential
        @. model.abs_∇ϕ₀_smoothed.data = abs(model.minus_∇ϕ₀_smoothed_x.data) + abs(model.minus_∇ϕ₀_smoothed_y.data) # here we don't need @at (Center, Center, Center) because fields.abs_∇ϕ₀ is initialized already as CenterField
        # fill_halo!(model.abs_∇ϕ₀_smoothed, grid) # used in accumulate_ψ_out! but this is commented out because we use .data in the line above which automatically sets the halo points as well

        return nothing
    end

end


"""
$(TYPEDSIGNATURES)

Helper function to the recursive function to calculate the ψ_out for every grid cell that has grounded ice.
"""
function accumulate_ψ_out!(model::KazmierczakHydroModel, i, j, grid::OGRectHydroGrid, state::HydroState)

    # We initialize all unvisited cells with -1.0 and q is positive semi-definite (see Eq. (R4) mentioned in update_q! function description) so if its positive then it has been visited and ψ_out calculated
    if model.ψ_out[i, j] ≥ 0.0
        return model.ψ_out[i, j]
    end

    # See Eq. (6) from Le Brocq et al 2009 (https://doi.org/10.3189/002214309790152564) for this udpate on ψ_out. We ensure that ψ_out stays non-negative, because ṁ can get negative.
    model.ψ_out[i, j] = max(0.0, model.ṁ_over_ρ_w[i, j]) * grid.grid.Δxᶜᵃᵃ * grid.grid.Δyᵃᶜᵃ # assumes that in each dimension the grid cell center spacing is uniform 

    @inbounds for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))

            ni, nj = i+di, j+dj 

            # Calculate the fraction of water w coming from the neighbour ni, nj
            w = -(model.minus_∇ϕ₀_smoothed_x[ni, nj]*di + model.minus_∇ϕ₀_smoothed_y[ni, nj]*dj) / (model.abs_∇ϕ₀_smoothed[ni, nj] + 1e-15) # safety 1e-15 treats the case when abs_∇ϕ₀ = 0 to allow w to go to 0 instead of NaN
        
            if w > 0
                model.ψ_out[i, j] += accumulate_ψ_out!(model, ni, nj, grid, state) * w
            end

    end
    return model.ψ_out[i, j]
end


"""
$(TYPEDSIGNATURES)

Recursive function to calculate the ψ_out for every grid cell that has grounded ice.
We initialize ψ_out to -1 since it is by definition positive semi-definite and hence we 
know that if a grid point has negative ψ_out, it is still unvisited. 
"""
function update_ψ_out!(model::KazmierczakHydroModel, grid::OGRectHydroGrid, state::HydroState)
    
    @inbounds for j in 1:grid.grid.Ny
        for i in 1:grid.grid.Nx
            if state.mask[i, j] == 1
                model.ψ_out[i, j] = -1.0
            end
        end
    end

    @inbounds for j in 1:grid.grid.Ny
        for i in 1:grid.grid.Nx
            if state.mask[i, j] == 1
                accumulate_ψ_out!(model, i, j, grid, state)
            end
        end
    end

    return nothing

end
