"""
$(TYPEDSIGNATURES)

Load a `.mat` file from the Kazmierczak et al. (2024) Thwaites hydrology output
and return the processed fields needed for the simulation.

# Arguments
- `path::String`: path to the `.mat` file.

# Returns
- `Nx, Ny`: grid dimensions.
- `xlims, ylims`: grid extent in meters (cell-centered, with half-cell padding to extend to grid edge faces).
- `mask`: grounded ice mask.
- `h`: ice thickness (m).
- `b`: bed elevation (m).
- `abs_v_b`: basal velocity magnitude (m/s).
- `A_visc`: viscocity parameter in Glen's flow law.
- `ṁ_over_ρ_w`: basal melt rate per unit area divided by water density (m s⁻¹).
- `κ`: bed hardness (0: hard, 1: soft).
"""
function load_Kazmierczak(path::String; bed_rheology = :hard)
    
    data = matread(path)
    Nx, Ny = size(data["H"])
    mask = data["MASKo"]
    h = data["H"]
    b = data["B"]
    abs_v_b = data["ub"]
    A_visc = data["A"]
    ṁ = perYear2perSecond.(data["Bmelt"]) .* 1000 # They stored this variable in per year units and as ṁ/ρ_w so we multiply by ρ_w = 1000 to get ṁ

    # Note: x and y are swapped in the file, and converted from km to m
    xc = Km2m.(data["y"]) 
    yc = Km2m.(data["x"])
    xlims, ylims = compute_lims(xc, yc)

    function initialize_κ!(Nx, Ny, b; bed_rheology = bed_rheology)
        
        T = eltype(b)

        if bed_rheology == :hard 
            κ = zeros(T, Nx, Ny) 
        elseif bed_rheology == :soft
            κ = ones(T, Nx, Ny)
        elseif bed_rheology == :mixed
            κ = zeros(T, Nx, Ny)
            κ[b .< -1000] .= T(1.0)
        elseif bed_rheology == :mixed_smooth
            κ = zeros(T, Nx, Ny)
            for j in 1:Ny
                for i in 1:Nx
                    if b[i, j] <= -1500
                        κ[i, j] = 1
                    elseif b[i, j] <= -500
                        κ[i, j] = (b[i, j] - (-500))/(-1500 - (-500))
                    end
                end
            end
        end

        return κ

    end

    κ = initialize_κ!(Nx, Ny, b; bed_rheology)

    return Nx, Ny, xlims, ylims, mask, h, b, abs_v_b, A_visc, ṁ, κ

end


"""
$(TYPEDSIGNATURES)

Load an NCDatasets file from the yelmox and return the processed fields needed for the simulation of the Kazmierczak et al 2024 hydrology model.

# Arguments
- `path::String`: path to the `.nc` file.
- `resolution::Int`: either 16 or 32 km resolution

# Returns
- `Nx, Ny`: grid dimensions.
- `xlims, ylims`: grid extent in meters (cell-centered, with half-cell padding to extend to grid edge faces).
- `mask`: grounded ice mask.
- `h`: ice thickness (m).
- `b`: bed elevation (m).
- `abs_v_b`: basal velocity magnitude (m/s).
- `A_visc`: viscocity parameter in Glen's flow law.
- `ṁ`: basal melt rate per unit area (Kg m⁻² s⁻¹).
- `κ`: bed hardness (0: hard, 1: soft).
"""
function load_yelmox(path::String; bed_rheology = :mixed_smooth)
    
    ds = NCDataset(path)

    Nx = length(ds["xc"])
    Ny = length(ds["yc"])

    xlims = (minimum(ds["xc"][:]), maximum(ds["xc"][:]))
    ylims = (minimum(ds["yc"][:]), maximum(ds["yc"][:]))

    mask = reshape(Int.(ds["f_ice"][:] .* ds["f_grnd"][:] .> 0.0), Nx, Ny)
    h = reshape(ds["H_ice"][:], Nx, Ny)
    b = reshape(ds["z_bed"][:], Nx, Ny)
    ux_b = reshape(ds["ux_b"][:], Nx, Ny)
    uy_b = reshape(ds["uy_b"][:], Nx, Ny)
    abs_v_b = reshape(sqrt.(ux_b.^2 .+ uy_b.^2), Nx, Ny)
    A_visc = mean(reshape(ds["ATT"][:], Nx, Ny, :), dims = 3)[:, :, 1]
    ṁ = reshape(-ds["bmb"][:], Nx, Ny)
   
    function initialize_κ!(Nx, Ny, b; bed_rheology = bed_rheology)
        
        T = eltype(b)

        if bed_rheology == :hard 
            κ = zeros(T, Nx, Ny) 
        elseif bed_rheology == :soft
            κ = ones(T, Nx, Ny)
        elseif bed_rheology == :mixed
            κ = zeros(T, Nx, Ny)
            κ[b .< -1000] .= T(1.0)
        elseif bed_rheology == :mixed_smooth
            κ = zeros(T, Nx, Ny)
            for j in 1:Ny
                for i in 1:Nx
                    if b[i, j] <= -1500
                        κ[i, j] = 1
                    elseif b[i, j] <= -500
                        κ[i, j] = (b[i, j] - (-500))/(-1500 - (-500))
                    end
                end
            end
        end

        return κ

    end

    κ = initialize_κ!(Nx, Ny, b; bed_rheology = bed_rheology)

    return Nx, Ny, xlims, ylims, mask, h, b, abs_v_b, A_visc, ṁ, κ

end

