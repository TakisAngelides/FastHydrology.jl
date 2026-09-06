"""
$(TYPEDSIGNATURES)

Compute the bed hardness field `κ` (0: hard, 1: soft) from the bed elevation `b` according to
`bed_rheology` (`:hard`, `:soft`, `:mixed`, or `:mixed_smooth`).
"""
function initialize_κ(Nx, Ny, b; bed_rheology)

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


"""
$(TYPEDSIGNATURES)

Load a `.mat` file from the Kazmierczak et al. (2024) Thwaites hydrology output
and return the processed fields needed for the simulation.

# Arguments
- `path::String`: path to the `.mat` file.
- `bed_rheology`: bed hardness rule used to compute `κ` (`:hard`, `:soft`, `:mixed`, or `:mixed_smooth`; default `:hard`).

# Returns
- `Nx, Ny`: grid dimensions.
- `xlims, ylims`: grid extent in meters (cell-centered, with half-cell padding to extend to grid edge faces).
- `mask`: grounded ice mask.
- `h`: ice thickness (m).
- `b`: bed elevation (m).
- `abs_v_b`: basal velocity magnitude (m/s).
- `A_visc`: viscocity parameter in Glen's flow law.
- `G`: geothermal heat flux (W m⁻²), for `KazmierczakHydroModel`'s `G_in`/`q_T_in` constructor (Eq. 3 of
  Kazmierczak et al 2024).
- `q_T`: conductive heat flux into the ice at the bed (W m⁻²). The source file has no field for this term
  (only relevant for cold-based ice), so it is returned as zero everywhere -- pass a real field yourself if
  your bed isn't uniformly temperate.
- `ṁ`: complete basal melt rate per unit area (kg m⁻² s⁻¹), i.e. the source model's own converged Eq. 3
  output (already includes its own frictional-heating and, likely, dissipation terms) -- for
  `KazmierczakHydroModel`'s `mdot_in` constructor, normally paired with `sliding_law =
  PrescribedFrictionSlidingLaw()`. If you want a real sliding law's `tau_b` (e.g. for `(q, N)`
  coupling) while still using this `ṁ`, pass `mdot_includes_friction = true` so its own frictional
  heating isn't added a second time -- see `AbstractMdotFriction`'s docstring in model.jl.
- `κ`: bed hardness (0: hard, 1: soft).
"""
function load_Kazmierczak(path::String; bed_rheology = :hard)
    
    data = matread(path)
    Nx, Ny = size(data["H"])
    mask = data["MASKo"]
    h = data["H"]
    b = data["B"]
    # `ub` is stored in per-year units, like `Bmelt` below -- the file's own `par["secperyear"]`
    # (3.1556926e7, matching FastHydrology's own SECONDS_PER_YEAR to 5 significant figures) is the
    # constant this source model uses to convert its per-year fields to SI, and raw `ub` values here
    # reach ~2634, which is only plausible as m/yr (a fast Thwaites trunk speed) -- taken as literal
    # m/s that would be supersonic ice flow.
    abs_v_b = perYear2perSecond.(data["ub"])
    # `A` (Glen's law rate factor) is also stored per-year: KORI-ULB's own reference implementation
    # of this effective-pressure formula (SchoofWaterFarField.m in
    # https://github.com/FrankPat/Kori-ULB/blob/main/subroutines/SchoofWaterFarField.m) explicitly
    # does `A = A/par.secperyear` alongside the same conversion for `ub`, confirming the source data
    # convention -- left unconverted, A_visc was ~3e7x too large, pinning N_inf at its sigmat*Po
    # floor almost everywhere (a degenerate, uniformly-low N field, not the sensible ~5 MPa
    # background with narrow low-N channels this model actually produces once fixed).
    A_visc = perYear2perSecond.(data["A"])
    ṁ = perYear2perSecond.(data["Bmelt"]) .* 1000 # They stored this variable in per year units and as ṁ/ρ_w so we multiply by ρ_w = 1000 to get ṁ
    # `G` (geothermal heat flux) is stored directly in W/m^2 -- values run 0.086-0.14 here, squarely in the
    # plausible range for Antarctica, so unlike ub/A/Bmelt above it needs no unit conversion.
    G = data["G"]
    q_T = zeros(eltype(G), Nx, Ny)

    # Note: x and y are swapped in the file, and converted from km to m
    xc = Km2m.(data["y"])
    yc = Km2m.(data["x"])
    xlims, ylims = compute_lims(xc, yc)

    κ = initialize_κ(Nx, Ny, b; bed_rheology)

    return Nx, Ny, xlims, ylims, mask, h, b, abs_v_b, A_visc, G, q_T, ṁ, κ

end


"""
$(TYPEDSIGNATURES)

Load an NCDatasets file from the yelmox and return the processed fields needed for the simulation of the Kazmierczak et al 2024 hydrology model.

# Arguments
- `path::String`: path to the `.nc` file.
- `bed_rheology`: bed hardness rule used to compute `κ` (`:hard`, `:soft`, `:mixed`, or `:mixed_smooth`; default `:mixed_smooth`).

# Returns
- `Nx, Ny`: grid dimensions.
- `xlims, ylims`: grid extent in meters (cell-centered, with half-cell padding to extend to grid edge faces).
- `mask`: grounded ice mask.
- `h`: ice thickness (m).
- `b`: bed elevation (m).
- `abs_v_b`: basal velocity magnitude (m/s).
- `A_visc`: viscocity parameter in Glen's flow law.
- `G`: geothermal heat flux (W m⁻²), for `KazmierczakHydroModel`'s `G_in`/`q_T_in` constructor (Eq. 3 of
  Kazmierczak et al 2024).
- `q_T`: conductive heat flux into the ice at the bed (W m⁻²), Yelmo's own `Q_ice_b`, for the same
  constructor.
- `ṁ`: complete basal melt rate per unit area (Kg m⁻² s⁻¹), i.e. Yelmo's own converged basal mass balance
  (already includes its own frictional-heating term `Q_b`) -- for `KazmierczakHydroModel`'s `mdot_in`
  constructor, normally paired with `sliding_law = PrescribedFrictionSlidingLaw()`. If you want a real
  sliding law's `tau_b` (e.g. for `(q, N)` coupling) while still using this `ṁ`, pass
  `mdot_includes_friction = true` so `Q_b` isn't added a second time -- see `AbstractMdotFriction`'s
  docstring in model.jl.
- `κ`: bed hardness (0: hard, 1: soft).
"""
function load_yelmox(path::String; bed_rheology = :mixed_smooth)
    
    ds = NCDataset(path)

    Nx = length(ds["xc"])
    Ny = length(ds["yc"])

    # xc/yc carry a "units" = "km" attribute in yelmox restart files (confirmed against
    # test/Kazmierczak et al 2024/input/yelmox/{32km,16km} restarts, where xc spacing is exactly
    # 32.0/16.0), so they need the same km -> m conversion as load_Kazmierczak's xc/yc above.
    xc = Km2m.(ds["xc"][:])
    yc = Km2m.(ds["yc"][:])
    xlims, ylims = compute_lims(xc, yc)

    mask = reshape(Int.(ds["f_ice"][:] .* ds["f_grnd"][:] .> 0.0), Nx, Ny)
    h = reshape(ds["H_ice"][:], Nx, Ny)
    b = reshape(ds["z_bed"][:], Nx, Ny)
    ux_b = reshape(ds["ux_b"][:], Nx, Ny)
    uy_b = reshape(ds["uy_b"][:], Nx, Ny)
    # ux_b/uy_b carry a "units" = "m/yr" attribute in yelmox restart files (confirmed against
    # test/Kaz24_antarctica/data/16km/yelmo_restart.nc), so the combined speed
    # needs the same per-year -> per-second conversion applied to Bmelt in load_Kazmierczak above.
    abs_v_b = perYear2perSecond.(reshape(sqrt.(ux_b.^2 .+ uy_b.^2), Nx, Ny))
    # ATT (Glen's law rate factor) is also per-year: Yelmo.jl's own rate-factor constants are
    # explicitly documented in units of [1/yr / Pa^3] (src/mat/rate_factor.jl, e.g.
    # `_RF_GB_A0_1 = 1.25671e-5 [1/yr / Pa^3]`), confirming ATT itself comes out per-year -- the
    # same convention as ux_b/uy_b/bmb above, not the Pa^-n s^-1 SI this model expects.
    A_visc = perYear2perSecond.(mean(reshape(ds["ATT"][:], Nx, Ny, :), dims = 3)[:, :, 1])
    # bmb ("Combined basal mass balance") also carries a "units" = "m/yr" attribute, and is an
    # ice-equivalent thickness rate: Yelmo.jl's own definition (src/thrm/helpers.jl) is
    # `bmb = -Q_net / (rho_ice * L_ice)`, i.e. a heat flux divided by rho_ice (not rho_w) -- the
    # inverse of how Bmelt is handled in load_Kazmierczak above, which is already a water-equivalent
    # rate divided by rho_w. Negative bmb is mass loss (melting), hence the sign flip to get a melt
    # rate; rho_ice = 917.0 matches Yelmo.jl's own default rho_ice constant (YelmoConst.jl).
    ṁ = perYear2perSecond.(reshape(-ds["bmb"][:], Nx, Ny)) .* 917.0
    # Q_geo/Q_ice_b are already instantaneous heat fluxes (not ice-equivalent thickness rates like
    # bmb above), so unlike ux_b/ATT/bmb they need no per-year conversion -- only Q_geo's stated
    # "mW m^-2" units need scaling to the W/m^2 this model works in.
    G = reshape(ds["Q_geo"][:], Nx, Ny) ./ 1000.0
    q_T = reshape(ds["Q_ice_b"][:], Nx, Ny)

    κ = initialize_κ(Nx, Ny, b; bed_rheology)

    return Nx, Ny, xlims, ylims, mask, h, b, abs_v_b, A_visc, G, q_T, ṁ, κ

end

