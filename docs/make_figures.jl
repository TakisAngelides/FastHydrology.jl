# Makes the figures of the example pages (docs/src/figures). Not part of the docs build, the input
# data is too big to ship. Run it by hand when the figures need to change:
#
#   julia -t 4 --project=test/Helheim docs/make_figures.jl [key ...]     (no key = all)
#
# Every run is K24 and HAB with the default settings and coupling_length_kamb86 = 0. Ice sheets (Yelmo
# restarts): q_T = 0 (Yelmo's Q_ice_b is not used, it cancels the basal melt) and friction heat from
# Yelmo's tau_b.

using FastHydrology
using CairoMakie
using Statistics: quantile
using MAT, JLD2

const ROOT   = dirname(@__DIR__)
const FIGS   = mkpath(joinpath(@__DIR__, "src", "figures"))

# Yelmo restarts: YELMO_RUNS = [(key, title, restart file), ...], defined in the local, git-ignored figure_paths.jl
include(joinpath(@__DIR__, "figure_paths.jl"))

# ------------------------------------------------------------------------------------------------
# Helheim: ISSM mesh -> regular grid, with the mask of Shakti.jl's Helheim example: outside the mesh
# OTHER_BASIN (closed), the mesh's boundary cells near the terminus OCEAN (the outlet).
function read_exp_points(path)
    lines = readlines(path)
    k = findfirst(l -> startswith(l, "# X pos"), lines)
    pts = [parse.(Float64, split(l)) for l in lines[k+1:end] if !isempty(strip(l))]
    return first.(pts), getindex.(pts, 2)
end

function point_in_polygon(px, py, poly_x, poly_y)
    inside, j = false, length(poly_x)
    for i in eachindex(poly_x)
        xi, yi, xj, yj = poly_x[i], poly_y[i], poly_x[j], poly_y[j]
        ((yi > py) != (yj > py)) && px < (xj - xi) * (py - yi) / (yj - yi) + xi && (inside = !inside)
        j = i
    end
    return inside
end

function load_helheim(; DX = 500.0)
    d  = matread(joinpath(ROOT, "test/Helheim/input/Helheim/Helheim_winter_A0_1d.mat"))
    md = d["md"].d
    ms = md["mesh"].d
    xs, ys = vec(ms["x"]), vec(ms["y"])
    el = Int.(ms["elements"])
    bed, thick = vec(md["geometry"].d["bed"]), vec(md["geometry"].d["thickness"])
    G_in = vec(md["basalforcings"].d["geothermalflux"])
    vx, vy = vec(md["initialization"].d["vx"]), vec(md["initialization"].d["vy"])
    A_in = vec(md["materials"].d["rheology_B"]) .^ -3.0
    x0, x1 = extrema(xs); y0, y1 = extrema(ys)
    Nx, Ny = floor(Int, (x1 - x0) / DX) + 1, floor(Int, (y1 - y0) / DX) + 1
    xc = [x0 + (i - 1) * DX for i in 1:Nx]
    yc = [y0 + (j - 1) * DX for j in 1:Ny]
    h, b, A, G, VX, VY = (zeros(Nx, Ny) for _ in 1:6)
    mask = fill(OTHER_BASIN, Nx, Ny)
    for t in axes(el, 1)
        i1, i2, i3 = el[t, :]
        xa, ya, xb, yb, xd, yd = xs[i1], ys[i1], xs[i2], ys[i2], xs[i3], ys[i3]
        den = (yb - yd) * (xa - xd) + (xd - xb) * (ya - yd)
        abs(den) < 1e-9 && continue
        ilo = max(1, floor(Int, (min(xa, xb, xd) - x0) / DX) + 1); ihi = min(Nx, ceil(Int, (max(xa, xb, xd) - x0) / DX) + 1)
        jlo = max(1, floor(Int, (min(ya, yb, yd) - y0) / DX) + 1); jhi = min(Ny, ceil(Int, (max(ya, yb, yd) - y0) / DX) + 1)
        for j in jlo:jhi, i in ilo:ihi
            px, py = xc[i], yc[j]
            w1 = ((yb - yd) * (px - xd) + (xd - xb) * (py - yd)) / den
            w2 = ((yd - ya) * (px - xd) + (xa - xd) * (py - yd)) / den
            w3 = 1 - w1 - w2
            (w1 >= -1e-9 && w2 >= -1e-9 && w3 >= -1e-9) || continue
            h[i, j]  = w1 * thick[i1] + w2 * thick[i2] + w3 * thick[i3]
            b[i, j]  = w1 * bed[i1] + w2 * bed[i2] + w3 * bed[i3]
            A[i, j]  = w1 * A_in[i1] + w2 * A_in[i2] + w3 * A_in[i3]
            G[i, j]  = w1 * G_in[i1] + w2 * G_in[i2] + w3 * G_in[i3]
            VX[i, j] = w1 * vx[i1] + w2 * vx[i2] + w3 * vx[i3]
            VY[i, j] = w1 * vy[i1] + w2 * vy[i2] + w3 * vy[i3]
            mask[i, j] = GROUNDED
        end
    end
    # the terminus: boundary cells within 1.5 cells of a mesh boundary vertex inside terminus13.exp
    tx, ty = read_exp_points(joinpath(ROOT, "test/Helheim/input/Helheim/terminus13.exp"))
    on_boundary = vec(ms["vertexonboundary"]) .> 0.5
    term = [v for v in eachindex(xs) if on_boundary[v] && point_in_polygon(xs[v], ys[v], tx, ty)]
    is_edge(i, j) = i == 1 || i == Nx || j == 1 || j == Ny ||
                    OTHER_BASIN in (mask[i - 1, j], mask[i + 1, j], mask[i, j - 1], mask[i, j + 1])
    ocean = [mask[i, j] == GROUNDED && is_edge(i, j) && any(v -> hypot(xc[i] - xs[v], yc[j] - ys[v]) <= 1.5DX, term)
             for i in 1:Nx, j in 1:Ny]
    mask[ocean] .= OCEAN
    xlims, ylims = compute_lims(xc, yc)
    return Nx, Ny, xlims, ylims, HydroMask(mask), h, b, perYear2perSecond.(sqrt.(VX .^ 2 .+ VY .^ 2)), A, G, zeros(Nx, Ny), zeros(Nx, Ny)
end

# Drang Drung: Shakti.jl's regular grids and mask (the same codes; its frozen bed, 4, is closed here,
# and its grounded cells without ice are land), no velocity data
function load_drangdrung(dx)
    f = jldopen(joinpath(dirname(ROOT), "Shakti.jl/test/drangdrung/data_v2/DRANGDRUNG$(dx)m_v2.jld2"), "r")
    h, b, A, G, m, xc, yc = f["H"], f["B"], f["A"], f["G"], f["mask"], f["xc"], f["yc"]
    close(f)
    Nx, Ny = size(h)
    xlims, ylims = compute_lims(xc, yc)
    mask = HydroMask(map((c, hc) -> c == 4 ? OTHER_BASIN : (c == GROUNDED && hc <= 0) ? LAND : Float64(c), m, h))
    return Nx, Ny, xlims, ylims, mask, h, b, zeros(Nx, Ny), A, G, zeros(Nx, Ny), zeros(Nx, Ny)
end

load_kazmierczak_thwaites() = load_Kazmierczak(joinpath(ROOT, "test/Kazmierczak et al 2024/input/Kazmierczak2024/THWAITES2km_m3_HAB_toto.mat"))
drop_tau_b(t) = (t[1:11]..., t[13])    # load_yelmox returns tau_b before kappa

# ------------------------------------------------------------------------------------------------
function run_case(key, title, (Nx, Ny, xlims, ylims, mask, h, b, abs_v_b, A_visc, G, q_T, kappa); transpose_data, ice_sheet = false, tau_b = nothing, q_pct = 0.99, extra_kamb = nothing)
    grid  = ArrayHydroGrid(Nx, Ny, xlims, ylims)
    grounded = mask .== GROUNDED
    # the default 100_000 of max_psi_out_calls is below the cell count of the finest grids
    sliding_law = tau_b === nothing ? NoFrictionSlidingLaw() : PrescribedFieldSlidingLaw(grid, tau_b)
    build(; kw...) = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, G, q_T;
                                           coupling_length_kamb86 = 0.0, dissipation_verbose = false,
                                           sliding_law, max_psi_out_calls = 1_000_000, kw...)

    state = HydroState(grid, mask, h, b)
    model = build()
    sim = SteadyStateSimulation(model, grid, state)
    run!(sim)                      # compile
    t = @elapsed run!(sim)

    state_hab = HydroState(grid, mask, h, b)
    run!(SteadyStateSimulation(HABHydroModel(grid), grid, state_hab))

    # ice sheets and Thwaites: q in m2/s, magma. Helheim and Drang Drung: q in 1e4 m2/a, viridis
    q_scale, q_unit, q_cmap = ice_sheet ? (1.0, "m² s⁻¹", :magma) : (perSecond2perYear(1.0) / 1e4, "10⁴ m² a⁻¹", :viridis)
    cr(x, p) = (0.0, quantile(x[grounded .& isfinite.(x)], p))
    q_range, N_range = cr(model.q .* q_scale, q_pct), cr(state.N ./ 1e6, 0.99)

    plot(file, field, ttl, range, cmap) = visualize_field(grid, mask_field(field, mask, NaN);
        plot_title = ttl, colormap = cmap, colorrange = range, display_flag = false,
        transpose_data = transpose_data, savefig_path = joinpath(FIGS, "$(file).png"))
    plot("$(key)_q",   model.q .* q_scale, "$title: water flux q [$q_unit]", q_range, q_cmap)
    plot("$(key)_N",   state.N ./ 1e6, "$title: effective pressure N [MPa]", N_range, :viridis)
    plot("$(key)_HAB", state_hab.N ./ 1e6, "$title: HAB effective pressure N [MPa]", N_range, :viridis)
    if extra_kamb !== nothing      # same case with the stress coupling switched on, q only
        state_k = HydroState(grid, mask, h, b)
        model_k = build(coupling_length_kamb86 = extra_kamb)
        run!(SteadyStateSimulation(model_k, grid, state_k))
        plot("$(key)_kamb$(Int(extra_kamb))_q", model_k.q .* q_scale, "$title, coupling length $(Int(extra_kamb)) ice thicknesses: water flux q [$q_unit]", q_range, q_cmap)
    end
    println(rpad(key, 12), " grid $(Nx) x $(Ny)  grounded $(count(grounded))  K24 solve $(round(1e3 * t; digits = 1)) ms")

end

keys_wanted = isempty(ARGS) ? nothing : Set(ARGS)
want(k) = keys_wanted === nothing || k in keys_wanted

for (key, title, path) in YELMO_RUNS
    want(key) || continue
    t = load_yelmox(path)
    run_case(key, title, (t[1:10]..., zeros(size(t[11])), t[13]); transpose_data = false, ice_sheet = true, tau_b = t[12], q_pct = 0.90)   # the ice-sheet q colours saturate at the 90th percentile
end
if want("Thwaites")    # q_T from the source model, friction heat from its tau_b
    t = load_kazmierczak_thwaites()
    run_case("Thwaites", "Thwaites 2 km", drop_tau_b(t); transpose_data = true, ice_sheet = true, tau_b = t[12], extra_kamb = 10.0)
end
want("Helheim") && run_case("Helheim", "Helheim", load_helheim(); transpose_data = false)
for dx in (100, 50, 25)
    want("DD_$(dx)m") && run_case("DD_$(dx)m", "Drang Drung $(dx) m", load_drangdrung(dx); transpose_data = false)
end
