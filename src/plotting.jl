"""
$(TYPEDSIGNATURES)

Visualize a field.
"""
function visualize_field(x, y, data; 
        plot_title = "", 
        transpose_data = false, 
        colorrange = nothing,       # now auto-computed with fallback
        display_flag = true, 
        colormap = Reverse(:RdBu), 
        colorscale = identity,
        savefig_path = nothing
    )
    fig = Figure(size = (900, 700))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = plot_title, aspect = DataAspect())

    if transpose_data
        data = data'
    end

    # Robust colorrange: handles all-zero, all-NaN, and flat fields
    if colorrange === nothing
        finite_vals = filter(isfinite, vec(data))
        if isempty(finite_vals)
            colorrange = (0.0, 1.0)   # nothing to show, dummy range
        else
            lo, hi = extrema(finite_vals)
            if lo ≈ hi
                colorrange = iszero(lo) ? (-1.0, 1.0) : (lo * 0.9, lo * 1.1)
            else
                colorrange = (lo, hi)
            end
        end
    end

    hm = heatmap!(ax, x, y, data; colormap, colorrange, colorscale)
    Colorbar(fig[1, 2], hm)

    if display_flag
        display(fig)
    end
    if savefig_path !== nothing
        save(savefig_path, fig)
    end
    return fig
end


"""
$(TYPEDSIGNATURES)

Visualize an Oceananigans field.
"""
function visualize_field(field::Oceananigans.Fields.Field; kwargs...)

    data = interior(field)[:, :, 1]
    x, y = nodes(field)
    visualize_field(x, y, data; kwargs...)
    
end


"""
$(TYPEDSIGNATURES)

Set a field to a given input value where the mask is not 1.
"""
function mask_field(field::Oceananigans.Fields.Field, mask, value)

    Nx, Ny = size(mask)
    res = deepcopy(field)
    for j in 1:Ny
        for i in 1:Nx
            if mask[i, j] != 1.0
                res[i, j, 1] = value
            end
        end
    end

    return res
    
end


"""
$(TYPEDSIGNATURES)

Visualize the corners of a hydrology grid, showing cell centers and boundaries.
Works with any concrete subtype of AbstractHydroGrid via the grid interface.
For grid types other than OGRectHydroGrid, override this function to extract
x/y node positions and spacings appropriate to that grid.
"""
function visualize_grid(grid::OGRectHydroGrid)

    og = grid.grid  # Oceananigans-specific internals, kept local to this method

    xc = xnodes(og, Center())
    yc = ynodes(og, Center())
    dx = xspacings(og, Center())
    dy = yspacings(og, Center())

    Nx = grid_Nx(grid)
    Ny = grid_Ny(grid)

    quadrants = [
        (1:5, Ny-4:Ny),     # top-left
        (Nx-4:Nx, Ny-4:Ny), # top-right
        (1:5, 1:5),          # bottom-left
        (Nx-4:Nx, 1:5)       # bottom-right
    ]

    titles = ["top left grid corner", "top right grid corner", "bottom left grid corner", "bottom right grid corner"]

    fig = Figure(size=(800,800))
    
    Label(fig[0, 1:2], "Nx = $(Nx), Ny = $(Ny), dx = $(dx[1]), dy = $(dy[1])", halign = :center)
    
    for (idx, (xi, yi)) in enumerate(quadrants)

        ax = Axis(fig[div(idx-1,2)+1, mod(idx-1,2)+1]; xlabel="x", ylabel="y", title=titles[idx], xticklabelsize=10, yticklabelsize=10, xgridvisible = false, ygridvisible = false)

        scatter!(ax, repeat(xc[xi], inner=length(yc[yi])), repeat(yc[yi], outer=length(xc[xi])), color=:blue, markersize=4)
    
        for (i, x) in enumerate(xc[xi])
            for (j, y) in enumerate(yc[yi])
                xs = [x - dx[xi[i]]/2, x + dx[xi[i]]/2, x + dx[xi[i]]/2, x - dx[xi[i]]/2]
                ys = [y - dy[yi[j]]/2, y - dy[yi[j]]/2, y + dy[yi[j]]/2, y + dy[yi[j]]/2]
                poly!(ax, xs, ys; color=:transparent, strokewidth=0.5, strokecolor=:red)
            end
        end
    end

    return fig

end