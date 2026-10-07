"""
Package extension activated automatically once both `FastHydrology` and a Makie backend (e.g.
`CairoMakie`) are loaded (`using FastHydrology, CairoMakie`). Adds the `visualize_field`/
`visualize_grid` methods declared as stubs in `src/common/plotting.jl`.

Kept as an extension (rather than a hard `using CairoMakie` in `src/FastHydrology.jl`) because
CairoMakie is one of the slower-compiling packages in the ecosystem and plotting is not needed for
running a simulation -- users who never call `visualize_field`/`visualize_grid` pay nothing for it.
"""
module FastHydrologyMakieExt

using FastHydrology
using FastHydrology: AbstractHydroGrid
using CairoMakie
using CairoMakie: Reverse

# Cell-centre coordinates of `grid`, measured from its lower-left corner (grids only store
# `Nx`/`Ny`/`dx`/`dy`, not an absolute origin).
_xcenters(grid::AbstractHydroGrid) = ((1:grid.Nx) .- 0.5) .* grid.dx
_ycenters(grid::AbstractHydroGrid) = ((1:grid.Ny) .- 0.5) .* grid.dy

"""
    visualize_field(x, y, data; kwargs...)

Visualize a scalar field `data` on coordinates `x`, `y`.
"""
function FastHydrology.visualize_field(x, y, data;
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
    visualize_field(grid::AbstractHydroGrid, field; kwargs...)

Visualize a cell-centred `field` on `grid`, with axes in grid coordinates measured from the grid's
lower-left corner.
"""
function FastHydrology.visualize_field(grid::AbstractHydroGrid, field; kwargs...)

    FastHydrology.visualize_field(_xcenters(grid), _ycenters(grid), field; kwargs...)

end

"""
    visualize_grid(grid::AbstractHydroGrid)

Visualize the corners of `grid`, showing cell centers and boundaries. Works for any grid type that
exposes the `Nx`, `Ny`, `dx`, `dy` fields of the grid interface; coordinates are measured from the
grid's lower-left corner.
"""
function FastHydrology.visualize_grid(grid::AbstractHydroGrid)

    xc = _xcenters(grid)
    yc = _ycenters(grid)
    dx = fill(grid.dx, grid.Nx)
    dy = fill(grid.dy, grid.Ny)

    Nx = grid.Nx
    Ny = grid.Ny

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

end
