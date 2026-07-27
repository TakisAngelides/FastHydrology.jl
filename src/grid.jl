"""
$(TYPEDSIGNATURES)

Abstract type for the grid of a simulation.
"""
abstract type AbstractHydroGrid end


"""
$(TYPEDSIGNATURES)

A struct for the Oceananigans Rectilinear grid. `Nx`, `Ny`, `dx`, `dy` are cached at construction
time (grid geometry never changes afterwards) so callers can read them directly, e.g. `grid.dx`,
rather than going through an accessor function.
"""
struct OGRectHydroGrid{T} <: AbstractHydroGrid
    grid::Oceananigans.RectilinearGrid
    Nx::Int
    Ny::Int
    dx::T
    dy::T
end


"""
$(TYPEDSIGNATURES)

The constructor to the struct for the Oceananigans RectilinearGrid.

# Arguments

- `Nx::I`: number of grid cells in the x direction.
- `Ny::I`: 
- `xlims`: tuple specifying the x values of the left-most and right-most edges of the grid in the x direction (e.g. xlims = (0, 1)).
- `ylims`: 

# Keywords

- `T`: type for the physical fields to live on the grid.
    (**Default**: `Float64`)
- `topology`: This specifies the boundary conditions for each of the x, y, z dimensions of the rectilinear grid.
    (**Default**: `(Bounded, Bounded, Flat)`)
- `halo`: tuple specifying the number of halo points in the x, y, z dimensions (e.g. when the z is flat i.e. the fields are not changing in that dimension, then halo = (1, 1) gives one ghost point for x and y to handle boundary conditions)
"""
function OGRectHydroGrid(Nx::I, Ny::I, xlims, ylims; T = Float64, topology = (Bounded, Bounded, Flat), halo = (1, 1)) where {I <: Integer}

    Nx > 0 || throw(ArgumentError("Nx must be positive"))
    Ny > 0 || throw(ArgumentError("Ny must be positive"))

    grid = Oceananigans.RectilinearGrid(T; size = (Nx, Ny), x = xlims, y = ylims, topology = topology, halo = halo)

    return OGRectHydroGrid(grid, grid.Nx, grid.Ny, grid.Δxᶜᵃᵃ, grid.Δyᵃᶜᵃ)
end



# ──────────────────────────────────────────────────────────────────────────────
# Grid interface
#
# Every concrete grid subtype must expose Nx, Ny, dx, dy as fields (see OGRectHydroGrid) so that
# generic physics/constructor code can read grid.Nx, grid.dx, etc. directly instead of going
# through an accessor function; the float element type is likewise read directly via
# typeof(grid.dx) rather than a dedicated function.
# ──────────────────────────────────────────────────────────────────────────────

"""
$(TYPEDSIGNATURES)

Allocate a scalar cell-centred field on `grid`, initialised to zero.
This is the only place that knows about the underlying field type (e.g. Oceananigans `CenterField`).
"""
alloc_field(grid::AbstractHydroGrid) = error("alloc_field not implemented for $(typeof(grid))")

"""
$(TYPEDSIGNATURES)

Allocate a scalar cell-centred field on `grid` and initialise it from `data`.
"""
alloc_field(grid::AbstractHydroGrid, data) = error("alloc_field not implemented for $(typeof(grid))")

"""
$(TYPEDSIGNATURES)

Fill ghost/halo points of `field` according to the boundary conditions encoded in `grid`.
"""
function fill_halo!(field, grid::AbstractHydroGrid)
    error("fill_halo! not implemented for $(typeof(grid))")
end


"""
$(TYPEDSIGNATURES)

Convolve `src` with `kernel` and write the result into `dest`, both cell-centered fields on `grid`.

Unlike the other grid-interface functions above, this one has a real default implementation
rather than an `error` stub: it assumes `dest`/`src` already behave like plain arrays, which holds
for most simple grid backends. Override it, as done below for `OGRectHydroGrid`, only for grid
backends whose fields wrap a different underlying array storage (so that `imfilter!`, which knows
nothing about that wrapper, can be handed the raw array instead).
"""
function convolve!(grid::AbstractHydroGrid, dest, src, kernel)
    imfilter!(dest, src, centered(kernel))
    return nothing
end


"""
$(TYPEDSIGNATURES)

Return the mean of `field` restricted to cells where `mask` equals 1.

As with `convolve!`, the default here assumes `field` already behaves like a plain array; override
it, as done below for `OGRectHydroGrid`, for grid backends whose fields wrap a different underlying
array storage.
"""
function masked_mean(grid::AbstractHydroGrid, field, mask)
    return mean(@views field[mask .== 1])
end


"""
$(TYPEDSIGNATURES)

The actual masked-overwrite logic, shared by every `overwrite_where!` grid method below: given
plain arrays (or a scalar for `src`), overwrite `dest` wherever `predicate(cond)` holds with
`scale .* src` (restricted to the same cells if `src` is array-like).
"""
function _overwrite_where!(dest, cond, predicate, src, scale)
    sel = predicate.(cond)
    if src isa Number
        @views dest[sel] .= scale * src
    else
        @views dest[sel] .= scale .* src[sel]
    end
    return nothing
end


"""
$(TYPEDSIGNATURES)

Overwrite cells of `dest` for which `predicate(cond)` holds with `scale .* src`, where `dest` and
`cond` are cell-centered fields on `grid`, `predicate` is a one-argument function (e.g. `==(0.0)`),
and `src` is either a scalar or another cell-centered field on `grid`.

As with `convolve!` and `masked_mean`, the default here assumes fields already behave like plain
arrays; override it, as done below for `OGRectHydroGrid`, for grid backends whose fields wrap a
different underlying array storage.
"""
function overwrite_where!(grid::AbstractHydroGrid, dest, cond, predicate, src; scale = true)
    _overwrite_where!(dest, cond, predicate, src, scale)
end


# ──────────────────────────────────────────────────────────────────────────────
# OGRectHydroGrid implementation of the grid interface
# ──────────────────────────────────────────────────────────────────────────────

alloc_field(g::OGRectHydroGrid)       = set!(CenterField(g.grid), 0.0)
alloc_field(g::OGRectHydroGrid, data) = set!(CenterField(g.grid), data)

function fill_halo!(field, ::OGRectHydroGrid)
    fill_halo_regions!(field)
end

function masked_mean(grid::OGRectHydroGrid, field, mask)
    return mean(@views interior(field, :, :, 1)[mask .== 1])
end

function overwrite_where!(grid::OGRectHydroGrid, dest, cond, predicate, src; scale = true)
    src_arr = src isa Number ? src : interior(src, :, :, 1)
    _overwrite_where!(interior(dest, :, :, 1), interior(cond, :, :, 1), predicate, src_arr, scale)
end

function convolve!(grid::OGRectHydroGrid, dest, src, kernel)
    imfilter!(dest.data, src.data, centered(kernel))
    return nothing
end