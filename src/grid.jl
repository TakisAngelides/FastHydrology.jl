"""
$(TYPEDSIGNATURES)

Abstract type for the grid of a simulation.
"""
abstract type AbstractHydroGrid end


"""
$(TYPEDSIGNATURES)

A struct for the Oceananigans Rectilinear grid.
"""
struct OGRectHydroGrid <: AbstractHydroGrid
    grid::Oceananigans.RectilinearGrid
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

    return OGRectHydroGrid(grid)
end



# ──────────────────────────────────────────────────────────────────────────────
# Grid interface
#
# Every concrete grid subtype must implement these six functions.
# Physics and constructors only call these; they never reach into `grid.grid.*`.
# ──────────────────────────────────────────────────────────────────────────────

"""$(TYPEDSIGNATURES) Number of interior cells in x."""
grid_Nx(grid::AbstractHydroGrid) = error("grid_Nx not implemented for $(typeof(grid))")

"""$(TYPEDSIGNATURES) Number of interior cells in y."""
grid_Ny(grid::AbstractHydroGrid) = error("grid_Ny not implemented for $(typeof(grid))")

"""$(TYPEDSIGNATURES) Uniform cell width in x [m]."""
grid_dx(grid::AbstractHydroGrid) = error("grid_dx not implemented for $(typeof(grid))")

"""$(TYPEDSIGNATURES) Uniform cell width in y [m]."""
grid_dy(grid::AbstractHydroGrid) = error("grid_dy not implemented for $(typeof(grid))")

"""$(TYPEDSIGNATURES) Floating-point element type used on this grid."""
grid_eltype(grid::AbstractHydroGrid) = error("grid_eltype not implemented for $(typeof(grid))")

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


# ──────────────────────────────────────────────────────────────────────────────
# OGRectHydroGrid implementation of the grid interface
# ──────────────────────────────────────────────────────────────────────────────

grid_Nx(g::OGRectHydroGrid)      = g.grid.Nx
grid_Ny(g::OGRectHydroGrid)      = g.grid.Ny
grid_dx(g::OGRectHydroGrid)      = g.grid.Δxᶜᵃᵃ
grid_dy(g::OGRectHydroGrid)      = g.grid.Δyᵃᶜᵃ
grid_eltype(g::OGRectHydroGrid)  = eltype(g.grid)

alloc_field(g::OGRectHydroGrid)        = set!(CenterField(g.grid), 0.0)
alloc_field(g::OGRectHydroGrid, data)  = set!(CenterField(g.grid), data)

function fill_halo!(field, ::OGRectHydroGrid)
    fill_halo_regions!(field)
end
