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

`conv_cache` holds FFT plans/buffers for `convolve!` (see fft_convolution.jl), reused across calls
since the same (image size, kernel size) combination repeats every timestep of a coupled
simulation. It's a `Ref` rather than a genuinely mutable struct field so `OGRectHydroGrid` itself
can stay immutable; its contents are lazily built on first use and rebuilt only if the kernel size
changes.
"""
struct OGRectHydroGrid{T} <: AbstractHydroGrid
    grid::Oceananigans.RectilinearGrid
    Nx::Int
    Ny::Int
    dx::T
    dy::T
    conv_cache::Base.RefValue{Any}
end


"""
$(TYPEDSIGNATURES)

The constructor to the struct for the Oceananigans RectilinearGrid.

# Arguments

- `Nx::I`: number of grid cells in the x direction.
- `Ny::I`: number of grid cells in the y direction.
- `xlims`: tuple specifying the x values of the left-most and right-most edges of the grid in the x direction (e.g. xlims = (0, 1)).
- `ylims`: tuple specifying the y values of the bottom-most and top-most edges of the grid in the y direction (e.g. ylims = (0, 1)).

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

    return OGRectHydroGrid(grid, grid.Nx, grid.Ny, grid.Δxᶜᵃᵃ, grid.Δyᵃᶜᵃ, Ref{Any}(nothing))
end


"""
$(TYPEDSIGNATURES)

A grid backed by plain `Array`s -- no Oceananigans dependency. `Nx`, `Ny`, `dx`, `dy` are cached at
construction time, same as `OGRectHydroGrid`, so generic code can read them directly.

Needs no overrides of the grid interface beyond `alloc_field`: `fill_halo!` is a no-op (fields carry
no separate halo storage), and the default plain-array implementations of `convolve!`,
`masked_mean`, `minus_gradient_x!`/`minus_gradient_y!`, etc. already operate correctly on the
`Array`s this type allocates.
"""
struct ArrayHydroGrid{T <: AbstractFloat} <: AbstractHydroGrid
    Nx::Int
    Ny::Int
    dx::T
    dy::T
end


"""
$(TYPEDSIGNATURES)

The constructor for `ArrayHydroGrid`.

# Arguments

- `Nx::I`: number of grid cells in the x direction.
- `Ny::I`: number of grid cells in the y direction.
- `xlims`: tuple specifying the x values of the left-most and right-most edges of the grid in the x direction (e.g. xlims = (0, 1)).
- `ylims`: tuple specifying the y values of the bottom-most and top-most edges of the grid in the y direction (e.g. ylims = (0, 1)).

# Keywords

- `T`: type for the physical fields to live on the grid.
    (**Default**: `Float64`)
"""
function ArrayHydroGrid(Nx::I, Ny::I, xlims, ylims; T = Float64) where {I <: Integer}

    Nx > 0 || throw(ArgumentError("Nx must be positive"))
    Ny > 0 || throw(ArgumentError("Ny must be positive"))

    dx = T(xlims[2] - xlims[1]) / Nx
    dy = T(ylims[2] - ylims[1]) / Ny

    return ArrayHydroGrid(Nx, Ny, dx, dy)
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

The default here is a no-op: plain arrays carry no explicit halo storage, and grid backends built
on them (unlike `OGRectHydroGrid`) are expected to embed boundary handling directly into their
operators instead (e.g. the edge-clamped stencil in `minus_gradient_x!`/`minus_gradient_y!`
below). Override this, as done below for `OGRectHydroGrid`, only for grid backends whose fields
carry real halo storage that needs to be kept in sync.
"""
function fill_halo!(field, grid::AbstractHydroGrid)
    return nothing
end


"""
$(TYPEDSIGNATURES)

Convolve `src` with `kernel` and write the result into `dest`, both cell-centered fields on `grid`.
`kernel` has the same number of dimensions as `dest`/`src` (i.e. 2D for a plain 2D field).

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

Write `-∂field/∂x` into `dest`, both cell-centered fields on `grid`.

The default here assumes `field`/`dest` already behave like plain arrays and computes a central
difference, clamping the neighbour index at the domain edges (i.e. edge-replicating) rather than
reading from an explicit halo. This matches Oceananigans' own default zero-flux boundary condition
at `Bounded` edges (verified empirically: with halo cells filled by `fill_halo_regions!`,
`OGRectHydroGrid`'s override below produces the same values as this formula would if given
edge-replicated ghost cells). Override this, as done below for `OGRectHydroGrid`, for grid
backends whose fields wrap a different underlying array storage.
"""
function minus_gradient_x!(grid::AbstractHydroGrid, dest, field)
    Nx, Ny = grid.Nx, grid.Ny
    dx = grid.dx
    @inbounds for j in 1:Ny, i in 1:Nx
        im1 = max(i - 1, 1)
        ip1 = min(i + 1, Nx)
        dest[i, j] = -(field[ip1, j] - field[im1, j]) / (2dx)
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Write `-∂field/∂y` into `dest`, both cell-centered fields on `grid`. See `minus_gradient_x!` for
details on the default (plain-array) implementation.
"""
function minus_gradient_y!(grid::AbstractHydroGrid, dest, field)
    Nx, Ny = grid.Nx, grid.Ny
    dy = grid.dy
    @inbounds for j in 1:Ny, i in 1:Nx
        jm1 = max(j - 1, 1)
        jp1 = min(j + 1, Ny)
        dest[i, j] = -(field[i, jp1] - field[i, jm1]) / (2dy)
    end
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

Return the maximum of `abs.(a .- b)`, restricted to cells where `mask` equals 1. Used to check
convergence of fixed-point iterations (e.g. the dissipation-melt Picard loop in `update_q!`)
without assuming `a`/`b` behave like plain arrays outside the masked reduction itself.
"""
function masked_max_abs_diff(grid::AbstractHydroGrid, a, b, mask)
    return maximum(abs.(@views a[mask .== 1] .- b[mask .== 1]))
end


"""
$(TYPEDSIGNATURES)

Return the maximum of `abs.(field)`, restricted to cells where `mask` equals 1. Paired with
`masked_max_abs_diff` to build a relative convergence tolerance.
"""
function masked_max_abs(grid::AbstractHydroGrid, field, mask)
    return maximum(abs.(@views field[mask .== 1]))
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

function minus_gradient_x!(::OGRectHydroGrid, dest, field)
    dest .= -∂x(field)
    return nothing
end

function minus_gradient_y!(::OGRectHydroGrid, dest, field)
    dest .= -∂y(field)
    return nothing
end

function masked_mean(grid::OGRectHydroGrid, field, mask)
    return mean(@views interior(field, :, :, 1)[mask .== 1])
end

function masked_max_abs_diff(grid::OGRectHydroGrid, a, b, mask)
    return maximum(abs.(@views interior(a, :, :, 1)[mask .== 1] .- interior(b, :, :, 1)[mask .== 1]))
end

function masked_max_abs(grid::OGRectHydroGrid, field, mask)
    return maximum(abs.(@views interior(field, :, :, 1)[mask .== 1]))
end

function overwrite_where!(grid::OGRectHydroGrid, dest, cond, predicate, src; scale = true)
    src_arr = src isa Number ? src : interior(src, :, :, 1)
    _overwrite_where!(interior(dest, :, :, 1), interior(cond, :, :, 1), predicate, src_arr, scale)
end

function convolve!(grid::OGRectHydroGrid, dest, src, kernel)
    # kernel arrives 2D (see the default convolve! docstring); cached_fft_convolve! indexes its
    # third argument with a trailing singleton dimension to match dest.data/src.data's (Nx,Ny,1)
    # Oceananigans layout, so reshape (no copy) rather than pushing the (Nx,Ny,1) shape onto callers.
    kernel3d = reshape(kernel, size(kernel, 1), size(kernel, 2), 1)
    cached_fft_convolve!(grid.conv_cache, dest.data, src.data, kernel3d)
    return nothing
end


# ──────────────────────────────────────────────────────────────────────────────
# ArrayHydroGrid implementation of the grid interface
#
# Only alloc_field is needed -- every other grid-interface function's default (above) already
# assumes plain arrays.
# ──────────────────────────────────────────────────────────────────────────────

alloc_field(g::ArrayHydroGrid{T}) where {T} = zeros(T, g.Nx, g.Ny)
alloc_field(g::ArrayHydroGrid{T}, data) where {T} = T.(reshape(data, g.Nx, g.Ny))