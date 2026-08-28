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

`conv_cache` holds FFT plans/buffers for `convolve!` (see fft_convolution.jl), same role and same
lazy-build-on-first-use behaviour as `OGRectHydroGrid`'s own `conv_cache` -- see its docstring.
`cached_fft_convolve!` is grid-agnostic (its `dest`/`src`/`kernel` arguments are plain
`AbstractMatrix`, nothing Oceananigans-specific), so `ArrayHydroGrid` reuses it directly instead of
falling back to the generic `convolve!` default below (which calls `ImageFiltering.imfilter!` --
confirmed by profiling to allocate its FFT plans and padded/complex buffers fresh on every call,
~20 MB on a realistic grid, since it exposes no hook to reuse them across the repeated calls one
coupled simulation makes).

Needs no other overrides of the grid interface beyond `alloc_field`: `fill_halo!` is a no-op
(fields carry no separate halo storage), and the default plain-array implementations of
`masked_mean`, `minus_gradient_x!`/`minus_gradient_y!`, etc. already operate correctly on the
`Array`s this type allocates.
"""
struct ArrayHydroGrid{T <: AbstractFloat} <: AbstractHydroGrid
    Nx::Int
    Ny::Int
    dx::T
    dy::T
    conv_cache::Base.RefValue{Any}
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

    return ArrayHydroGrid(Nx, Ny, dx, dy, Ref{Any}(nothing))
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
for most simple grid backends. This default calls `ImageFiltering.imfilter!` directly, which
allocates fresh FFT plans and padded/complex buffers on every call (no caching hook available) --
fine for a grid backend that only convolves occasionally, but both concrete grid types this package
ships (`ArrayHydroGrid`, `OGRectHydroGrid`) override this instead with a `conv_cache`-backed call to
`cached_fft_convolve!` (fft_convolution.jl), since `update_smoothed_potential_gradients!` calls this
every solve of a coupled simulation. A new grid backend whose fields behave like plain arrays can
either accept this default (if convolution is rare for it) or add its own `conv_cache` field and
override, following the same pattern.
"""
function convolve!(grid::AbstractHydroGrid, dest, src, kernel)
    imfilter!(dest, src, centered(kernel))
    return nothing
end

function convolve!(grid::ArrayHydroGrid, dest, src, kernel)
    cached_fft_convolve!(grid.conv_cache, dest, src, kernel)
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

Return the mean of `field` restricted to cells where `mask == 1`. Fuses the mask check into the
reduction (branchless, via `ifelse`, so it vectorizes) instead of materializing a list of grounded
indices first -- benchmarked faster than the earlier `findall(==(1), mask)` + index-list approach
at every grounded-fraction tested (1%-95%), including when amortized over the several masked_*
calls per Picard/coupling iteration in `resolve_q!` (water_flux.jl): `findall`'s vector-growth cost
and the scattered (non-sequential) memory access of indexing through a `CartesianIndex` list both
lose to a single dense, branchless pass over `field`/`mask` together, even though the dense pass
touches every cell rather than just the grounded ones.

As with `convolve!`, the default here assumes `field`/`mask` already behave like plain arrays;
override it, as done below for `OGRectHydroGrid`, for grid backends whose fields wrap a different
underlying array storage.
"""
function masked_mean(grid::AbstractHydroGrid, field, mask)
    s = zero(eltype(field))
    cnt = 0
    @inbounds @simd for i in eachindex(field, mask)
        m = mask[i] == 1
        s   += ifelse(m, field[i], zero(eltype(field)))
        cnt += ifelse(m, 1, 0)
    end
    return s / cnt
end


"""
$(TYPEDSIGNATURES)

Return the maximum of `abs(a[i] - b[i])` over cells where `mask == 1`. Used to check convergence of
fixed-point iterations (e.g. the dissipation-melt Picard loop in `update_q!`). Same fused,
branchless reduction as `masked_mean` above -- see its docstring for why this beats precomputing a
grounded-index list.
"""
function masked_max_abs_diff(grid::AbstractHydroGrid, a, b, mask)
    best = zero(eltype(a))
    @inbounds @simd for i in eachindex(a, b, mask)
        m = mask[i] == 1
        v = ifelse(m, abs(a[i] - b[i]), zero(eltype(a)))
        best = ifelse(v > best, v, best)
    end
    return best
end


"""
$(TYPEDSIGNATURES)

Return the maximum of `abs(field[i])` over cells where `mask == 1`. Paired with
`masked_max_abs_diff` to build a relative convergence tolerance. Same fused, branchless reduction
as `masked_mean` above -- see its docstring for why this beats precomputing a grounded-index list.
"""
function masked_max_abs(grid::AbstractHydroGrid, field, mask)
    best = zero(eltype(field))
    @inbounds @simd for i in eachindex(field, mask)
        m = mask[i] == 1
        v = ifelse(m, abs(field[i]), zero(eltype(field)))
        best = ifelse(v > best, v, best)
    end
    return best
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


# `field[i, j, 1]` on an Oceananigans `Field` is already halo-aware -- it returns exactly the same
# value as `interior(field, :, :, 1)[i, j]`, since the field's underlying storage is an OffsetArray
# whose index 1 lands on the first interior cell (confirmed: for a (Nx,Ny) field with any halo size,
# direct (i,j,1) indexing over i in 1:Nx, j in 1:Ny reproduces `interior(...)` exactly). So the loop
# below skips `interior(...)` entirely rather than paying for it up front and indexing the result.
#
# It also avoids `eachindex(interior(field,:,:,1), interior(mask,:,:,1))` on principle, not just to
# skip the `interior` call: `interior(...)` returns a `SubArray` that does not support fast linear
# indexing, so `eachindex` on it returns `CartesianIndices`, and `@simd for i in <CartesianIndices>`
# falls through to a generic, non-specializing iteration path -- confirmed by profiling to fully
# lose type inference (every intermediate boxed to `Any`) and allocate ~6 times per cell instead of
# running allocation-free. A plain nested `for j in 1:Ny; @simd for i in 1:Nx` loop with direct
# (i,j,1) indexing sidesteps that trap entirely.
function masked_mean(grid::OGRectHydroGrid, field, mask)
    Nx, Ny = grid.Nx, grid.Ny
    s = zero(eltype(field))
    cnt = 0
    @inbounds for j in 1:Ny
        @simd for i in 1:Nx
            m = mask[i, j, 1] == 1
            s   += ifelse(m, field[i, j, 1], zero(eltype(field)))
            cnt += ifelse(m, 1, 0)
        end
    end
    return s / cnt
end

function masked_max_abs_diff(grid::OGRectHydroGrid, a, b, mask)
    Nx, Ny = grid.Nx, grid.Ny
    best = zero(eltype(a))
    @inbounds for j in 1:Ny
        @simd for i in 1:Nx
            m = mask[i, j, 1] == 1
            v = ifelse(m, abs(a[i, j, 1] - b[i, j, 1]), zero(eltype(a)))
            best = ifelse(v > best, v, best)
        end
    end
    return best
end

function masked_max_abs(grid::OGRectHydroGrid, field, mask)
    Nx, Ny = grid.Nx, grid.Ny
    best = zero(eltype(field))
    @inbounds for j in 1:Ny
        @simd for i in 1:Nx
            m = mask[i, j, 1] == 1
            v = ifelse(m, abs(field[i, j, 1]), zero(eltype(field)))
            best = ifelse(v > best, v, best)
        end
    end
    return best
end

function overwrite_where!(grid::OGRectHydroGrid, dest, cond, predicate, src; scale = true)
    src_arr = src isa Number ? src : interior(src, :, :, 1)
    _overwrite_where!(interior(dest, :, :, 1), interior(cond, :, :, 1), predicate, src_arr, scale)
end

function convolve!(grid::OGRectHydroGrid, dest, src, kernel)
    # Must NOT pass dest.data/src.data directly: Field.data carries Oceananigans' own halo padding
    # around the (Nx, Ny) interior (e.g. (7,7,1) storage for a (5,5) grid with the default halo =
    # (1,1)), and cached_fft_convolve! derives its logical domain size from its src argument's own
    # shape -- handed .data directly, it would silently convolve over the inflated (Nx+2,Ny+2) region
    # (garbage/boundary-condition-filled halo cells included as if they were real data) and crop back
    # using that wrong size, corrupting every cell, not just the boundary ones. Confirmed by comparing
    # against a brute-force replicate-padded reference: passing .data diverged by up to several
    # hundred percent; interior(_, :, :, 1) -- which already gives a clean, correctly-(Nx,Ny)-sized
    # view -- matches to floating-point precision.
    #
    # Passed straight through with no intermediate copy: cached_fft_convolve! only ever reads src (via
    # a broadcast into its own cached padded buffer) and writes dest (via a broadcast out of its own
    # cached result buffer), so these views are exactly as good as a plain-Array copy would be for
    # that purpose, at zero extra allocation -- unlike an earlier version of this function, which
    # allocated a fresh (Nx,Ny,1) Array for each of src/dest on every call before finding this out.
    cached_fft_convolve!(grid.conv_cache, interior(dest, :, :, 1), interior(src, :, :, 1), kernel)
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