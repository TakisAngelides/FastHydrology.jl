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

Return the list of `CartesianIndex`es where `mask` equals 1, for use with `masked_mean`/
`masked_max_abs`/`masked_max_abs_diff` below. Callers that call those functions repeatedly on the
*same* mask within one scope (e.g. `resolve_q!`'s Picard/coupling loop in water_flux.jl, which calls
them up to `max_dissipation_iters`/`max_coupling_iters` times per solve) should compute this once as
a local variable and reuse it, rather than letting each call recompute `mask .== 1` from scratch --
that recomputation is real, measured cost (confirmed: cutting it saves ~9x memory and ~35% time on a
realistic `OGRectHydroGrid` solve). Deliberately NOT cached on `HydroState`/`KazmierczakParams`
across separate `resolve_q!` calls: the grounded-cell count changes as a coupled ice-flow model's
mask evolves timestep to timestep, so a persistent cache would need active invalidation to avoid
going stale, and -- since its length varies as the grounded-cell count does, unlike every other
field on `HydroState`, which are all fixed `(Nx, Ny)` arrays whose contents can be updated in place
with zero extra allocation -- caching it would reintroduce exactly the kind of avoidable GC pressure
this exists to eliminate, just at a per-timestep rather than per-iteration frequency. Computing it
fresh once per `resolve_q!` call keeps all of the benefit within one solve with none of that risk.
"""
grounded_indices(grid::AbstractHydroGrid, mask) = findall(==(1), mask)

"""
$(TYPEDSIGNATURES)

Return the mean of `field` restricted to grounded cells, given `idx` (see `grounded_indices`
above).

As with `convolve!`, the default here assumes `field` already behaves like a plain array; override
it, as done below for `OGRectHydroGrid`, for grid backends whose fields wrap a different underlying
array storage.
"""
function masked_mean(grid::AbstractHydroGrid, field, idx)
    return mean(@views field[idx])
end


"""
$(TYPEDSIGNATURES)

Return the maximum of `abs(a[i] - b[i])` over grounded cells `i` (see `grounded_indices`'s
docstring for `idx`). Used to check convergence of fixed-point iterations (e.g. the
dissipation-melt Picard loop in `update_q!`) without assuming `a`/`b` behave like plain arrays
outside the masked reduction itself. Written as `maximum(f, idx)`, not
`maximum(abs.(a[idx] .- b[idx]))`, so the per-cell `abs` fuses into the reduction instead of
materializing an intermediate array.
"""
function masked_max_abs_diff(grid::AbstractHydroGrid, a, b, idx)
    return maximum(i -> abs(a[i] - b[i]), idx)
end


"""
$(TYPEDSIGNATURES)

Return the maximum of `abs(field[i])` over grounded cells `i` (see `grounded_indices`'s docstring
for `idx`). Paired with `masked_max_abs_diff` to build a relative convergence tolerance. Fuses the
`abs` into the reduction rather than allocating an intermediate array, same reasoning as
`masked_max_abs_diff` above.
"""
function masked_max_abs(grid::AbstractHydroGrid, field, idx)
    return maximum(i -> abs(field[i]), idx)
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

grounded_indices(grid::OGRectHydroGrid, mask) = findall(==(1), interior(mask, :, :, 1))

function masked_mean(grid::OGRectHydroGrid, field, idx)
    return mean(@views interior(field, :, :, 1)[idx])
end

function masked_max_abs_diff(grid::OGRectHydroGrid, a, b, idx)
    ai = interior(a, :, :, 1)
    bi = interior(b, :, :, 1)
    return maximum(i -> abs(ai[i] - bi[i]), idx)
end

function masked_max_abs(grid::OGRectHydroGrid, field, idx)
    fi = interior(field, :, :, 1)
    return maximum(i -> abs(fi[i]), idx)
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