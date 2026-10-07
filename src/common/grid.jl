"""
$(TYPEDSIGNATURES)

Abstract type for the grid of a simulation.
"""
abstract type AbstractHydroGrid end


"""
$(TYPEDSIGNATURES)

A grid backed by plain `Array`s. `Nx`, `Ny`, `dx`, `dy` are cached at construction time so generic
code can read them directly.

`conv_cache` holds FFT plans/buffers for `convolve!` (see fft_convolution.jl), reused across calls
since the same (image size, kernel size) combination repeats every timestep of a coupled
simulation. It's a `Ref` rather than a genuinely mutable struct field so `ArrayHydroGrid` itself
can stay immutable; its contents are lazily built on first use and rebuilt only if the kernel size
changes. `cached_fft_convolve!` takes plain `AbstractMatrix` arguments, so `ArrayHydroGrid` uses it
directly instead of falling back to the generic `convolve!` default below (which calls `ImageFiltering.imfilter!` --
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
# Every concrete grid subtype must expose Nx, Ny, dx, dy as fields (see ArrayHydroGrid) so that
# generic physics/constructor code can read grid.Nx, grid.dx, etc. directly instead of going
# through an accessor function; the float element type is likewise read directly via
# typeof(grid.dx) rather than a dedicated function.
# ──────────────────────────────────────────────────────────────────────────────

"""
$(TYPEDSIGNATURES)

Allocate a scalar cell-centred field on `grid`, initialised to zero.
This is the only place that knows about the underlying field type (e.g. `Array` for `ArrayHydroGrid`).
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
on them are expected to embed boundary handling directly into their operators instead (e.g. the
edge-clamped stencil in `minus_gradient_x!`/`minus_gradient_y!` below). Override this only for grid
backends whose fields carry real halo storage that needs to be kept in sync.
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
fine for a grid backend that only convolves occasionally, but `ArrayHydroGrid` overrides this instead with a `conv_cache`-backed call to
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

Write `-∂field/∂x` into the 2D array `dest` from the 2D array `field` (both `Nx × Ny`, spacing `dx`).
Shared by every grid backend's `minus_gradient_x!` so they cannot drift apart.

Interior cells use a centred difference `(f[i+1] - f[i-1]) / 2dx`. The two domain-edge columns use a
one-sided difference over the single available cell, `(f[2] - f[1]) / dx` at `i = 1` and
`(f[Nx] - f[Nx-1]) / dx` at `i = Nx`, i.e. the full first-order gradient. (Replicating the edge cell as
a ghost value instead and keeping the `2dx` denominator gives exactly half the gradient at the edge.)
A single-column domain (`Nx == 1`) has no gradient and gives 0.
"""
function minus_gradient_x_kernel!(dest, field, Nx, Ny, dx)
    @inbounds for j in 1:Ny, i in 1:Nx
        if Nx == 1
            dest[i, j] = zero(eltype(dest))
        elseif i == 1
            dest[i, j] = -(field[2, j] - field[1, j]) / dx
        elseif i == Nx
            dest[i, j] = -(field[Nx, j] - field[Nx - 1, j]) / dx
        else
            dest[i, j] = -(field[i + 1, j] - field[i - 1, j]) / (2dx)
        end
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

The `y` counterpart of [`minus_gradient_x_kernel!`](@ref): centred in the interior, one-sided
(full-gradient) at `j = 1` and `j = Ny`, 0 when `Ny == 1`.
"""
function minus_gradient_y_kernel!(dest, field, Nx, Ny, dy)
    @inbounds for j in 1:Ny, i in 1:Nx
        if Ny == 1
            dest[i, j] = zero(eltype(dest))
        elseif j == 1
            dest[i, j] = -(field[i, 2] - field[i, 1]) / dy
        elseif j == Ny
            dest[i, j] = -(field[i, Ny] - field[i, Ny - 1]) / dy
        else
            dest[i, j] = -(field[i, j + 1] - field[i, j - 1]) / (2dy)
        end
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Write `-∂field/∂x` into `dest`, both cell-centered fields on `grid`.

The default here assumes `field`/`dest` already behave like plain arrays and applies
[`minus_gradient_x_kernel!`](@ref): centred differences in the interior and one-sided differences at
the domain edges, so edge cells get the full gradient rather than a halved one. A grid backend whose
fields wrap a different storage (e.g. with halo padding) should override this only to hand the
kernel plain-array views, so every backend shares the same edge handling.
"""
function minus_gradient_x!(grid::AbstractHydroGrid, dest, field)
    minus_gradient_x_kernel!(dest, field, grid.Nx, grid.Ny, grid.dx)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Write `-∂field/∂y` into `dest`, both cell-centered fields on `grid`. See `minus_gradient_x!` for
details on the default (plain-array) implementation.
"""
function minus_gradient_y!(grid::AbstractHydroGrid, dest, field)
    minus_gradient_y_kernel!(dest, field, grid.Nx, grid.Ny, grid.dy)
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
override it for grid backends whose fields wrap a different
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

Overwrite cells of `dest` for which `predicate(cond)` holds with `scale * src`, where `dest` and
`cond` are cell-centered fields on `grid`, `predicate` is a one-argument function (e.g. `==(0.0)`),
and `src` is a scalar. Fused into a single branchless (`ifelse`) pass rather than building the
intermediate boolean mask array `predicate.(cond)` and assigning through it -- same reasoning as
`masked_mean` above: this runs inside `update_S_inf!`/`update_N_inf!` (effective_pressure.jl), which
`update_N!` calls up to `model.max_qN_iters` times per solve for N-dependent sliding laws, so
avoiding a fresh full-size allocation every call is real, measured cost there.

As with `convolve!` and `masked_mean`, the default here assumes fields already behave like plain
arrays; override it for grid backends whose fields wrap a
different underlying array storage.
"""
function overwrite_where!(grid::AbstractHydroGrid, dest, cond, predicate, src::Number; scale = true)
    val = scale * src
    @inbounds @simd for i in eachindex(dest, cond)
        dest[i] = ifelse(predicate(cond[i]), val, dest[i])
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Overwrite cells of `dest` for which `predicate(cond)` holds with `scale .* src`, where `dest`,
`cond`, and `src` are all cell-centered fields on `grid`. See the scalar-`src` method above for why
this is a fused branchless pass rather than a boolean-mask assignment.
"""
function overwrite_where!(grid::AbstractHydroGrid, dest, cond, predicate, src; scale = true)
    @inbounds @simd for i in eachindex(dest, cond, src)
        dest[i] = ifelse(predicate(cond[i]), scale * src[i], dest[i])
    end
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