"""
Cached, plan-reusing replacement for `ImageFiltering.imfilter!`'s FFT algorithm, used for the
stress-gradient-coupling convolution in `update_smoothed_potential_gradients!`.

`imfilter!`'s FFT path allocates fresh FFTW plans and padded/complex buffers on every call
(confirmed by profiling: ~20 MB per convolution on a realistic grid), because it offers no public
hook to reuse them across calls. Since the same (image size, kernel size) combination repeats on
every timestep of a coupled simulation, caching the plans and buffers here removes nearly all of
that allocation.

Only symmetric kernels are supported (convolution == correlation for those), i.e. kernel(-x, -y) ==
kernel(x, y) -- which holds for the coupling kernel this is used for (physically circular/elliptical
in real distance, so point-symmetric about its center) regardless of whether the kernel *array* is
square. The kernel array itself need not be square either: `frb_x`/`frb_y` (its radius in each
dimension) are tracked independently throughout, since `update_smoothed_potential_gradients!`
builds a rectangular kernel array whenever the grid's dx != dy.
"""

using FFTW: FFTW
using LinearAlgebra: mul!

"""
$(TYPEDSIGNATURES)

Holds the FFTW plans and padded/complex buffers for one (image size, kernel radius) combination,
so `cached_fft_convolve!` can reuse them across calls instead of reallocating on every timestep.
"""
mutable struct FFTConvCache
    frb_x::Int                      # kernel radius in x: kernel is (2*frb_x+1, 2*frb_y+1)
    frb_y::Int                      # kernel radius in y
    full_size::Tuple{Int,Int}       # padded array size, (Nx+2*frb_x, Ny+2*frb_y)
    padded_src::Matrix{Float64}     # src embedded in the middle, "replicate"-padded border
    padded_kernel::Matrix{Float64}  # kernel embedded via wrap-around indexing, for circular FFT convolution
    plan::Any     # FFTW real-to-complex plan; left untyped to avoid depending on FFTW's internal plan type name
    inv_plan::Any # FFTW complex-to-real (inverse) plan
    Fsrc::Matrix{ComplexF64}     # rfft(padded_src)
    Fkern::Matrix{ComplexF64}    # rfft(padded_kernel)
    Fresult::Matrix{ComplexF64}  # Fsrc .* Fkern, the frequency-domain product
    result_full::Matrix{Float64} # irfft(Fresult); the full circular convolution, before cropping
end

"""
$(TYPEDSIGNATURES)

Build a new `FFTConvCache` for an `(Nx, Ny)` image convolved with a kernel of radius `(frb_x,
frb_y)`, allocating the padded buffers and FFTW plans once so later calls (via
`get_fft_conv_cache!`) only pay for the FFT/multiply/inverse-FFT work itself.
"""
function FFTConvCache(Nx::Int, Ny::Int, frb_x::Int, frb_y::Int)
    full_size = (Nx + 2 * frb_x, Ny + 2 * frb_y) # frb_x/frb_y cells of border padding on every side, per axis
    padded_src = zeros(Float64, full_size)
    padded_kernel = zeros(Float64, full_size)
    plan = FFTW.plan_rfft(padded_src)   # this measures/builds the FFTW plan, the expensive step being cached
    Fsrc = plan * padded_src            # also fixes the frequency-domain array's shape/eltype for the buffers below
    Fkern = similar(Fsrc)
    Fresult = similar(Fsrc)
    inv_plan = FFTW.plan_irfft(Fresult, full_size[1]) # original row count must be passed explicitly: rfft's compressed output shape can't tell an even-length input from an odd-length one
    result_full = zeros(Float64, full_size)
    return FFTConvCache(frb_x, frb_y, full_size, padded_src, padded_kernel, plan, inv_plan, Fsrc, Fkern, Fresult, result_full)
end

"""
$(TYPEDSIGNATURES)

Return the cache stored in `cache_ref`, rebuilding it if it doesn't exist yet or if the
(image size, kernel size) combination has changed since the last call.
"""
function get_fft_conv_cache!(cache_ref::Base.RefValue, Nx::Int, Ny::Int, frb_x::Int, frb_y::Int)
    c = cache_ref[]
    # rebuild on first use, or if the (image size, kernel radius) combination changed since
    # the last call (e.g. a different grid or a different coupling kernel)
    if c === nothing || !(c isa FFTConvCache) || c.frb_x != frb_x || c.frb_y != frb_y || c.full_size != (Nx + 2 * frb_x, Ny + 2 * frb_y)
        c = FFTConvCache(Nx, Ny, frb_x, frb_y)
        cache_ref[] = c
    end
    # cache_ref is typed as Ref{Any} (so callers can hold it without depending on this module's
    # concrete cache type), so assert the concrete type here to keep this function type-stable
    return c::FFTConvCache
end

"""
$(TYPEDSIGNATURES)

Copy `src` into the interior of `padded`, then fill the surrounding `(frb_x, frb_y)`-cell border
with "replicate" padding (nearest edge/corner pixel extended outward), matching `imfilter!`'s
default border behavior. `padded` must already be sized `(size(src,1)+2*frb_x,
size(src,2)+2*frb_y)`.
"""
function fill_padded_src!(padded::Matrix{Float64}, src::AbstractMatrix, frb_x::Int, frb_y::Int)
    Nx, Ny = size(src)
    padded[frb_x+1:frb_x+Nx, frb_y+1:frb_y+Ny] .= src # interior = the real data, unchanged

    # left/right borders: replicate the first/last column across the padding width, restricted
    # to the real data rows (the four corners are handled separately below)
    for j in 1:frb_y
        padded[frb_x+1:frb_x+Nx, j] .= @view src[:, 1]
        padded[frb_x+1:frb_x+Nx, frb_y+Ny+j] .= @view src[:, Ny]
    end

    # top/bottom borders: replicate the first/last real row across the padding height,
    # restricted to the real data columns (the four corners are handled separately below)
    for i in 1:frb_x
        padded[i, frb_y+1:frb_y+Ny] .= @view padded[frb_x+1, frb_y+1:frb_y+Ny]
        padded[frb_x+Nx+i, frb_y+1:frb_y+Ny] .= @view padded[frb_x+Nx, frb_y+1:frb_y+Ny]
    end

    # corners: replicate the corresponding corner pixel of src directly, since the row/column
    # loops above never reach the corner blocks
    padded[1:frb_x, 1:frb_y] .= padded[frb_x+1, frb_y+1]
    padded[1:frb_x, frb_y+Ny+1:end] .= padded[frb_x+1, frb_y+Ny]
    padded[frb_x+Nx+1:end, 1:frb_y] .= padded[frb_x+Nx, frb_y+1]
    padded[frb_x+Nx+1:end, frb_y+Ny+1:end] .= padded[frb_x+Nx, frb_y+Ny]
    return padded
end

"""
$(TYPEDSIGNATURES)

Embed the centered `kernel2d` (indexed `[di+frb_x+1, dj+frb_y+1]` for offset `(di, dj)` from its
center) into `padded` using wrap-around (circular) indexing, so that an ordinary circular
convolution via FFT lines up with `fill_padded_src!`'s padding: multiplying the two FFTs and
inverse-transforming reproduces the centered correlation/convolution at each output cell.
"""
function fill_padded_kernel!(padded::Matrix{Float64}, kernel2d::AbstractMatrix, frb_x::Int, frb_y::Int, full_size::Tuple{Int,Int})
    fill!(padded, 0.0) # padded is reused across calls, so clear any stale kernel from a previous call
    for dj in -frb_y:frb_y, di in -frb_x:frb_x
        v = kernel2d[di+frb_x+1, dj+frb_y+1] # kernel value at offset (di, dj) from its center
        # wrap negative/positive offsets around the padded array's edges (mod into a 1-based
        # index) so offset (0,0) lands at padded[1,1] and e.g. offset (-1,0) lands at the last
        # row instead of going out of bounds
        ii = mod(di, full_size[1]) + 1
        jj = mod(dj, full_size[2]) + 1
        padded[ii, jj] = v
    end
    return padded
end

"""
$(TYPEDSIGNATURES)

Cached equivalent of `imfilter!(dest, src, centered(kernel))` with the default "replicate" border,
for a point-symmetric `kernel` (so convolution and correlation coincide). `dest`, `src` are
cell-centered (Nx, Ny, 1) arrays (possibly `OffsetArray`s, e.g. `Field.data`); `kernel` is
(2*frb_x+1, 2*frb_y+1, 1) -- not necessarily square. `cache_ref` should be a `Base.RefValue{Any}`
owned by the caller, reused across repeated calls with the same sizes.
"""
function cached_fft_convolve!(cache_ref::Base.RefValue, dest::AbstractArray, src::AbstractArray, kernel::AbstractArray)
    # Work through `parent` so indexing below is always plain 1-based, regardless of whether
    # dest/src/kernel are OffsetArrays (e.g. Field.data) or plain arrays.
    dest_p, src_p, kernel_p = parent(dest), parent(src), parent(kernel)

    frb_x = (size(kernel_p, 1) - 1) ÷ 2 # kernel is (2*frb_x+1, 2*frb_y+1, 1); recover frb_x/frb_y from its size
    frb_y = (size(kernel_p, 2) - 1) ÷ 2
    Nx, Ny = size(src_p, 1), size(src_p, 2)
    c = get_fft_conv_cache!(cache_ref, Nx, Ny, frb_x, frb_y) # (re)allocates only if size/frb changed since the last call

    fill_padded_src!(c.padded_src, view(src_p, :, :, 1), frb_x, frb_y)
    fill_padded_kernel!(c.padded_kernel, view(kernel_p, :, :, 1), frb_x, frb_y, c.full_size)

    mul!(c.Fsrc, c.plan, c.padded_src)      # forward FFT of the padded source
    mul!(c.Fkern, c.plan, c.padded_kernel)  # forward FFT of the wrapped kernel
    c.Fresult .= c.Fsrc .* c.Fkern          # pointwise product in frequency domain == convolution in space
    mul!(c.result_full, c.inv_plan, c.Fresult) # inverse FFT back to the padded spatial domain

    # crop back to the original (Nx, Ny) region, discarding the frb_x/frb_y-cell border used for padding
    view(dest_p, :, :, 1) .= @view c.result_full[frb_x+1:frb_x+Nx, frb_y+1:frb_y+Ny]
    return nothing
end
