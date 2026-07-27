"""
Cached, plan-reusing replacement for `ImageFiltering.imfilter!`'s FFT algorithm, used for the
stress-gradient-coupling convolution in `update_smoothed_potential_gradients!`.

`imfilter!`'s FFT path allocates fresh FFTW plans and padded/complex buffers on every call
(confirmed by profiling: ~20 MB per convolution on a realistic grid), because it offers no public
hook to reuse them across calls. Since the same (image size, kernel size) combination repeats on
every timestep of a coupled simulation, caching the plans and buffers here removes nearly all of
that allocation.

The math (replicate-border padding, wrapped kernel embedding, real FFT, crop) is validated against
`imfilter!`'s output to floating-point precision — see the numerical cross-check in the docs before
this was wired in. Only symmetric kernels are supported (convolution == correlation for those),
which holds for the radially-symmetric coupling kernel this is used for.
"""

using FFTW: FFTW
using LinearAlgebra: mul!

mutable struct FFTConvCache
    frb::Int
    full_size::Tuple{Int,Int}
    padded_src::Matrix{Float64}
    padded_kernel::Matrix{Float64}
    plan::Any     # FFTW real-to-complex plan; left untyped to avoid depending on FFTW's internal plan type name
    inv_plan::Any # FFTW complex-to-real (inverse) plan
    Fsrc::Matrix{ComplexF64}
    Fkern::Matrix{ComplexF64}
    Fresult::Matrix{ComplexF64}
    result_full::Matrix{Float64}
end

function FFTConvCache(Nx::Int, Ny::Int, frb::Int)
    full_size = (Nx + 2 * frb, Ny + 2 * frb)
    padded_src = zeros(Float64, full_size)
    padded_kernel = zeros(Float64, full_size)
    plan = FFTW.plan_rfft(padded_src)
    Fsrc = plan * padded_src
    Fkern = similar(Fsrc)
    Fresult = similar(Fsrc)
    inv_plan = FFTW.plan_irfft(Fresult, full_size[1])
    result_full = zeros(Float64, full_size)
    return FFTConvCache(frb, full_size, padded_src, padded_kernel, plan, inv_plan, Fsrc, Fkern, Fresult, result_full)
end

"""
$(TYPEDSIGNATURES)

Return the cache stored in `cache_ref`, rebuilding it if it doesn't exist yet or if the
(image size, kernel size) combination has changed since the last call.
"""
function get_fft_conv_cache!(cache_ref::Base.RefValue, Nx::Int, Ny::Int, frb::Int)
    c = cache_ref[]
    if c === nothing || !(c isa FFTConvCache) || c.frb != frb || c.full_size != (Nx + 2 * frb, Ny + 2 * frb)
        c = FFTConvCache(Nx, Ny, frb)
        cache_ref[] = c
    end
    return c::FFTConvCache
end

function fill_padded_src!(padded::Matrix{Float64}, src::AbstractMatrix, frb::Int)
    Nx, Ny = size(src)
    padded[frb+1:frb+Nx, frb+1:frb+Ny] .= src
    for j in 1:frb
        padded[frb+1:frb+Nx, j] .= @view src[:, 1]
        padded[frb+1:frb+Nx, frb+Ny+j] .= @view src[:, Ny]
    end
    for i in 1:frb
        padded[i, frb+1:frb+Ny] .= @view padded[frb+1, frb+1:frb+Ny]
        padded[frb+Nx+i, frb+1:frb+Ny] .= @view padded[frb+Nx, frb+1:frb+Ny]
    end
    padded[1:frb, 1:frb] .= padded[frb+1, frb+1]
    padded[1:frb, frb+Ny+1:end] .= padded[frb+1, frb+Ny]
    padded[frb+Nx+1:end, 1:frb] .= padded[frb+Nx, frb+1]
    padded[frb+Nx+1:end, frb+Ny+1:end] .= padded[frb+Nx, frb+Ny]
    return padded
end

function fill_padded_kernel!(padded::Matrix{Float64}, kernel2d::AbstractMatrix, frb::Int, full_size::Tuple{Int,Int})
    fill!(padded, 0.0)
    for dj in -frb:frb, di in -frb:frb
        v = kernel2d[di+frb+1, dj+frb+1]
        ii = mod(di, full_size[1]) + 1
        jj = mod(dj, full_size[2]) + 1
        padded[ii, jj] = v
    end
    return padded
end

"""
$(TYPEDSIGNATURES)

Cached equivalent of `imfilter!(dest, src, centered(kernel))` with the default "replicate" border,
for a radially-symmetric `kernel` (so convolution and correlation coincide). `dest`, `src` are
cell-centered (Nx, Ny, 1) arrays (possibly `OffsetArray`s, e.g. `Field.data`); `kernel` is
(2*frb+1, 2*frb+1, 1). `cache_ref` should be a `Base.RefValue{Any}` owned by the caller, reused
across repeated calls with the same sizes.
"""
function cached_fft_convolve!(cache_ref::Base.RefValue, dest::AbstractArray, src::AbstractArray, kernel::AbstractArray)
    # Work through `parent` so indexing below is always plain 1-based, regardless of whether
    # dest/src/kernel are OffsetArrays (e.g. Field.data) or plain arrays.
    dest_p, src_p, kernel_p = parent(dest), parent(src), parent(kernel)

    frb = (size(kernel_p, 1) - 1) ÷ 2
    Nx, Ny = size(src_p, 1), size(src_p, 2)
    c = get_fft_conv_cache!(cache_ref, Nx, Ny, frb)

    fill_padded_src!(c.padded_src, view(src_p, :, :, 1), frb)
    fill_padded_kernel!(c.padded_kernel, view(kernel_p, :, :, 1), frb, c.full_size)

    mul!(c.Fsrc, c.plan, c.padded_src)
    mul!(c.Fkern, c.plan, c.padded_kernel)
    c.Fresult .= c.Fsrc .* c.Fkern
    mul!(c.result_full, c.inv_plan, c.Fresult)

    view(dest_p, :, :, 1) .= @view c.result_full[frb+1:frb+Nx, frb+1:frb+Ny]
    return nothing
end
