global SECONDS_PER_YEAR = 60^2 * 24 * 365.25
global METERS_PER_KILOMETER = 1000


"""
$(TYPEDSIGNATURES)

Compute the cell-centered grid limits with half-cell padding from coordinate vectors.

# Arguments

- `xc::AbstractArray`: the x values of the grid cell centers.
- `yc::AbstractArray`: the y values of the grid cell centers.
"""
function compute_lims(xc::AbstractArray, yc::AbstractArray)
    dx = xc[2] - xc[1]
    dy = yc[2] - yc[1]
    xlims_raw = extrema(xc)
    ylims_raw = extrema(yc)
    xlims = (xlims_raw[1] - dx/2, xlims_raw[2] + dx/2)
    ylims = (ylims_raw[1] - dy/2, ylims_raw[2] + dy/2)
    return xlims, ylims
end


"""
$(TYPEDSIGNATURES)

Convert input float `f` from per year to per second.
"""
function perYear2perSecond(f::T) where {T <: Real}
    f / T(SECONDS_PER_YEAR)
end


"""
$(TYPEDSIGNATURES)

Convert input float `f` from per second to per year.
"""
function perSecond2perYear(f::T) where {T <: Real}
    f * T(SECONDS_PER_YEAR)
end


"""
$(TYPEDSIGNATURES)

Convert input float `f` from kilometers to meters.
"""
function Km2m(f::T) where {T <: Real}
    f * T(METERS_PER_KILOMETER)
end
