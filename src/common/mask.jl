####################################
# Hydrology mask: cell categories  #
####################################

# The mask codes are Shakti.jl's (src/mask.jl there), so one mask array drives both packages. They
# are floats, like the fields they are stored next to in `HydroState`, so a comparison such as
# `mask[i, j] == GROUNDED` needs no conversion on any array type.

"""
Mask value: grounded ice whose hydrology is solved.
"""
const GROUNDED = 0.0

"""
Mask value: ocean or floating ice. An outlet: water routed into it leaves the system, never the
reverse. Its water pressure is the ocean's hydrostatic pressure at the bed.
"""
const OCEAN = 1.0

"""
Mask value: ice-free land. An outlet: water routed into it leaves the system, never the reverse. Its
water pressure is atmospheric.
"""
const LAND = 2.0

"""
Mask value: grounded ice outside the basin being solved, or a cell without data. Closed: no water
crosses a face between it and a `GROUNDED` cell. Where it carries grounded ice, its potential still
counts in the coupling smoothing where a model allows it (see `KazmierczakHydroModel`'s
`potential_weight`).
"""
const OTHER_BASIN = 3.0

const MASK_CODES = (GROUNDED, OCEAN, LAND, OTHER_BASIN)


"""
$(TYPEDSIGNATURES)

`true` for the outlet codes `OCEAN` and `LAND`: water may leave the grounded system into the cell.
"""
@inline is_outlet(m) = (m == OCEAN) | (m == LAND)


"""
$(TYPEDSIGNATURES)

The hydrology mask: every cell's category, one of the codes [`GROUNDED`](@ref) (0), [`OCEAN`](@ref)
(1), [`LAND`](@ref) (2) and [`OTHER_BASIN`](@ref) (3), as in Shakti.jl. Required by
[`HydroState`](@ref), which takes no plain array: a FastHydrology mask from before version 3 used
`1` for grounded ice, which is `OCEAN` here, so an old array must not be read with the new meaning.
Wrap codes you set yourself, `HydroMask(codes)`, or build one from the geometry with [`classify_mask`](@ref).

Water leaves the grounded system only into `OCEAN` and `LAND` cells. Faces to `OTHER_BASIN` cells and
the edge of the grid are closed.

A `HydroMask` is a matrix of the codes: index it, compare it (`mask .== GROUNDED`) or change it
(`mask[end, :] .= OCEAN`) like any array. The constructor and every assignment check that the values
are mask codes.
"""
struct HydroMask{T <: Real, A <: AbstractMatrix{T}} <: AbstractMatrix{T}
    codes::A
    function HydroMask(codes::AbstractMatrix{<:Real})
        bad = findfirst(!is_mask_code, codes)
        bad === nothing || throw(ArgumentError("HydroMask: value $(codes[bad]) at $(Tuple(bad)) is not a mask code. " *
            "Use GROUNDED = 0, OCEAN = 1, LAND = 2 or OTHER_BASIN = 3 (Shakti.jl's codes), or build the mask with classify_mask."))
        return new{eltype(codes), typeof(codes)}(codes)
    end
end

is_mask_code(c) = any(==(c), MASK_CODES)

Base.size(m::HydroMask) = size(m.codes)
Base.parent(m::HydroMask) = m.codes
Base.IndexStyle(::Type{<:HydroMask{T, A}}) where {T, A} = IndexStyle(A)
Base.@propagate_inbounds Base.getindex(m::HydroMask, i::Int...) = m.codes[i...]
Base.@propagate_inbounds function Base.setindex!(m::HydroMask, v, i::Int...)
    is_mask_code(v) || throw(ArgumentError("HydroMask: $v is not a mask code (GROUNDED = 0, OCEAN = 1, LAND = 2, OTHER_BASIN = 3)"))
    m.codes[i...] = v
    return m
end


"""
$(TYPEDSIGNATURES)

Build a [`HydroMask`](@ref) from ice thickness `h` and bed elevation `b` [m], for inputs that do not
come with the four categories. A convenience, not a substitute for knowing the domain: check the result.

Cells are classified in this order:
1. `closed[i, j]` (if given): [`OTHER_BASIN`](@ref), e.g. ice outside the basin being solved, or
   padding without data around a rasterised mesh.
2. `grounded[i, j]` (if given): [`GROUNDED`](@ref). Use it for the host model's own grounded mask, which
   overrides the flotation test below. Without it, every cell with grounded ice is `GROUNDED`.
3. Open water or floating ice, `rho_i*h < -rho_sw*b`: [`OCEAN`](@ref).
4. Other ice (`h > 0`): grounded ice not in `grounded`, so [`OTHER_BASIN`](@ref).
5. The rest, ice-free with the bed at or above sea level: [`LAND`](@ref).

`rho_i` and `rho_sw` [kg/m3] default to `KazmierczakHydroModel`'s.
"""
function classify_mask(h::AbstractMatrix{<:Real}, b::AbstractMatrix{<:Real};
                       grounded::Union{Nothing, AbstractMatrix{Bool}} = nothing,
                       closed::Union{Nothing, AbstractMatrix{Bool}} = nothing,
                       rho_i::Real = 917.0, rho_sw::Real = 1028.0)
    size(h) == size(b) || throw(ArgumentError("classify_mask: h size $(size(h)) != b size $(size(b))"))
    for (name, a) in (("grounded", grounded), ("closed", closed))
        (a === nothing || size(a) == size(h)) || throw(ArgumentError("classify_mask: $name size $(size(a)) != h size $(size(h))"))
    end
    codes = map(CartesianIndices(h)) do I
        floating = rho_i * h[I] < -rho_sw * b[I]
        if closed !== nothing && closed[I]
            OTHER_BASIN
        elseif grounded === nothing ? (h[I] > 0 && !floating) : grounded[I]
            GROUNDED
        elseif floating
            OCEAN
        elseif h[I] > 0
            OTHER_BASIN
        else
            LAND
        end
    end
    return HydroMask(codes)
end
