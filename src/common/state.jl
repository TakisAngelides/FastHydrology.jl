"""
$(TYPEDSIGNATURES)

Abstract type for a hydrology state to hold fields common to all hydrology models.
"""
abstract type AbstractHydroState end


"""
$(TYPEDSIGNATURES)

The struct for the HydroState which includes physical fields important to hydrology.
"""
struct HydroState{A} <: AbstractHydroState
    # Inputs
    mask ::A  # hydrology mask codes: GROUNDED, OCEAN, LAND, OTHER_BASIN (see HydroMask)
    h    ::A  # ice thickness [m]
    b    ::A  # bedrock elevation [m]

    # Outputs
    N    ::A  # effective pressure [Pa]
    W    ::A  # water layer thickness [m]
end


"""
$(TYPEDSIGNATURES)

The constructor to the struct for the HydroState. The user must provide inputs for the ice thickness h, the bedrock elevation b,
and the hydrology mask, a [`HydroMask`](@ref) saying which cells are grounded ice to solve, outlets (ocean, land) or closed
(another basin). The remaining fields of the state are initialized to zero, specifically the effective pressure N [Pa] and
water layer thickness W [m]. `state.mask` holds the codes as a plain field.

Works with any concrete subtype of AbstractHydroGrid -- changing the grid does not require changing this constructor.

# Arguments

- `grid::AbstractHydroGrid`: grid of the simulation
- `mask_in::HydroMask`: the hydrology mask (see [`HydroMask`](@ref), [`classify_mask`](@ref)).
- `h_in::AbstractArray{<:AbstractFloat}`: ice thickness input
- `b_in::AbstractArray{<:AbstractFloat}`: bedrock elevation
"""
function HydroState(
    grid::AbstractHydroGrid,
    mask_in::HydroMask,
    h_in::AbstractArray{<:AbstractFloat},
    b_in::AbstractArray{<:AbstractFloat},
)

    # Check that the inputs match the grid size
    expected_size = (grid.Nx, grid.Ny)
    for (name, arr) in [("mask", mask_in.codes), ("h", h_in), ("b", b_in)]
        size(arr)[1:2] == expected_size || throw(ArgumentError("$name size $(size(arr)) != grid size $expected_size"))
    end

    # Inputs
    mask = alloc_field(grid, mask_in.codes)  # hydrology mask codes
    h    = alloc_field(grid, h_in)     # ice thickness
    b    = alloc_field(grid, b_in)     # bedrock elevation

    # Outputs
    N = alloc_field(grid)  # effective pressure
    W = alloc_field(grid)  # water layer thickness

    return HydroState(mask, h, b, N, W)

end


# A plain array is refused rather than read as codes: before version 3 FastHydrology used 1 for
# grounded ice, which is OCEAN in the current codes.
function HydroState(grid::AbstractHydroGrid, mask_in::AbstractArray{<:Real}, h_in::AbstractArray{<:AbstractFloat}, b_in::AbstractArray{<:AbstractFloat})
    throw(ArgumentError("HydroState takes a HydroMask, not a plain array. The mask codes are GROUNDED = 0, OCEAN = 1, LAND = 2, " *
        "OTHER_BASIN = 3 (as in Shakti.jl); an old 0/1 mask with 1 = grounded would read as ocean. Wrap codes you have set in " *
        "HydroMask(codes), or build the mask from the geometry, e.g. classify_mask(h, b; grounded = old_mask .== 1)."))
end
