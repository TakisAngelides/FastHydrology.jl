"""
$(TYPEDSIGNATURES)

Visualize a field. Implemented by the `FastHydrologyMakieExt` package extension, active once a
Makie backend (e.g. `CairoMakie`) is loaded alongside `FastHydrology` (`using FastHydrology,
CairoMakie`). Defined here as a stub (no methods) so the name is owned by `FastHydrology` and the
extension can add methods to it, mirroring how `step!` in run.jl is owned by `FastHydrology` and
extended by `FastHydrologyShaktiExt`.
"""
function visualize_field end

"""
$(TYPEDSIGNATURES)

Visualize the corners of a hydrology grid, showing cell centers and boundaries. Implemented by the
`FastHydrologyMakieExt` package extension -- see `visualize_field` for why this is a stub here.
"""
function visualize_grid end

"""
$(TYPEDSIGNATURES)

Set a field to a given input value where the mask is not 1. Does not touch plotting/Makie at all
(just field indexing), so unlike `visualize_field`/`visualize_grid` this has a real implementation
here rather than living behind the `FastHydrologyMakieExt` extension.
"""
function mask_field(field::Oceananigans.Fields.Field, mask, value)

    Nx, Ny = size(mask)
    res = deepcopy(field)
    for j in 1:Ny
        for i in 1:Nx
            if mask[i, j] != 1.0
                res[i, j, 1] = value
            end
        end
    end

    return res

end
