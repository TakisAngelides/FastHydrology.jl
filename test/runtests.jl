using FastHydrology
using Oceananigans: interior
using Test
using MAT
using NCDatasets

# Tests construct OGRectHydroGrid explicitly, so it's fine to read field values back out via
# Oceananigans' own `interior`, rather than needing a grid-agnostic accessor.
field_values(field) = interior(field, :, :, 1)

@testset "FastHydrology.jl" begin
    include("common/grid_state_test.jl")
    include("models/kazmierczak2024/kazmierczak2024_test.jl")
    include("models/hab/hab_test.jl")
    include("models/kazmierczak2024/data_loaders_test.jl")
end

# Reuses field_values/interior from above, so include at top level (unlike shakti_ext_test.jl,
# ArrayHydroGrid has no naming collisions requiring its own module).
include("common/array_grid_test.jl")

# Reuses field_values/interior and the run! helpers from above, same reasoning as array_grid_test.jl.
include("common/output_checkpoint_test.jl")

# Own module (see shakti_ext_test.jl) since FastHydrology and Shakti both export `run!`.
include("models/shakti/shakti_ext_test.jl")
