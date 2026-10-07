using FastHydrology
using Test
using MAT
using NCDatasets

@testset "FastHydrology.jl" begin
    include("common/grid_state_test.jl")
    include("models/kazmierczak2024/kazmierczak2024_test.jl")
    include("models/kazmierczak2024/routing_schemes_test.jl")
    include("models/hab/hab_test.jl")
    include("models/kazmierczak2024/data_loaders_test.jl")
end

include("common/array_grid_test.jl")

include("common/output_checkpoint_test.jl")

# Own module (see shakti_ext_test.jl) since FastHydrology and Shakti both export `run!`.
include("models/shakti/shakti_ext_test.jl")
