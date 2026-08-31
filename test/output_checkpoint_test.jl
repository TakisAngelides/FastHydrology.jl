"""
Exercises `output.jl`/`checkpoint.jl`: periodic NetCDF output (including resume-after-trim) and
full-state checkpoint save/restore. Included at top level from `runtests.jl` (no naming collisions
with `FastHydrology`'s own exports, unlike `shakti_ext_test.jl`).

Follows the coupled-driver pattern this capability exists for (see e.g. Kryonomos.jl's MISMIP
scripts): a caller-owned loop calls `run!` on a `SteadyStateSimulation` each iteration and decides
itself when to call `write_output!`/`save_checkpoint`, since `SteadyStateSimulation` has no
internal loop of its own to hook a fixed interval into (see `AbstractOutputWriter`'s docstring).
"""

@testset "NetCDFOutputWriter: write, close, read back" begin
    grid = OGRectHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
    mask = ones(5, 5)
    h    = [500.0 - 5.0 * i for i in 1:5, j in 1:5]
    b    = [-100.0 - 2.0 * j for i in 1:5, j in 1:5]
    state = HydroState(grid, mask, h, b)

    kappa   = zeros(5, 5)
    abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), 5, 5)
    A_visc  = fill(1e-24, 5, 5)
    mdot    = fill(1e-6, 5, 5)
    model = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot; longcoupwater = 0.0, dissipation_verbose = false)
    sim = SteadyStateSimulation(model, grid, state)

    mktempdir() do dir
        path = joinpath(dir, "out.nc")
        writer = NetCDFOutputWriter(path, grid, [:N, :W, :mdot])
        @test writer isa AbstractOutputWriter

        expected_N = Vector{Array{Float64, 2}}(undef, 3)
        for k in 1:3
            run!(sim)
            write_output!(writer, k, Float64(k), (N = state.N, W = state.W, mdot = model.mdot))
            expected_N[k] = copy(field_values(state.N))
        end
        close_output!(writer)

        NCDataset(path, "r") do ds
            @test Array(ds["step"]) == [1, 2, 3]
            @test Array(ds["time"]) == [1.0, 2.0, 3.0]
            @test size(ds["N"]) == (5, 5, 3)
            for k in 1:3
                @test ds["N"][:, :, k] == expected_N[k]
            end
        end
    end
end

@testset "NetCDFOutputWriter: overwrite keyword" begin
    grid = OGRectHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
    mktempdir() do dir
        path = joinpath(dir, "out.nc")
        writer = NetCDFOutputWriter(path, grid, [:N])
        close_output!(writer)

        @test_throws ErrorException NetCDFOutputWriter(path, grid, [:N]; overwrite = false)

        # overwrite = true (the default) should succeed and start a fresh (empty) file.
        writer2 = NetCDFOutputWriter(path, grid, [:N])
        close_output!(writer2)
        NCDataset(path, "r") do ds
            @test size(ds["step"], 1) == 0
        end
    end
end

@testset "NetCDFOutputWriter: resume_step trims a partial write past the checkpoint" begin
    # Mirrors the incident this capability was built for: a checkpoint saved at step 3, but the
    # output file already has slices up through step 5 written by the time the process was killed
    # -- resuming from the step-3 checkpoint must not leave duplicate/orphaned slices for steps
    # 4-5 once the caller's loop replays them.
    grid = OGRectHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
    mask = ones(5, 5)
    h    = [500.0 - 5.0 * i for i in 1:5, j in 1:5]
    b    = [-100.0 - 2.0 * j for i in 1:5, j in 1:5]
    state = HydroState(grid, mask, h, b)

    kappa   = zeros(5, 5)
    abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), 5, 5)
    A_visc  = fill(1e-24, 5, 5)
    mdot    = fill(1e-6, 5, 5)
    model = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot; longcoupwater = 0.0, dissipation_verbose = false)
    sim = SteadyStateSimulation(model, grid, state)

    mktempdir() do dir
        path = joinpath(dir, "out.nc")
        writer = NetCDFOutputWriter(path, grid, [:N, :W])
        for k in 1:5
            run!(sim)
            write_output!(writer, k, Float64(k), (N = state.N, W = state.W))
        end
        close_output!(writer)

        # Resume as if only a step-3 checkpoint survived: steps 4-5 must be dropped before the
        # caller's loop re-does (and re-writes) them.
        writer2 = NetCDFOutputWriter(path, grid, [:N, :W]; resume_step = 3)
        for k in 4:6
            run!(sim)
            write_output!(writer2, k, Float64(k), (N = state.N, W = state.W))
        end
        close_output!(writer2)

        NCDataset(path, "r") do ds
            @test Array(ds["step"]) == [1, 2, 3, 4, 5, 6]
            @test Array(ds["time"]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        end
    end
end

@testset "NetCDFOutputWriter: resume_step with nothing to trim" begin
    # No slices past resume_step exist -- _trim_output_after! should be a no-op, not corrupt or
    # empty the file.
    grid = OGRectHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
    mktempdir() do dir
        path = joinpath(dir, "out.nc")
        writer = NetCDFOutputWriter(path, grid, [:N])
        for k in 1:3
            write_output!(writer, k, Float64(k), (N = zeros(5, 5) .+ k,))
        end
        close_output!(writer)

        writer2 = NetCDFOutputWriter(path, grid, [:N]; resume_step = 3)
        write_output!(writer2, 4, 4.0, (N = fill(4.0, 5, 5),))
        close_output!(writer2)

        NCDataset(path, "r") do ds
            @test Array(ds["step"]) == [1, 2, 3, 4]
        end
    end
end

@testset "NetCDFOutputWriter: missing field errors" begin
    grid = OGRectHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
    mktempdir() do dir
        path = joinpath(dir, "out.nc")
        writer = NetCDFOutputWriter(path, grid, [:N, :W])
        @test_throws ErrorException write_output!(writer, 1, 1.0, (N = zeros(5, 5),))
        close_output!(writer)
    end
end

@testset "save_checkpoint/load_checkpoint!: round-trip on HydroState" begin
    grid = OGRectHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
    mask = ones(5, 5)
    h    = [500.0 - 5.0 * i for i in 1:5, j in 1:5]
    b    = [-100.0 - 2.0 * j for i in 1:5, j in 1:5]
    state = HydroState(grid, mask, h, b)

    kappa   = zeros(5, 5)
    abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), 5, 5)
    A_visc  = fill(1e-24, 5, 5)
    mdot    = fill(1e-6, 5, 5)
    model = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot; longcoupwater = 0.0, dissipation_verbose = false)
    run!(SteadyStateSimulation(model, grid, state))

    mktempdir() do dir
        path = joinpath(dir, "ckpt.nc")
        save_checkpoint(path, state, 42, 123.5)

        fresh = HydroState(grid, zeros(5, 5), zeros(5, 5), zeros(5, 5))
        step, time = load_checkpoint!(fresh, path)

        @test step == 42
        @test time == 123.5
        @test field_values(fresh.mask) == field_values(state.mask)
        @test field_values(fresh.h)    == field_values(state.h)
        @test field_values(fresh.b)    == field_values(state.b)
        @test field_values(fresh.N)    == field_values(state.N)
        @test field_values(fresh.W)    == field_values(state.W)
    end
end

@testset "save_checkpoint: atomic write leaves no partial .tmp file behind" begin
    grid = OGRectHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
    mask = ones(5, 5)
    h    = fill(500.0, 5, 5)
    b    = fill(-100.0, 5, 5)
    state = HydroState(grid, mask, h, b)

    mktempdir() do dir
        path = joinpath(dir, "ckpt.nc")
        save_checkpoint(path, state, 1, 1.0)
        @test isfile(path)
        @test !isfile(path * ".tmp")

        # Overwriting an existing checkpoint (as a periodic checkpoint_every call would) must also
        # leave no stray temp file and must fully replace the old content.
        state.N .= 999.0 # not a real HydroState field mutation path, just to change something checkable
        save_checkpoint(path, state, 2, 2.0)
        @test !isfile(path * ".tmp")

        fresh = HydroState(grid, zeros(5, 5), zeros(5, 5), zeros(5, 5))
        step, time = load_checkpoint!(fresh, path)
        @test step == 2
        @test time == 2.0
    end
end

@testset "save_checkpoint/load_checkpoint! on ArrayHydroGrid (plain-array fields)" begin
    grid = ArrayHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
    mask = ones(5, 5)
    h    = [500.0 - 5.0 * i for i in 1:5, j in 1:5]
    b    = [-100.0 - 2.0 * j for i in 1:5, j in 1:5]
    state = HydroState(grid, mask, h, b)

    kappa   = zeros(5, 5)
    abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), 5, 5)
    A_visc  = fill(1e-24, 5, 5)
    mdot    = fill(1e-6, 5, 5)
    model = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot; longcoupwater = 0.0, dissipation_verbose = false)
    run!(SteadyStateSimulation(model, grid, state))

    mktempdir() do dir
        path = joinpath(dir, "ckpt.nc")
        save_checkpoint(path, state, 7, 7.0)

        fresh = HydroState(grid, zeros(5, 5), zeros(5, 5), zeros(5, 5))
        step, time = load_checkpoint!(fresh, path)

        @test step == 7
        @test time == 7.0
        @test fresh.N == state.N
        @test fresh.W == state.W
        @test fresh.mask == state.mask
    end
end

@testset "NetCDFOutputWriter on ArrayHydroGrid (plain-array fields)" begin
    grid = ArrayHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
    mask = ones(5, 5)
    h    = [500.0 - 5.0 * i for i in 1:5, j in 1:5]
    b    = [-100.0 - 2.0 * j for i in 1:5, j in 1:5]
    state = HydroState(grid, mask, h, b)

    kappa   = zeros(5, 5)
    abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), 5, 5)
    A_visc  = fill(1e-24, 5, 5)
    mdot    = fill(1e-6, 5, 5)
    model = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot; longcoupwater = 0.0, dissipation_verbose = false)
    run!(SteadyStateSimulation(model, grid, state))

    mktempdir() do dir
        path = joinpath(dir, "out.nc")
        writer = NetCDFOutputWriter(path, grid, [:N, :W])
        write_output!(writer, 1, 1.0, (N = state.N, W = state.W))
        close_output!(writer)

        NCDataset(path, "r") do ds
            @test ds["N"][:, :, 1] == state.N
            @test ds["W"][:, :, 1] == state.W
        end
    end
end
