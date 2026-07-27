"""
Exercises the `FastHydrologyShaktiExt` package extension: wraps a real `Shakti.Simulation` in a
`ShaktiHydroModel`/`TimeSimulation`, then checks that `FastHydrology.run!` actually dispatches into
`Shakti.run!` and steps the state forward. Kept in its own module (rather than folded into
`runtests.jl`) because `FastHydrology` and `Shakti` both export a function named `run!` -- `using`
both in the same scope makes the unqualified name ambiguous, so every call here is qualified as
`FastHydrology.run!`/`Shakti.run!` rather than relying on `using`-merged exports.
"""
module ShaktiExtTest

using FastHydrology
using Shakti
using Test

@testset "ShaktiHydroModel via TimeSimulation" begin

    # Same minimal nontrivial mask/state recipe as Shakti's own test suite (see
    # Shakti/test/runtests.jl): sloped bed, a point-source moulin, mixed
    # GROUNDED/OCEAN/LAND/OTHER_BASIN boundary cells.
    nx, ny = 4, 4
    grid = Shakti.Grid(nx, ny, 1e3, 1e3)
    state = Shakti.State(grid)
    p = Shakti.ModelParameters(e_v = 0.0)
    mi = Shakti.ConstantMeltInput()
    sl = Shakti.RegularizedCoulombSlidingLaw(0.25)

    mask = fill(Shakti.GROUNDED, nx, ny)
    mask[end, :] .= Shakti.OCEAN
    mask[1, :]   .= Shakti.LAND
    mask[:, 1]   .= Shakti.OTHER_BASIN

    A_visc = fill(5e-25, nx, ny)
    zb     = repeat(reshape(-0.02 .* grid.x, nx, 1), 1, ny)
    zs     = zb .+ 500.0
    b      = fill(0.01, nx, ny)
    G      = fill(0.06, nx, ny)
    ub_x   = fill(1e-6, nx + 1, ny)
    ub_y   = zeros(nx, ny + 1)
    ieb    = zeros(nx, ny)
    ieb[2, 2] = 3 / (grid.dx * grid.dy)
    taub_x = zeros(nx + 1, ny)
    taub_y = zeros(nx, ny + 1)

    Shakti.set_initial_conditions!(state, grid, p, mi, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

    ls = Shakti.CholeskyDirectSolver(grid)
    ps = Shakti.PicardSolver(500, 1e-6, ls, grid)

    tsteps = 3
    dt = 3600.0
    shakti_sim = Shakti.Simulation(grid, state, tsteps, dt, p, "implicit", String[], mi, sl; ps = ps)

    model = ShaktiHydroModel(shakti_sim)
    @test model isa FastHydrology.AbstractHydroModel

    time_sim = TimeSimulation(model)
    @test time_sim isa FastHydrology.AbstractSimulation

    FastHydrology.run!(time_sim)

    @test shakti_sim.total_time[] ≈ tsteps * dt
    @test all(isfinite, Array(shakti_sim.state.h))
    @test all(isfinite, Array(shakti_sim.state.N))
    @test all(isfinite, Array(shakti_sim.state.b))

end

@testset "ShaktiHydroModel step! (coupled-driver entry point)" begin

    # Fresh grid/state/solver, independent of the run! testset above -- step! is meant to be
    # called from a driver's own loop (e.g. an ice flow model updating geometry each iteration),
    # not through Shakti.run!, so this exercises that path directly.
    nx, ny = 4, 4
    grid = Shakti.Grid(nx, ny, 1e3, 1e3)
    state = Shakti.State(grid)
    p = Shakti.ModelParameters(e_v = 0.0)
    mi = Shakti.ConstantMeltInput()
    sl = Shakti.RegularizedCoulombSlidingLaw(0.25)

    mask = fill(Shakti.GROUNDED, nx, ny)
    mask[end, :] .= Shakti.OCEAN
    mask[1, :]   .= Shakti.LAND
    mask[:, 1]   .= Shakti.OTHER_BASIN

    A_visc = fill(5e-25, nx, ny)
    zb     = repeat(reshape(-0.02 .* grid.x, nx, 1), 1, ny)
    zs     = zb .+ 500.0
    b      = fill(0.01, nx, ny)
    G      = fill(0.06, nx, ny)
    ub_x   = fill(1e-6, nx + 1, ny)
    ub_y   = zeros(nx, ny + 1)
    ieb    = zeros(nx, ny)
    ieb[2, 2] = 3 / (grid.dx * grid.dy)
    taub_x = zeros(nx + 1, ny)
    taub_y = zeros(nx, ny + 1)

    Shakti.set_initial_conditions!(state, grid, p, mi, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

    ls = Shakti.CholeskyDirectSolver(grid)
    ps = Shakti.PicardSolver(500, 1e-6, ls, grid)

    dt = 3600.0
    shakti_sim = Shakti.Simulation(grid, state, 1, dt, p, "implicit", String[], mi, sl; ps = ps)

    time_sim = TimeSimulation(ShaktiHydroModel(shakti_sim))

    # Simulate a coupled driver: mutate geometry directly on the Shakti state, then advance the
    # hydrology by one step, three times -- checking total_time[] tracks the number of step! calls
    # (not the tsteps the Simulation happened to be constructed with, since step! ignores tsteps).
    for i in 1:3
        shakti_sim.state.zs .+= 0.1 # a stand-in for an ice flow model updating surface elevation
        FastHydrology.step!(time_sim)
        @test shakti_sim.total_time[] ≈ i * dt
    end

    @test all(isfinite, Array(shakti_sim.state.h))
    @test all(isfinite, Array(shakti_sim.state.N))

end

end # module
