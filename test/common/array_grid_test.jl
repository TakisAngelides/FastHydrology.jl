# Mirrors the OGRectHydroGrid testsets in runtests.jl, but on ArrayHydroGrid (plain Arrays, no
# Oceananigans dependency), to catch anything in src/ that implicitly assumes Oceananigans Field
# semantics (e.g. halo cells readable at index 0/Nx+1) instead of going through the grid interface.
# `field_values`/`interior` are already in scope from runtests.jl, which includes this file.

array_field_values(field) = field  # ArrayHydroGrid fields are already plain arrays

@testset "ArrayHydroGrid" begin

    @testset "Grid and state construction" begin
        grid = ArrayHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
        @test grid.Nx == 5
        @test grid.Ny == 5

        mask = ones(5, 5)
        h    = fill(500.0, 5, 5)
        b    = fill(-100.0, 5, 5)
        state = HydroState(grid, mask, h, b)

        @test state.N isa Matrix{Float64}
        @test all(isfinite, array_field_values(state.N))
        @test all(isfinite, array_field_values(state.W))
    end

    @testset "KazmierczakHydroModel steady state" begin
        grid = ArrayHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
        mask = ones(5, 5)
        h    = [500.0 - 5.0 * i for i in 1:5, j in 1:5]
        b    = [-100.0 - 2.0 * j for i in 1:5, j in 1:5]
        state = HydroState(grid, mask, h, b)

        kappa   = zeros(5, 5)
        abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), 5, 5)
        A_visc  = fill(1e-24, 5, 5)
        mdot    = fill(1e-6, 5, 5)
        model = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot)

        sim = SteadyStateSimulation(model, grid, state)
        run!(sim)

        @test all(isfinite, array_field_values(state.N))
        @test all(isfinite, array_field_values(state.W))
        @test all(>=(0.0), array_field_values(state.N))
        @test all(>=(0.0), array_field_values(state.W))
        # Wmin/Wmax default to 0.0/Inf (no clamp) -- see the equivalent comment in runtests.jl.
    end

    @testset "KazmierczakHydroModel dissipation melt term" begin
        grid = ArrayHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
        mask = ones(5, 5)
        h    = [500.0 - 5.0 * i for i in 1:5, j in 1:5]
        b    = [-100.0 - 2.0 * j for i in 1:5, j in 1:5]

        kappa   = zeros(5, 5)
        abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), 5, 5)
        A_visc  = fill(1e-24, 5, 5)
        mdot    = fill(1e-6, 5, 5)

        model_on  = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot; dissipation_melt = true, dissipation_verbose = false)
        model_off = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot; dissipation_melt = false, dissipation_verbose = false)

        @test model_on.dissipation_melt isa FastHydrology.DissipationMeltOn
        @test model_off.dissipation_melt isa FastHydrology.DissipationMeltOff

        run!(SteadyStateSimulation(model_on, grid, HydroState(grid, mask, h, b)))
        run!(SteadyStateSimulation(model_off, grid, HydroState(grid, mask, h, b)))

        @test all(array_field_values(model_off.mdot_total) .== array_field_values(model_off.mdot))
        @test all(array_field_values(model_on.mdot_total) .>= array_field_values(model_on.mdot))

        q_on  = array_field_values(model_on.q)
        q_off = array_field_values(model_off.q)

        @test q_on != q_off
        @test maximum(abs.(q_on .- q_off)) / maximum(abs.(q_off)) < 0.05

        @test all(isfinite, array_field_values(model_on.q))
        @test all(isfinite, array_field_values(model_off.q))

        model_capped = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot; max_dissipation_iters = 1, dissipation_verbose = false)
        run!(SteadyStateSimulation(model_capped, grid, HydroState(grid, mask, h, b)))
        @test all(isfinite, array_field_values(model_capped.q))
    end

    @testset "KazmierczakHydroModel with a flat, zero-flux region" begin
        grid = ArrayHydroGrid(10, 10, (0.0, 1000.0), (0.0, 1000.0))
        mask = [i <= 5 ? 1.0 : 0.0 for i in 1:10, j in 1:10]
        h    = [i <= 5 ? 500.0 - 5.0 * i : 0.0 for i in 1:10, j in 1:10]
        b    = [i <= 5 ? -100.0 - 2.0 * j : 0.0 for i in 1:10, j in 1:10]
        state = HydroState(grid, mask, h, b)

        kappa   = zeros(10, 10)
        abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), 10, 10)
        A_visc  = fill(1e-24, 10, 10)
        mdot    = fill(1e-6, 10, 10)
        model = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot)

        sim = SteadyStateSimulation(model, grid, state)
        run!(sim)

        @test all(isfinite, array_field_values(model.S_inf))
        @test all(isfinite, array_field_values(state.N))
    end

    @testset "HABHydroModel steady state" begin
        grid = ArrayHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
        mask = ones(5, 5)
        h    = fill(500.0, 5, 5)
        b    = fill(-100.0, 5, 5)
        state = HydroState(grid, mask, h, b)

        model = HABHydroModel(grid)
        sim = SteadyStateSimulation(model, grid, state)
        run!(sim)

        @test all(isfinite, array_field_values(state.N))
        @test all(>=(0.0), array_field_values(state.N))
    end

    @testset "ArrayHydroGrid vs OGRectHydroGrid numerical agreement" begin
        # Same physical setup on both grid backends should give the same steady-state result,
        # since the underlying physics code is grid-agnostic (see grid.jl's interface).
        #
        # Uses the model's default longcoupwater (5.0, stress-gradient-coupling smoothing on) and a
        # mixed kappa pattern deliberately: this is a regression test for a real bug where
        # OGRectHydroGrid's convolve! (grid.jl) passed Field.data -- which includes Oceananigans'
        # own halo padding around the (Nx, Ny) interior -- straight to cached_fft_convolve!
        # (fft_convolution.jl), which infers its logical domain size from the array's own shape and
        # so silently convolved over the halo-inflated domain instead of the real one. That corrupted
        # abs_grad_phi0_s (confirmed by an isolated comparison: up to several hundred percent off)
        # and, downstream, q/S_inf/H/W -- but not state.N in the uniform-kappa, longcoupwater=0
        # configuration the previous version of this test used, since N happened to saturate at its
        # sigmat*Po/Po clamp regardless. Checking q/S_inf/H/W directly, with kappa varying (so H's
        # hard/soft blend is actually exercised) and the default longcoupwater, is what catches it.
        Nx, Ny = 5, 5
        mask = ones(Nx, Ny)
        h    = [500.0 - 5.0 * i for i in 1:Nx, j in 1:Ny]
        b    = [-100.0 - 2.0 * j for i in 1:Nx, j in 1:Ny]
        kappa   = [mod(i + j, 2) == 0 ? 0.0 : 0.7 for i in 1:Nx, j in 1:Ny]
        abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), Nx, Ny)
        A_visc  = fill(1e-24, Nx, Ny)
        mdot    = fill(1e-6, Nx, Ny)

        grid_a = ArrayHydroGrid(Nx, Ny, (0.0, 500.0), (0.0, 500.0))
        state_a = HydroState(grid_a, mask, h, b)
        model_a = KazmierczakHydroModel(grid_a, kappa, abs_v_b, A_visc, mdot; dissipation_verbose = false)
        run!(SteadyStateSimulation(model_a, grid_a, state_a))

        grid_o = OGRectHydroGrid(Nx, Ny, (0.0, 500.0), (0.0, 500.0))
        state_o = HydroState(grid_o, mask, h, b)
        model_o = KazmierczakHydroModel(grid_o, kappa, abs_v_b, A_visc, mdot; dissipation_verbose = false)
        run!(SteadyStateSimulation(model_o, grid_o, state_o))

        @test isapprox(array_field_values(state_a.N), field_values(state_o.N); rtol = 1e-8)
        @test isapprox(array_field_values(state_a.W), field_values(state_o.W); rtol = 1e-8)
        @test isapprox(array_field_values(model_a.q), field_values(model_o.q); rtol = 1e-8)
        @test isapprox(array_field_values(model_a.H), field_values(model_o.H); rtol = 1e-8)
        @test isapprox(array_field_values(model_a.S_inf), field_values(model_o.S_inf); rtol = 1e-8)
        @test isapprox(array_field_values(model_a.abs_grad_phi0_s), field_values(model_o.abs_grad_phi0_s); rtol = 1e-8)
    end

    @testset "convolve! grid-backend agreement and mass conservation" begin
        # Lower-level, direct regression test for the convolve! bug described above -- isolates the
        # convolution itself from the rest of the physics. A normalized kernel convolved with
        # "replicate" (edge-extended) boundary padding must conserve the sum of the field (nothing
        # enters or leaves through a replicated edge), and OGRectHydroGrid's FFT path must agree with
        # ArrayHydroGrid's imfilter! path to floating-point precision, not just qualitatively.
        Nx, Ny = 5, 5
        src = [Float64(i + j) for i in 1:Nx, j in 1:Ny]

        k = 3 # kernel radius comparable to the grid size, like the model's default longcoupwater does on a small domain -- this is what made the bug's effect large enough to be obvious
        kernel = [max(0.0, 1.0 - sqrt((i - k - 1)^2 + (j - k - 1)^2) / k) for i in 1:2k+1, j in 1:2k+1]
        kernel ./= sum(kernel)

        grid_a = ArrayHydroGrid(Nx, Ny, (0.0, 500.0), (0.0, 500.0))
        dest_a = zeros(Nx, Ny)
        convolve!(grid_a, dest_a, src, kernel)

        grid_o = OGRectHydroGrid(Nx, Ny, (0.0, 500.0), (0.0, 500.0))
        src_o  = FastHydrology.alloc_field(grid_o, src)
        dest_o = FastHydrology.alloc_field(grid_o)
        convolve!(grid_o, dest_o, src_o, kernel)
        dest_o_vals = field_values(dest_o)

        @test isapprox(sum(dest_a), sum(src); rtol = 1e-10)
        @test isapprox(sum(dest_o_vals), sum(src); rtol = 1e-10)
        @test isapprox(dest_a, dest_o_vals; rtol = 1e-8)
    end

end
