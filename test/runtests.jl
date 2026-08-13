using FastHydrology
using Oceananigans: interior
using Test
using MAT
using NCDatasets

# Tests construct OGRectHydroGrid explicitly, so it's fine to read field values back out via
# Oceananigans' own `interior`, rather than needing a grid-agnostic accessor.
field_values(field) = interior(field, :, :, 1)

@testset "FastHydrology.jl" begin

    @testset "Grid and state construction" begin
        grid = OGRectHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
        @test grid.Nx == 5
        @test grid.Ny == 5

        mask = ones(5, 5)
        h    = fill(500.0, 5, 5)
        b    = fill(-100.0, 5, 5)
        state = HydroState(grid, mask, h, b)

        @test all(isfinite, field_values(state.N))
        @test all(isfinite, field_values(state.W))
    end

    @testset "KazmierczakHydroModel steady state" begin
        grid = OGRectHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
        mask = ones(5, 5)
        # A flat h/b gives a zero geometric-potential gradient everywhere, which is
        # degenerate for this flux-routing model (division by ~0 in the correction
        # factor). Use a sloped surface, as in any real ice-sheet domain.
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

        @test all(isfinite, field_values(state.N))
        @test all(isfinite, field_values(state.W))
        @test all(>=(0.0), field_values(state.N))
        @test all(w -> model.Wmin <= w <= model.Wmax, field_values(state.W))
    end

    @testset "KazmierczakHydroModel dissipation melt term" begin
        # dissipation_melt toggles whether update_q! includes the |q * grad(phi0)| / L_w
        # source term (Eq. 3, Sec. 2.2.2 of Kazmierczak et al 2024, dropped there as negligible).
        grid = OGRectHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
        mask = ones(5, 5)
        h    = [500.0 - 5.0 * i for i in 1:5, j in 1:5]
        b    = [-100.0 - 2.0 * j for i in 1:5, j in 1:5]

        kappa   = zeros(5, 5)
        abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), 5, 5)
        A_visc  = fill(1e-24, 5, 5)
        mdot    = fill(1e-6, 5, 5)

        model_on  = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot; dissipation_melt = true, dissipation_verbose = false)
        model_off = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot; dissipation_melt = false, dissipation_verbose = false)

        # The on/off choice is resolved by multiple dispatch on this trait field, not a runtime Bool.
        @test model_on.dissipation_melt isa FastHydrology.DissipationMeltOn
        @test model_off.dissipation_melt isa FastHydrology.DissipationMeltOff

        run!(SteadyStateSimulation(model_on, grid, HydroState(grid, mask, h, b)))
        run!(SteadyStateSimulation(model_off, grid, HydroState(grid, mask, h, b)))

        # With the term off, the routing algorithm's source is exactly mdot.
        @test all(field_values(model_off.mdot_total) .== field_values(model_off.mdot))

        # With the term on, mdot_total = mdot + |q * grad(phi0)| / L_w is pointwise >= mdot.
        @test all(field_values(model_on.mdot_total) .>= field_values(model_on.mdot))

        q_on  = field_values(model_on.q)
        q_off = field_values(model_off.q)

        # The term has a real (if small) effect on the solution...
        @test q_on != q_off
        # ...consistent with the paper's own claim that it's negligible.
        @test maximum(abs.(q_on .- q_off)) / maximum(abs.(q_off)) < 0.05

        @test all(isfinite, field_values(model_on.q))
        @test all(isfinite, field_values(model_off.q))

        # max_dissipation_iters is a hard cap: it must not error even when it cuts the Picard
        # iteration off before convergence.
        model_capped = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot; max_dissipation_iters = 1, dissipation_verbose = false)
        run!(SteadyStateSimulation(model_capped, grid, HydroState(grid, mask, h, b)))
        @test all(isfinite, field_values(model_capped.q))
    end

    @testset "Sliding laws: calc_tau_b" begin
        N  = 1e5   # Pa
        vb = 200.0 / (60^2 * 24 * 365.25)  # m/s

        @test calc_tau_b(NoSlidingLaw(), N, vb) == 0.0

        law_w = WeertmanSlidingLaw(C = 1e7, q = 1/3)
        @test calc_tau_b(law_w, N, vb) ≈ 1e7 * vb^(1/3)
        # Independent of N.
        @test calc_tau_b(law_w, N, vb) == calc_tau_b(law_w, 2 * N, vb)

        law_pp = PowerPlasticSlidingLaw(c_till = 0.5, q = 1.0, u0 = perYear2perSecond(100.0))
        @test calc_tau_b(law_pp, N, vb) ≈ 0.5 * N * (vb / law_pp.u0)^1.0
        @test calc_tau_b(law_pp, 0.0, vb) == 0.0

        law_rc = RegularizedCoulombSlidingLaw(c_till = 0.5, q = 1/3, u0 = perYear2perSecond(100.0))
        @test calc_tau_b(law_rc, N, vb) ≈ 0.5 * N * (vb / (vb + law_rc.u0))^(1/3)
        # Saturates toward the Coulomb limit c_till * N as v_b -> infinity.
        @test calc_tau_b(law_rc, N, 1e10) ≈ 0.5 * N atol = 1e-3 * N
    end

    @testset "KazmierczakHydroModel with WeertmanSlidingLaw" begin
        # Weertman tau_b does not depend on N, so it needs no Picard loop: a single pass is exact.
        grid = OGRectHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
        mask = ones(5, 5)
        h    = [500.0 - 5.0 * i for i in 1:5, j in 1:5]
        b    = [-100.0 - 2.0 * j for i in 1:5, j in 1:5]
        state = HydroState(grid, mask, h, b)

        kappa   = zeros(5, 5)
        abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), 5, 5)
        A_visc  = fill(1e-24, 5, 5)
        mdot    = fill(1e-6, 5, 5)

        law   = WeertmanSlidingLaw(C = 1e7, q = 1/3)
        model = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot;
                                       sliding_law = law, dissipation_melt = false, dissipation_verbose = false)

        run!(SteadyStateSimulation(model, grid, state))

        @test all(isfinite, field_values(model.q))
        @test all(field_values(model.tau_b) .≈ 1e7 .* field_values(model.abs_v_b) .^ (1/3))
        @test all(field_values(model.mdot_total) .>= field_values(model.mdot))
    end

    @testset "KazmierczakHydroModel with N-dependent sliding laws (q, N) coupling" begin
        # PowerPlasticSlidingLaw and RegularizedCoulombSlidingLaw make tau_b depend on N, which is
        # itself downstream of q, so resolve_q! widens its Picard loop to also update N each sweep
        # (see water_flux.jl). Exercise both laws, with and without the (independent) dissipation
        # melt term, and confirm the joint loop produces finite, physically sane results.
        grid = OGRectHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
        mask = ones(5, 5)
        h    = [500.0 - 5.0 * i for i in 1:5, j in 1:5]
        b    = [-100.0 - 2.0 * j for i in 1:5, j in 1:5]

        kappa   = zeros(5, 5)
        abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), 5, 5)
        A_visc  = fill(1e-24, 5, 5)
        mdot    = fill(1e-6, 5, 5)

        for law in (
            PowerPlasticSlidingLaw(c_till = 0.5, q = 1.0, u0 = perYear2perSecond(100.0)),
            RegularizedCoulombSlidingLaw(c_till = 0.5, q = 1/3, u0 = perYear2perSecond(100.0)),
        ), dissipation_melt in (false, true)

            model = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot;
                                           sliding_law = law, dissipation_melt = dissipation_melt,
                                           dissipation_verbose = false, coupling_verbose = false)
            state = HydroState(grid, mask, h, b)
            run!(SteadyStateSimulation(model, grid, state))

            @test all(isfinite, field_values(state.N))
            @test all(isfinite, field_values(model.q))
            @test all(isfinite, field_values(model.tau_b))
            @test all(>=(0.0), field_values(state.N))
            @test all(>=(0.0), field_values(model.tau_b))
            # Frictional heating only adds to the background melt rate.
            @test all(field_values(model.mdot_total) .>= field_values(model.mdot))
        end

        # max_coupling_iters is a hard cap: it must not error even when it cuts the Picard
        # iteration off before convergence.
        model_capped = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot;
                                              sliding_law = PowerPlasticSlidingLaw(c_till = 0.5, q = 1.0, u0 = perYear2perSecond(100.0)),
                                              max_coupling_iters = 1, coupling_verbose = false)
        run!(SteadyStateSimulation(model_capped, grid, HydroState(grid, mask, h, b)))
        @test all(isfinite, field_values(model_capped.q))
    end

    @testset "KazmierczakHydroModel with a flat, zero-flux region" begin
        # Regression test: on real datasets (THWAITES), a few cells beyond the glacier extent
        # have h=b=0 (flat, zero geometric-potential gradient) and mask=0 (zero water flux).
        # update_S_inf! evaluated 0^(negative) * 0^(positive) = Inf * 0 = NaN there, since
        # (1-beta)/alpha < 0 for the default alpha, beta. Build a grid with such a flat/ungrounded
        # region and confirm N stays finite.
        grid = OGRectHydroGrid(10, 10, (0.0, 1000.0), (0.0, 1000.0))
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

        @test all(isfinite, field_values(model.S_inf))
        @test all(isfinite, field_values(state.N))
    end

    @testset "HABHydroModel steady state" begin
        # Regression test: update_N! used to call max(::Array, ::Array) instead of
        # max.(...), which threw a MethodError for any grid larger than a scalar.
        grid = OGRectHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
        mask = ones(5, 5)
        h    = fill(500.0, 5, 5)
        b    = fill(-100.0, 5, 5)
        state = HydroState(grid, mask, h, b)

        model = HABHydroModel(grid)
        sim = SteadyStateSimulation(model, grid, state)
        run!(sim)

        @test all(isfinite, field_values(state.N))
        @test all(>=(0.0), field_values(state.N))
    end

    @testset "load_Kazmierczak bed rheology options" begin
        # Regression test: the :soft branch used to call one(T, Nx, Ny) instead of
        # ones(T, Nx, Ny), which threw a MethodError.
        mktempdir() do dir
            path = joinpath(dir, "synthetic_kazmierczak.mat")
            Nx, Ny = 5, 5
            matwrite(path, Dict(
                "H"     => fill(500.0, Nx, Ny),
                "MASKo" => ones(Nx, Ny),
                "B"     => fill(-100.0, Nx, Ny),
                "ub"    => fill(100.0, Nx, Ny),
                "A"     => fill(1e-24, Nx, Ny),
                "Bmelt" => fill(0.1, Nx, Ny),
                "x"     => collect(0.0:1.0:(Ny - 1)),
                "y"     => collect(0.0:1.0:(Nx - 1)),
            ))

            for bed_rheology in (:hard, :soft, :mixed, :mixed_smooth)
                Nx_out, Ny_out, xlims, ylims, mask, h, b, abs_v_b, A_visc, ṁ, κ =
                    load_Kazmierczak(path; bed_rheology = bed_rheology)
                @test Nx_out == Nx
                @test Ny_out == Ny
                @test all(isfinite, κ)
                if bed_rheology == :hard
                    @test all(==(0.0), κ)
                elseif bed_rheology == :soft
                    @test all(==(1.0), κ)
                end

                # Regression test: `ub` is stored per-year (like `Bmelt`), so `abs_v_b` must come
                # back converted to m/s -- previously it was passed through raw.
                @test all(≈(perYear2perSecond(100.0)), abs_v_b)
            end
        end
    end

    @testset "load_yelmox bed rheology options" begin
        mktempdir() do dir
            path = joinpath(dir, "synthetic_yelmox.nc")
            Nx, Ny = 5, 5

            NCDataset(path, "c") do ds
                defDim(ds, "xc", Nx)
                defDim(ds, "yc", Ny)
                defDim(ds, "layer", 1)

                defVar(ds, "xc", collect(0.0:1.0:(Nx - 1)), ("xc",))
                defVar(ds, "yc", collect(0.0:1.0:(Ny - 1)), ("yc",))
                defVar(ds, "f_ice", ones(Nx, Ny), ("xc", "yc"))
                defVar(ds, "f_grnd", ones(Nx, Ny), ("xc", "yc"))
                defVar(ds, "H_ice", fill(500.0, Nx, Ny), ("xc", "yc"))
                defVar(ds, "z_bed", fill(-100.0, Nx, Ny), ("xc", "yc"))
                defVar(ds, "ux_b", fill(50.0, Nx, Ny), ("xc", "yc"))
                defVar(ds, "uy_b", fill(50.0, Nx, Ny), ("xc", "yc"))
                defVar(ds, "ATT", fill(1e-24, Nx, Ny, 1), ("xc", "yc", "layer"))
                defVar(ds, "bmb", fill(0.1, Nx, Ny), ("xc", "yc"))
            end

            for bed_rheology in (:hard, :soft, :mixed, :mixed_smooth)
                Nx_out, Ny_out, xlims, ylims, mask, h, b, abs_v_b, A_visc, ṁ, κ =
                    load_yelmox(path; bed_rheology = bed_rheology)
                @test Nx_out == Nx
                @test Ny_out == Ny
                @test all(isfinite, κ)
                if bed_rheology == :hard
                    @test all(==(0.0), κ)
                elseif bed_rheology == :soft
                    @test all(==(1.0), κ)
                end

                # Regression test: ux_b/uy_b carry a "units" = "m/yr" attribute in real yelmox
                # restart files, so abs_v_b = sqrt(ux_b^2 + uy_b^2) must come back converted to m/s.
                @test all(≈(perYear2perSecond(sqrt(50.0^2 + 50.0^2))), abs_v_b)

                # Regression test: bmb also carries a "units" = "m/yr" attribute and is an
                # ice-equivalent thickness rate (Yelmo.jl: bmb = -Q_net / (rho_ice * L_ice)), so ṁ
                # must come back converted to m/s and scaled by rho_ice = 917.0, not left raw.
                @test all(≈(perYear2perSecond(-0.1) * 917.0), ṁ)
            end
        end
    end

end

# Reuses field_values/interior from above, so include at top level (unlike shakti_ext_test.jl,
# ArrayHydroGrid has no naming collisions requiring its own module).
include("array_grid_test.jl")

# Own module (see shakti_ext_test.jl) since FastHydrology and Shakti both export `run!`.
include("shakti_ext_test.jl")
