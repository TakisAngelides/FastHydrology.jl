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
            end
        end
    end

end
