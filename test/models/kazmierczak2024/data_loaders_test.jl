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
                "G"     => fill(0.1, Nx, Ny),
                # three levels, bed at zeta = 1: q_T = 2.1 * (-2 - (-10)) / (500 * 0.5)
                "zeta"  => [0.0 0.5 1.0],
                "tmp"   => cat(fill(-20.0, Nx, Ny), fill(-10.0, Nx, Ny), fill(-2.0, Nx, Ny); dims = 3),
                "taudxy" => fill(5e4, Nx, Ny),
                "x"     => collect(0.0:1.0:(Ny - 1)),
                "y"     => collect(0.0:1.0:(Nx - 1)),
            ))

            for bed_rheology in (:hard, :soft, :mixed, :mixed_smooth)
                Nx_out, Ny_out, xlims, ylims, mask, h, b, abs_v_b, A_visc, G, q_T, tau_b, κ =
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

                # Regression test: `A` is also stored per-year (confirmed against KORI-ULB's own
                # SchoofWaterFarField.m, which divides A by secperyear alongside ub), so A_visc must
                # come back converted to Pa^-n s^-1 -- previously it was passed through raw.
                @test all(≈(perYear2perSecond(1e-24)), A_visc)

                # Terms of the melt rate: q_T from the basal temperature gradient, tau_b = taudxy
                @test all(≈(0.1), G)
                @test all(≈(2.1 * 8.0 / 250.0), q_T)
                @test all(≈(5e4), tau_b)
            end

            # KORI-ULB's own melt rate, for comparison only: per year, water equivalent
            @test all(≈(perYear2perSecond(0.1) * 1000), load_Kazmierczak_melt(path))
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
                defVar(ds, "Q_geo", fill(60.0, Nx, Ny), ("xc", "yc"))
                defVar(ds, "Q_ice_b", fill(50.0, Nx, Ny), ("xc", "yc"))   # mW m^-2, as Yelmo writes it
                defVar(ds, "taub", fill(2e4, Nx, Ny), ("xc", "yc"))
            end

            for bed_rheology in (:hard, :soft, :mixed, :mixed_smooth)
                Nx_out, Ny_out, xlims, ylims, mask, h, b, abs_v_b, A_visc, G, q_T, tau_b, κ =
                    load_yelmox(path; bed_rheology = bed_rheology)
                @test Nx_out == Nx
                @test Ny_out == Ny
                @test all(isfinite, κ)
                if bed_rheology == :hard
                    @test all(==(0.0), κ)
                elseif bed_rheology == :soft
                    @test all(==(1.0), κ)
                end

                # Regression test: xc/yc carry a "units" = "km" attribute in real yelmox restart
                # files, so xlims/ylims must come back converted to meters (dx/dy here is 1.0 in the
                # synthetic file's raw units, so it must come back as 1000.0 m, not left at O(1) --
                # a scale of "tens" or less would mean the km -> m conversion was skipped).
                dx_implied = (xlims[2] - xlims[1]) / Nx
                dy_implied = (ylims[2] - ylims[1]) / Ny
                @test dx_implied ≈ 1000.0
                @test dy_implied ≈ 1000.0

                # Regression test: ux_b/uy_b carry a "units" = "m/yr" attribute in real yelmox
                # restart files, so abs_v_b = sqrt(ux_b^2 + uy_b^2) must come back converted to m/s.
                @test all(≈(perYear2perSecond(sqrt(50.0^2 + 50.0^2))), abs_v_b)

                # Regression test: ATT (Glen's law rate factor) is documented in [1/yr / Pa^3] by
                # Yelmo.jl's own rate_factor.jl constants, so A_visc must come back converted to
                # Pa^-n s^-1.
                @test all(≈(perYear2perSecond(1e-24)), A_visc)

                # Terms of the melt rate. Regression test: Q_geo and Q_ice_b are both in mW m^-2 in
                # Yelmo, so both come back in W/m^2 (Q_ice_b used to be passed through raw).
                @test all(≈(0.06), G)
                @test all(≈(0.05), q_T)
                @test all(≈(2e4), tau_b)
            end
        end
    end
