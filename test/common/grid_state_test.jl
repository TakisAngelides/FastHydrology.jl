    @testset "Grid and state construction" begin
        grid = ArrayHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
        @test grid.Nx == 5
        @test grid.Ny == 5

        mask = fill(GROUNDED, 5, 5)
        h    = fill(500.0, 5, 5)
        b    = fill(-100.0, 5, 5)
        state = HydroState(grid, HydroMask(mask), h, b)

        @test all(isfinite, state.N)
        @test all(isfinite, state.W)
    end

    @testset "Hydrology mask" begin
        grid = ArrayHydroGrid(2, 4, (0.0, 200.0), (0.0, 400.0))
        # Shakti.jl's codes
        @test (GROUNDED, OCEAN, LAND, OTHER_BASIN) == (0.0, 1.0, 2.0, 3.0)
        codes = [GROUNDED OCEAN LAND OTHER_BASIN; GROUNDED GROUNDED OCEAN OCEAN]
        state = HydroState(grid, HydroMask(codes), fill(500.0, 2, 4), fill(-100.0, 2, 4))
        @test state.mask == codes
        # a matrix of the codes, checked on every assignment
        m = HydroMask(copy(codes))
        @test m isa AbstractMatrix && m == codes && m[1, 2] == OCEAN
        @test (m .== GROUNDED) == (codes .== GROUNDED)
        m[2, :] .= LAND
        @test all(==(LAND), m[2, :]) && m.codes[2, 1] == LAND
        @test_throws ArgumentError (m[1, 1] = 4.0)
        @test_throws ArgumentError HydroMask([GROUNDED 4.0])
        @test_throws ArgumentError HydroMask([GROUNDED -1.0])
        @test_throws ArgumentError HydroMask([0.5 1.0])
        # a plain array is refused, so an old 0/1 mask (1 = grounded) is never read as ocean
        @test_throws ArgumentError HydroState(grid, ones(2, 4), fill(500.0, 2, 4), fill(-100.0, 2, 4))
        @test_throws ArgumentError HydroState(grid, HydroMask(fill(GROUNDED, 3, 4)), fill(500.0, 2, 4), fill(-100.0, 2, 4))
        @test mask_field(ones(2, 4), HydroMask(codes), NaN)[1, 1] == 1 && isnan(mask_field(ones(2, 4), HydroMask(codes), NaN)[1, 2])

        # classify_mask: grounded ice, floating ice, open water, ice-free land, ice outside the basin
        h = [1000.0 100.0 0.0   0.0  500.0]
        b = [-200.0 -500.0 -50.0 30.0 100.0]
        @test classify_mask(h, b) == [GROUNDED OCEAN OCEAN LAND GROUNDED]
        @test classify_mask(h, b; grounded = [true false false false false]) == [GROUNDED OCEAN OCEAN LAND OTHER_BASIN]
        # the host's grounded mask overrides flotation; closed overrides everything
        @test classify_mask(h, b; grounded = [false true false false false], closed = [false false false true false]) ==
              [OTHER_BASIN GROUNDED OCEAN OTHER_BASIN OTHER_BASIN]
        @test_throws ArgumentError classify_mask(h, b[:, 1:4])
        @test_throws ArgumentError classify_mask(h, b; grounded = trues(1, 4))
    end
