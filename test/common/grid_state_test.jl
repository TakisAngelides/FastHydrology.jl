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
