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
