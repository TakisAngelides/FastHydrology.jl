# Grid-interface checks on ArrayHydroGrid: convolution, edge gradients, routing geometry and
# potential filling, exercised directly rather than through a full model solve.

@testset "ArrayHydroGrid" begin

    @testset "convolve! mass conservation" begin
        # A normalized kernel convolved with "replicate" (edge-extended) boundary padding must
        # conserve the sum of the field (nothing enters or leaves through a replicated edge).
        Nx, Ny = 5, 5
        src = [Float64(i + j) for i in 1:Nx, j in 1:Ny]

        k = 3 # kernel radius comparable to the grid size, like the model's default longcoupwater on a small domain
        kernel = [max(0.0, 1.0 - sqrt((i - k - 1)^2 + (j - k - 1)^2) / k) for i in 1:2k+1, j in 1:2k+1]
        kernel ./= sum(kernel)

        grid_a = ArrayHydroGrid(Nx, Ny, (0.0, 500.0), (0.0, 500.0))
        dest_a = zeros(Nx, Ny)
        convolve!(grid_a, dest_a, src, kernel)

        @test isapprox(sum(dest_a), sum(src); rtol = 1e-10)
    end

    @testset "minus_gradient: edges use full one-sided gradient" begin
        # A linear field must give the exact gradient everywhere -- including the domain-edge rows and
        # columns, which used to get half of it (edge-replicated ghost cell with a 2dx denominator).
        Nx, Ny, dx, dy = 6, 5, 200.0, 500.0
        f = [3.0 * (i * dx) - 2.0 * (j * dy) for i in 1:Nx, j in 1:Ny]
        grid_a = ArrayHydroGrid(Nx, Ny, (0.0, Nx * dx), (0.0, Ny * dy))

        gx_a, gy_a = zeros(Nx, Ny), zeros(Nx, Ny)
        minus_gradient_x!(grid_a, gx_a, f)
        minus_gradient_y!(grid_a, gy_a, f)
        @test all(isapprox.(gx_a, -3.0))
        @test all(isapprox.(gy_a, 2.0))
    end

    @testset "routing_weight matches the face-flux geometry for dx != dy" begin
        # Diagonal flow (sx == sy) on dx = 4dy: 1/5 of the flux leaves through the x-face, 4/5 through
        # the y-face (face widths dy and dx). The weights sum to 1 and reduce to 1/2, 1/2 for dx == dy.
        dx, dy = 2000.0, 500.0
        wx = FastHydrology.routing_weight(1.0, 1.0, 1, 0, dx, dy, 0.0)
        wy = FastHydrology.routing_weight(1.0, 1.0, 0, 1, dx, dy, 0.0)
        @test wx ≈ 0.2
        @test wy ≈ 0.8
        @test FastHydrology.routing_weight(1.0, 1.0, -1, 0, dx, dy, 0.0) < 0 # upstream side gets nothing
        @test FastHydrology.routing_weight(1.0, 1.0, 1, 0, 1.0, 1.0, 0.0) ≈ 0.5
    end

    @testset "potential filling affects routing but not N's potential" begin
        Nx, Ny = 9, 9
        h = [600.0 - 8.0 * i for i in 1:Nx, j in 1:Ny]
        b = fill(-100.0, Nx, Ny)
        h[5, 5] -= 60.0 # a pit in the potential that filling will raise
        grid = ArrayHydroGrid(Nx, Ny, (0.0, 9000.0), (0.0, 9000.0))
        state = HydroState(grid, ones(Nx, Ny), h, b)
        model = KazmierczakHydroModel(grid, zeros(Nx, Ny), fill(1e-6, Nx, Ny), fill(1e-24, Nx, Ny), fill(1e-6 * FastHydrology.KAZMIERCZAK_DEFAULT_L_W, Nx, Ny), zeros(Nx, Ny);
                                       coupling_length_kamb86 = 0.0)
        FastHydrology.update_phi0!(model, grid, state)
        true_phi0 = copy(model.phi0)
        FastHydrology.potential_filling!(model, grid, state)
        @test model.phi0 == true_phi0
        @test model.phi0_filled[5, 5] > true_phi0[5, 5]
    end

end
