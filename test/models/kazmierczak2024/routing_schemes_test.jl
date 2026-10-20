# Le Brocq et al. (2006) routing schemes, q conversion and dissipation discretisation options.

@testset "Routing schemes (Le Brocq et al. 2006)" begin

    Nx, Ny = 40, 30
    dx = 1e3
    grid = ArrayHydroGrid(Nx, Ny, (0.0, Nx * dx), (0.0, Ny * dx))
    x = [(i - 0.5) * dx for i in 1:Nx, j in 1:Ny]
    y = [(j - 0.5) * dx for i in 1:Nx, j in 1:Ny]
    h = fill(1000.0, Nx, Ny)
    bumpy = -2e-3 .* x .- 1e-3 .* y .+ 20 .* sin.(x ./ 3e3) .* cos.(y ./ 4e3)
    z = zeros(Nx, Ny)
    mdot = fill(1e-6, Nx, Ny)
    G, q_T = mdot .* FastHydrology.KAZMIERCZAK_DEFAULT_L_W, zero(mdot)  # same melt, supplied as geothermal heat
    build(scheme, b; kw...) = (KazmierczakHydroModel(grid, z, z, z .+ 1e-24, G, q_T; coupling_length_kamb86 = 0.0,
                                   dissipation_melt = false, dissipation_verbose = false, routing_scheme = scheme, kw...),
                               HydroState(grid, ones(Nx, Ny), h, b))

    # Volume flux leaving the domain: through edges, plus what cells with nowhere to send it keep.
    function exits(model)
        W = model.routing_tape.w8
        e = 0.0
        for j in 1:Ny, i in 1:Nx
            tot = sum(@view W[:, i, j])
            tot == 0 && (e += model.psi_out[i, j]; continue)
            for d in 1:8
                di, dj = FastHydrology.ROUTE_OFFSETS[d]
                FastHydrology.in_domain(i + di, j + dj, Nx, Ny) || (e += model.psi_out[i, j] * W[d, i, j])
            end
        end
        return e
    end

    @testset "$(nameof(typeof(scheme))) conserves water on a priority-flood-filled bumpy surface" for scheme in (Warner(), Quinn(), Quinn(original = true), Tarboton())
        model, state = build(scheme, bumpy; fill_algorithm = PriorityFloodFill())
        update_steady_state!(model, grid, state)
        W = model.routing_tape.w8
        sums = [sum(@view W[:, i, j]) for i in 1:Nx, j in 1:Ny]
        @test all(s -> s == 0 || isapprox(s, 1; atol = 1e-12), sums)
        # interior cells always have a strictly lower neighbour after priority-flood filling
        @test all(isapprox.(sums[2:end-1, 2:end-1], 1; atol = 1e-12))
        source = sum(mdot) * dx * dx / model.rho_w
        @test isapprox(exits(model), source; rtol = 1e-10)
        @test all(isfinite, model.q) && all(isfinite, state.N)
    end

    @testset "$(nameof(typeof(scheme))) runs" for scheme in (ModifiedTarboton(), GDSTarboton())
        model, state = build(scheme, bumpy)
        update_steady_state!(model, grid, state)
        @test all(isfinite, model.q) && all(isfinite, state.N)
        @test all(>=(0), model.psi_out)
    end

    @testset "Quinn(original = true) weights" begin
        # one cell with a unit drop to every neighbour on a square grid: cardinal weight
        # tan(beta) L = (1/dx)(0.5 dx) = 0.5, diagonal (1/(sqrt(2) dx))(sqrt(2)/4 dx) = 0.25
        W = zeros(8, 3, 3)
        phi = zeros(3, 3); phi[2, 2] = 1.0
        FastHydrology.cell_weights!(W, Quinn(original = true), 2, 2, phi, phi, phi, 3, 3, 1e3, 1e3)
        @test isapprox(W[1:4, 2, 2], fill(1 / 6, 4); rtol = 1e-12)
        @test isapprox(W[5:8, 2, 2], fill(1 / 12, 4); rtol = 1e-12)
        FastHydrology.cell_weights!(fill!(W, 0.0), Quinn(), 2, 2, phi, phi, phi, 3, 3, 1e3, 1e3)
        @test isapprox(W[:, 2, 2], fill(1 / 8, 8); rtol = 1e-12)
    end

    @testset "QFromFaceAverage on a uniform x-slope" begin
        plane = -1e-3 .* x
        m_out, s_out = build(Warner(), plane; q_conversion = QFromOutflow())
        m_face, s_face = build(Warner(), plane; q_conversion = QFromFaceAverage())
        update_steady_state!(m_out, grid, s_out)
        update_steady_state!(m_face, grid, s_face)
        src = mdot[1] * dx * dx / m_out.rho_w
        # outflow q describes the downstream face; the face average is half a cell's source lower
        @test isapprox(m_face.q[2:end-1, :], m_out.q[2:end-1, :] .- src / (2dx); rtol = 1e-10)
    end

    @testset "FaceDissipation is positive for downhill flow" begin
        model, state = KazmierczakHydroModel(grid, z, z, z .+ 1e-24, G, q_T; coupling_length_kamb86 = 0.0, dissipation_verbose = false,
                                             routing_scheme = Warner(), dissipation_discretization = FaceDissipation()), HydroState(grid, ones(Nx, Ny), h, -1e-3 .* x)
        update_steady_state!(model, grid, state)
        @test all(>=(0), model.routing_tape.diss)
        @test sum(model.routing_tape.diss) > 0
    end

    @testset "gradient_discretization" begin
        # default follows the routing scheme
        @test build(Warner(), bumpy)[1].gradient_discretization isa FaceGradient
        @test build(GDSWarner(), bumpy)[1].gradient_discretization isa CentredGradient
        @test build(Quinn(), bumpy)[1].gradient_discretization isa CentredGradient

        # abs_grad_phi0 is the face (resp. centred) magnitude of the true potential; the components
        # (flow direction) are the same under both
        mf, sf = build(Warner(), bumpy)
        mc, sc = build(Warner(), bumpy; gradient_discretization = CentredGradient())
        update_steady_state!(mf, grid, sf)
        update_steady_state!(mc, grid, sc)
        ref = zeros(Nx, Ny)
        face_gradient_magnitude!(grid, ref, mf.phi0)
        @test mf.abs_grad_phi0 == ref
        gx, gy = zeros(Nx, Ny), zeros(Nx, Ny)
        minus_gradient_x!(grid, gx, mc.phi0); minus_gradient_y!(grid, gy, mc.phi0)
        @test mc.abs_grad_phi0 ≈ hypot.(gx, gy)
        @test mf.minus_grad_phi0_sx == mc.minus_grad_phi0_sx && mf.minus_grad_phi0_sy == mc.minus_grad_phi0_sy
        face_gradient_magnitude!(grid, ref, mf.phi0_filled)
        @test mf.abs_grad_phi0_s == ref
        @test mc.abs_grad_phi0_s ≈ hypot.(mc.minus_grad_phi0_sx, mc.minus_grad_phi0_sy)
        # averaging squares >= squaring averages
        @test all(mf.abs_grad_phi0 .>= mc.abs_grad_phi0 .- 1e-12)
        @test all(isfinite, mf.q) && all(isfinite, sf.N)

        # also selectable for the GDS schemes (unsmoothed with coupling length 0)
        mg, sg = build(GDSWarner(), bumpy; gradient_discretization = FaceGradient())
        update_steady_state!(mg, grid, sg)
        face_gradient_magnitude!(grid, ref, mg.phi0)
        @test mg.abs_grad_phi0 == ref
    end

    @testset "invalid combinations are rejected" begin
        @test_throws ArgumentError build(Warner(), bumpy; psi_out_algorithm = RecursivePsiOut())
        @test_throws ArgumentError build(Quinn(), bumpy; q_conversion = QFromFaceAverage())
        @test_throws ArgumentError build(Tarboton(), bumpy; dissipation_melt = true, dissipation_discretization = FaceDissipation())
    end

    @testset "StaggeredFriction" begin
        u = perYear2perSecond(100.0)
        vb = fill(u, Nx, Ny)
        tau = fill(5e4, Nx, Ny)
        # uniform flow along x on the C-grid: every face carries the same velocity, so both staggered
        # forms must reduce to tau * |u| exactly
        for quadrature in (false, true)
            fd = StaggeredFriction(fill(u, Nx, Ny), zeros(Nx, Ny); quadrature)
            mt = zeros(Nx, Ny)
            FastHydrology.staggered_friction_kernel!(mt, tau, vb, fd.ux, fd.uy, Nx, Ny, fd.u_floor, fd.quadrature)
            @test all(isapprox.(mt, 5e4 * u; rtol = 1e-12))
        end
        # face form: each face heat beta_face*u_face^2 >= 0, and the domain total equals the total face work
        ux = u .* (1 .+ 0.5 .* sin.(x ./ 5e3)); uy = 0.3u .* cos.(y ./ 7e3)
        vb2 = hypot.(ux, uy)
        mt = zeros(Nx, Ny)
        FastHydrology.staggered_friction_kernel!(mt, tau, vb2, ux, uy, Nx, Ny, perYear2perSecond(1e-3), false)
        @test all(>=(0), mt)
        # model plumbing: size check and a full solve with an N-dependent law
        @test_throws ArgumentError KazmierczakHydroModel(grid, z, vb, z .+ 1e-24, G, q_T; coupling_length_kamb86 = 0.0,
                                                         friction_discretization = StaggeredFriction(zeros(3, 3), zeros(3, 3)))
        model = KazmierczakHydroModel(grid, z, vb2, z .+ 1e-24, G, q_T; coupling_length_kamb86 = 0.0, dissipation_verbose = false, qN_verbose = false,
                                      sliding_law = RegularizedCoulombSlidingLaw(c_till = 0.5), friction_discretization = StaggeredFriction(ux, uy))
        state = HydroState(grid, ones(Nx, Ny), h, bumpy)
        update_steady_state!(model, grid, state)
        @test all(isfinite, model.q) && all(isfinite, state.N)
    end


    @testset "freeze_on_capacity! closes the routing balance ($(nameof(typeof(scheme))))" for (scheme, surf) in ((Warner(), bumpy), (Quinn(), bumpy), (Tarboton(), bumpy),
                                                  # GDS routing has flow cycles on the bumpy surface (cut by the router), so the plane
                                                  (GDSWarner(), -2e-3 .* x .- 1e-3 .* y))
        # water from above [kg/m2/s], varying in space, comparable to the melt (1e-3 kg/m2/s)
        i_eb = 5e-4 .* (1 .+ sin.(x ./ 5e3) .^ 2)
        model, state = build(scheme, surf; fill_algorithm = PriorityFloodFill(), i_eb = i_eb)
        update_steady_state!(model, grid, state)
        C = freeze_on_capacity!(zeros(Nx, Ny), model, grid, state)
        @test all(>=(0), C)
        # C = (rho_w Psi_in/(dx dy) + i_eb)/rho_i, and psi_out = Psi_in + own source wherever the
        # max(0, .) clamp did not bite
        psi_in = (C .* model.rho_i .- i_eb) .* (dx * dx / model.rho_w)
        own    = model.mdot_total .* (dx * dx / model.rho_w)
        @test isapprox(model.psi_out, psi_in .+ own; rtol = 1e-10)
        # cells nothing flows into can freeze only the water from above
        @test any(C .== i_eb ./ model.rho_i)
        @test all(C .>= i_eb ./ model.rho_i .* (1 - 1e-14))
    end

    # A sloping grounded patch padded with h = b = 0 cells (as around a rasterised mesh): phi0 drops from
    # ~9 MPa to 0 across the patch edge. None of that jump may reach the smoothing, the true gradient
    # or the dissipation.
    @testset "edge of the grounded ice" begin
        ip, jp = 6:35, 6:25
        inside = [(i in ip) && (j in jp) for i in 1:Nx, j in 1:Ny]
        hp = ifelse.(inside, 1000.0, 0.0)
        slope = (-2e-3, -1e-3)
        bp = ifelse.(inside, slope[1] .* x .+ slope[2] .* y, 0.0)
        mp = Float64.(inside)
        exact = 1000.0 * 9.81 * hypot(slope...)      # |grad(rho_w g b)| of the plane
        pbuild(; kw...) = (KazmierczakHydroModel(grid, z, z, z .+ 1e-24, G, q_T; dissipation_verbose = false, kw...),
                           HydroState(grid, mp, hp, bp))
        rim = inside .& .![all(inside[clamp(i + di, 1, Nx), clamp(j + dj, 1, Ny)] for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                            for i in 1:Nx, j in 1:Ny]

        # the weight: the grounded mask here; ice outside the mask counts when grounded by flotation
        m, s = pbuild(coupling_length_kamb86 = 0.0, dissipation_melt = false)
        update_phi0!(m, grid, s)
        @test m.phi_weight == mp
        h2, b2 = copy(hp), copy(bp)
        h2[1, 1], b2[1, 1] = 100.0, 0.0       # grounded ice outside the mask
        h2[2, 1], b2[2, 1] = 100.0, -500.0    # floating: 917 * 100 < 1028 * 500
        s2 = HydroState(grid, mp, h2, b2)
        update_phi0!(m, grid, s2)
        @test m.phi_weight[1, 1] == 1 && m.phi_weight[2, 1] == 0 && m.phi_weight[3, 1] == 0
        mm, _ = pbuild(coupling_length_kamb86 = 0.0, dissipation_melt = false, potential_weight = GroundedMask())
        update_phi0!(mm, grid, s2)
        @test mm.phi_weight == mp

        # true gradient: the plane's slope on every grounded cell, rim included, for both stencils
        for gd in (FaceGradient(), CentredGradient())
            m, s = pbuild(coupling_length_kamb86 = 0.0, dissipation_melt = false, gradient_discretization = gd)
            update_steady_state!(m, grid, s)
            @test all(isapprox.(m.abs_grad_phi0[inside], exact; rtol = 1e-8))
        end

        # smoothing: an unweighted average of phi0 over the padding has an interior maximum (a
        # spurious divide); the weighted one does not
        m, s = pbuild(coupling_length_kamb86 = 10.0, dissipation_melt = false)
        update_steady_state!(m, grid, s)
        interior_max(phi) = count(CartesianIndices(phi)) do c
            i, j = Tuple(c)
            all(inside[i + di, j + dj] for di in -1:1, dj in -1:1 if FastHydrology.in_domain(i + di, j + dj, Nx, Ny)) &&
                2 <= i <= Nx - 1 && 2 <= j <= Ny - 1 &&
                phi[i, j] > maximum(phi[i + di, j + dj] for di in -1:1, dj in -1:1 if (di, dj) != (0, 0))
        end
        kernel = FastHydrology.coupling_kernel(m, grid, 1000.0)
        @test size(kernel, 1) > 1
        plain = zeros(Nx, Ny)
        convolve!(grid, plain, m.phi0, kernel)
        @test interior_max(plain) > 0
        @test interior_max(m.phi0_filled) == 0
        @test all(isfinite, m.q) && all(isfinite, s.N)

        # GDS: the weighted smoothing of the components of a plane is the plane's gradient on every
        # grounded cell; the unweighted one carries the jump at the edge a kernel radius inward
        m, s = pbuild(coupling_length_kamb86 = 10.0, dissipation_melt = false, routing_scheme = GDSWarner())
        update_steady_state!(m, grid, s)
        sx, sy = -1000.0 * 9.81 .* slope       # -grad(rho_w g b) of the plane
        @test all(isapprox.(m.minus_grad_phi0_sx[inside], sx; rtol = 1e-8))
        @test all(isapprox.(m.minus_grad_phi0_sy[inside], sy; rtol = 1e-8))
        kernel = FastHydrology.coupling_kernel(m, grid, max(masked_mean(grid, m.h, s.mask), 10.0))
        @test size(kernel, 1) > 1
        convolve!(grid, plain, m.minus_grad_phi0_x, kernel)
        @test maximum(abs.(plain[inside] .- sx)) > abs(sx)
        @test all(isfinite, m.q) && all(isfinite, s.N)

        # face dissipation: no energy for the drop to the padding on the exit faces
        m, s = pbuild(coupling_length_kamb86 = 0.0, dissipation_melt = true)
        update_steady_state!(m, grid, s)
        d = m.routing_tape.diss
        @test all(>=(0), d[inside])
        @test maximum(d[rim]) < 10 * maximum(d[inside .& .!rim])

        # an isolated grounded cell (no neighbour inside the weight) keeps a finite gradient and N
        mi = copy(mp); hi = copy(hp)
        mi[2, 2] = 1.0; hi[2, 2] = 1000.0
        m = KazmierczakHydroModel(grid, z, z, z .+ 1e-24, G, q_T; coupling_length_kamb86 = 0.0, dissipation_verbose = false)
        s = HydroState(grid, mi, hi, bp)
        update_steady_state!(m, grid, s)
        @test m.abs_grad_phi0[2, 2] > 0
        @test all(isfinite, s.N)
    end

end
