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
        # Default water_thickness_algorithm is DarcyWeisbachThickness(), clamped to [Wmin, Wmax];
        # the full per-algorithm behaviour (including the two unclamped closures) is checked below.
        @test all(w -> model.Wmin <= w <= model.Wmax, field_values(state.W))
    end

    @testset "water_thickness_algorithm keyword selects the update_W! closure" begin
        grid = OGRectHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
        mask = ones(5, 5)
        h    = [500.0 - 5.0 * i for i in 1:5, j in 1:5]
        b    = [-100.0 - 2.0 * j for i in 1:5, j in 1:5]

        kappa   = zeros(5, 5)
        abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), 5, 5)
        A_visc  = fill(1e-24, 5, 5)
        mdot    = fill(1e-6, 5, 5)

        # ConduitThickness/ArealConduitThickness are unclamped (they report real conduit-scale
        # depths, not a thin-sheet approximation), so only finiteness/non-negativity applies.
        # DarcyWeisbachThickness/LaminarThickness both represent a thin sheet and are clamped to
        # [Wmin, Wmax] -- see AbstractWaterThicknessAlgorithm's docstring in model.jl.
        for (algorithm, clamped) in (
            (ConduitThickness(), false),
            (ArealConduitThickness(), false),
            (DarcyWeisbachThickness(), true),
            (LaminarThickness(), true),
        )
            state = HydroState(grid, mask, h, b)
            model = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot;
                                           water_thickness_algorithm = algorithm, dissipation_verbose = false)
            sim = SteadyStateSimulation(model, grid, state)
            run!(sim)

            @test all(isfinite, field_values(state.W))
            @test all(>=(0.0), field_values(state.W))
            if clamped
                @test all(w -> model.Wmin <= w <= model.Wmax, field_values(state.W))
            end
        end
    end

    @testset "KazmierczakHydroModel q clamp is per-year 1e5, not raw 1e5" begin
        # Regression test: KORI-ULB's own SubWaterFlux.m clamps its per-year-native flw at
        # 1e5 m2/yr (confirmed against a real KORI-ULB output file's flw field, which is genuinely
        # pinned at exactly 1e5 in the fastest-draining cells). q here is SI (m2/s), so the bare
        # literal 1e5 used to make this clamp a no-op (~3.16e7x too permissive to ever bind). Force
        # an extreme mdot so q would blow well past perYear2perSecond(1e5) unclamped, and check it's
        # actually capped there.
        grid = OGRectHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
        mask = ones(5, 5)
        h    = [500.0 - 5.0 * i for i in 1:5, j in 1:5]
        b    = [-100.0 - 2.0 * j for i in 1:5, j in 1:5]
        state = HydroState(grid, mask, h, b)

        kappa   = zeros(5, 5)
        abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), 5, 5)
        A_visc  = fill(1e-24, 5, 5)
        mdot    = fill(1.0, 5, 5)  # extreme -- forces the routing algorithm well past the clamp
        model = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot; dissipation_verbose = false)

        run!(SteadyStateSimulation(model, grid, state))

        q_max = perYear2perSecond(1e5)
        @test all(<=(q_max), field_values(model.q))
        @test any(>=(q_max - 1e-12), field_values(model.q))
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

    @testset "update_psi_out_iterative! matches recursive update_psi_out!" begin
        # update_psi_out_iterative! (water_flux.jl) is a stack-based rewrite of the recursive
        # accumulate_psi_out!/update_psi_out! flow-routing algorithm, meant to be a drop-in
        # equivalent (same visitation order: outer j-then-i loop, same 4-direction neighbour order),
        # not just a numerically-close one. Both write psi_out only for mask == 1 cells, in the same
        # per-cell 4-term summation order, so results should match exactly (not just approximately).
        # A roughly circular grounded region (rather than a filled rectangle) gives every cell a
        # different number/arrangement of grounded neighbours, exercising the boundary- and
        # masked-neighbour branches that a uniform mask never reaches.
        grid = OGRectHydroGrid(12, 12, (0.0, 1200.0), (0.0, 1200.0))
        mask = [((i - 6)^2 + (j - 7)^2 <= 25) ? 1.0 : 0.0 for i in 1:12, j in 1:12]
        h    = [500.0 - 5.0 * i - 3.0 * j for i in 1:12, j in 1:12]
        b    = [-100.0 - 2.0 * j + 1.5 * i for i in 1:12, j in 1:12]
        state = HydroState(grid, mask, h, b)

        kappa   = zeros(12, 12)
        abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), 12, 12)
        A_visc  = fill(1e-24, 12, 12)
        # A mix of melt (positive) and net-refreezing (negative) source cells, not a uniform
        # positive mdot: accumulate_psi_out!'s max_psi_out_calls cap-trip branch returns before its
        # final `max(0.0, psi_out)` clamp, so a cell cut off there can be left negative if its local
        # mdot_total is negative -- a uniform positive mdot never exercises that path and would let
        # update_psi_out_iterative! clamp there (silently diverging) without this test catching it.
        mdot    = [isodd(i + j) ? -1e-6 : 1e-6 for i in 1:12, j in 1:12]

        for max_psi_out_calls in (50_000, 5) # 5 forces the safety cap to bind mid-sweep
            model = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot; max_psi_out_calls)
            run!(SteadyStateSimulation(model, grid, state))

            psi_out_recursive = copy(field_values(model.psi_out))

            update_psi_out_iterative!(model, grid, state)
            psi_out_iterative = field_values(model.psi_out)

            @test psi_out_iterative[mask .== 1] == psi_out_recursive[mask .== 1]
        end
    end

    @testset "KazmierczakHydroModel psi_out_algorithm keyword dispatches recursive vs iterative" begin
        # End-to-end (not just a single psi_out sweep): route_psi_out! (water_flux.jl) dispatches on
        # model.psi_out_algorithm, set via the psi_out_algorithm keyword (RecursivePsiOut() by
        # default). Build two otherwise-identical models differing only in that keyword and confirm
        # a full run! (including the dissipation-melt Picard loop, so route_psi_out! is called
        # several times) converges to the same q and N.
        grid = OGRectHydroGrid(12, 12, (0.0, 1200.0), (0.0, 1200.0))
        mask = [((i - 6)^2 + (j - 7)^2 <= 25) ? 1.0 : 0.0 for i in 1:12, j in 1:12]
        h    = [500.0 - 5.0 * i - 3.0 * j for i in 1:12, j in 1:12]
        b    = [-100.0 - 2.0 * j + 1.5 * i for i in 1:12, j in 1:12]

        kappa   = zeros(12, 12)
        abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), 12, 12)
        A_visc  = fill(1e-24, 12, 12)
        mdot    = [isodd(i + j) ? -1e-6 : 1e-6 for i in 1:12, j in 1:12]

        model_recursive = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot; psi_out_algorithm = RecursivePsiOut())
        model_iterative = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot; psi_out_algorithm = IterativePsiOut())

        @test model_recursive.psi_out_algorithm isa RecursivePsiOut
        @test model_iterative.psi_out_algorithm isa IterativePsiOut

        state_recursive = HydroState(grid, mask, h, b)
        state_iterative = HydroState(grid, mask, h, b)

        run!(SteadyStateSimulation(model_recursive, grid, state_recursive))
        run!(SteadyStateSimulation(model_iterative, grid, state_iterative))

        @test field_values(model_iterative.q)[mask .== 1] == field_values(model_recursive.q)[mask .== 1]
        @test field_values(state_iterative.N)[mask .== 1] == field_values(state_recursive.N)[mask .== 1]
    end

    @testset "TopologicalPsiOut matches RecursivePsiOut on acyclic (idealized) grids" begin
        # TopologicalPsiOut (Kahn's algorithm over the flow-direction dependency graph) is only exact
        # when that graph is genuinely acyclic -- true on smooth, monotonic synthetic grids like this
        # one, but NOT reliably true on real ice-sheet data (confirmed by explicit cycle detection on
        # a real Thwaites-2km dataset, at any longcoupwater -- see TopologicalPsiOut's docstring in
        # model.jl). This test only establishes exactness on the idealized case; it is not evidence
        # that TopologicalPsiOut is safe on real data, which is exactly why it errors by default
        # (allow_cycles = false) rather than silently degrading -- see the dedicated cycle-detection
        # testset below for that safety mechanism itself.
        grid = OGRectHydroGrid(12, 12, (0.0, 1200.0), (0.0, 1200.0))
        mask = [((i - 6)^2 + (j - 7)^2 <= 25) ? 1.0 : 0.0 for i in 1:12, j in 1:12]
        h    = [500.0 - 5.0 * i - 3.0 * j for i in 1:12, j in 1:12]
        b    = [-100.0 - 2.0 * j + 1.5 * i for i in 1:12, j in 1:12]

        kappa   = zeros(12, 12)
        abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), 12, 12)
        A_visc  = fill(1e-24, 12, 12)
        mdot    = [isodd(i + j) ? -1e-6 : 1e-6 for i in 1:12, j in 1:12]

        for longcoupwater in (5.0, 0.0) # the model's default (smoothing on) and smoothing off
            model_recursive   = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot; psi_out_algorithm = RecursivePsiOut(), longcoupwater)
            model_topological = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot; psi_out_algorithm = TopologicalPsiOut(), longcoupwater)

            state_recursive   = HydroState(grid, mask, h, b)
            state_topological = HydroState(grid, mask, h, b)

            run!(SteadyStateSimulation(model_recursive, grid, state_recursive))
            run!(SteadyStateSimulation(model_topological, grid, state_topological))

            @test isapprox(field_values(model_topological.q)[mask .== 1], field_values(model_recursive.q)[mask .== 1]; rtol = 1e-10)
            @test isapprox(field_values(state_topological.N)[mask .== 1], field_values(state_recursive.N)[mask .== 1]; rtol = 1e-10)
        end
    end

    @testset "TopologicalPsiOut cycle detection" begin
        # Deterministic unit test of the cycle-safety mechanism itself: real cycles in the
        # flow-direction graph are a real (if data-dependent) occurrence -- see the testset above --
        # but constructing one from legitimate h/b inputs isn't straightforward, so this directly
        # forces a mutual edge between two adjacent cells by overwriting their post-smoothing gradient
        # fields after the normal update_q! pre-routing steps have run, then calls
        # update_psi_out_topological! on that doctored state.
        grid = OGRectHydroGrid(5, 5, (0.0, 500.0), (0.0, 500.0))
        mask = ones(5, 5)
        h    = [500.0 - 5.0 * i for i in 1:5, j in 1:5]
        b    = [-100.0 - 2.0 * j for i in 1:5, j in 1:5]
        state = HydroState(grid, mask, h, b)

        kappa   = zeros(5, 5)
        abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), 5, 5)
        A_visc  = fill(1e-24, 5, 5)
        mdot    = fill(1e-6, 5, 5)

        model = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot; dissipation_verbose = false)
        FastHydrology.update_phi0!(model, grid, state)
        FastHydrology.potential_filling!(model, grid, state)
        FastHydrology.update_potential_gradients!(model, grid)
        FastHydrology.update_smoothed_potential_gradients!(model, grid, state)
        FastHydrology.update_tau_b!(model, state, model.sliding_law)
        model.mdot_total .= model.mdot

        # Force cells (2,2) and (3,2) to flow into each other: (2,2)'s gradient points toward (3,2)
        # (+x direction, i.e. minus_grad_phi0_sx > 0) and (3,2)'s gradient points back toward (2,2)
        # (-x direction), each with zero y-component so only the x-direction test matters.
        interior(model.minus_grad_phi0_sx, :, :, 1)[2, 2] = 1.0
        interior(model.minus_grad_phi0_sy, :, :, 1)[2, 2] = 0.0
        interior(model.abs_grad_phi0_s, :, :, 1)[2, 2] = 1.0
        interior(model.minus_grad_phi0_sx, :, :, 1)[3, 2] = -1.0
        interior(model.minus_grad_phi0_sy, :, :, 1)[3, 2] = 0.0
        interior(model.abs_grad_phi0_s, :, :, 1)[3, 2] = 1.0

        @test_throws ErrorException FastHydrology.update_psi_out_topological!(model, grid, state, false)

        # allow_cycles = true: same doctored (2,2)<->(3,2) cycle, but this time it should complete
        # (with a warning, not an error) and leave every other cell -- outside the forced cycle --
        # finite, since only (2,2)/(3,2) themselves are inside it.
        @test_logs (:warn, r"cycle") FastHydrology.update_psi_out_topological!(model, grid, state, true)
        @test all(isfinite, field_values(model.psi_out))
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
                "G"     => fill(0.1, Nx, Ny),
                "x"     => collect(0.0:1.0:(Ny - 1)),
                "y"     => collect(0.0:1.0:(Nx - 1)),
            ))

            for bed_rheology in (:hard, :soft, :mixed, :mixed_smooth)
                Nx_out, Ny_out, xlims, ylims, mask, h, b, abs_v_b, A_visc, G, q_T, ṁ, κ =
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
                defVar(ds, "Q_geo", fill(60.0, Nx, Ny), ("xc", "yc"))
                defVar(ds, "Q_ice_b", fill(0.05, Nx, Ny), ("xc", "yc"))
            end

            for bed_rheology in (:hard, :soft, :mixed, :mixed_smooth)
                Nx_out, Ny_out, xlims, ylims, mask, h, b, abs_v_b, A_visc, G, q_T, ṁ, κ =
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

                # Regression test: ATT (Glen's law rate factor) is documented in [1/yr / Pa^3] by
                # Yelmo.jl's own rate_factor.jl constants, so A_visc must come back converted to
                # Pa^-n s^-1.
                @test all(≈(perYear2perSecond(1e-24)), A_visc)

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
