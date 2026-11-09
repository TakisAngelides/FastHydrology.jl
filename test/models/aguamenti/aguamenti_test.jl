using FastHydrology
using FastHydrology: AguamentiParams, sheet_flux, film_flux, sheet_saturation_flux, sheet_N_for_flux,
    channel_discharge, channel_N_for_flux, canal_depth, canal_discharge, canal_N_for_flux, critical_flux,
    onset_flux, channel_growth_rate, total_flux, saddle_node_flux, flux_balance_root, regime_pressure,
    strength_bound, grounding_line_correction, advance_channel_indicator, steady_channel_indicator,
    till_void_ratio, till_stress, till_bounds, till_consolidation_time, till_relax, till_from_exchange,
    AGUAMENTI_UPLIFT, AGUAMENTI_PRESSURIZED, AGUAMENTI_FREE_SURFACE, AGUAMENTI_STRENGTH, AGUAMENTI_DRY,
    AGUAMENTI_COLD, AGUAMENTI_NOT_GROUNDED
using Test

const YR = 365.25 * 86400

# A small ice-sheet margin: thickness decreasing towards an ocean row at i = Nx, bed sloping down.
function aguamenti_margin(Nx = 12, Ny = 6; bump = 0.0)
    grid = ArrayHydroGrid(Nx, Ny, (0.0, Nx * 5e3), (0.0, Ny * 5e3))
    L = Nx * grid.dx
    h = [600.0 + 1400.0 * sqrt(max(0.0, 1 - (i - 0.5) * grid.dx / L)) for i in 1:Nx, j in 1:Ny]
    b = [-100.0 - 200.0 * (i - 0.5) * grid.dx / L + bump * sin(j) for i in 1:Nx, j in 1:Ny]
    mask = fill(GROUNDED, Nx, Ny); mask[end, :] .= OCEAN
    return grid, HydroState(grid, HydroMask(mask), h, b)
end

aguamenti_model(grid; kappa = 0.0, u_b = 100.0, G = 0.1, kwargs...) =
    AguamentiHydroModel(grid, fill(kappa, grid.Nx, grid.Ny), fill(u_b / YR, grid.Nx, grid.Ny),
                        fill(2.4e-24, grid.Nx, grid.Ny), fill(G, grid.Nx, grid.Ny), zeros(grid.Nx, grid.Ny); kwargs...)

@testset "Aguamenti" begin

    grid1 = ArrayHydroGrid(3, 3, (0.0, 3e3), (0.0, 3e3))
    p = aguamenti_model(grid1).params
    A = 2.4e-24
    u_b = 100 / YR

    @testset "element library: values quoted in the paper" begin
        # Saturation flux of the sheet at Psi = 100 Pa/m: 2e4 m2/a (Sect. 2.4.1), laminar limit k_d h_r^3 Psi.
        @test isapprox(sheet_saturation_flux(p, 100.0) * YR, 2e4; rtol = 0.15)
        @test sheet_saturation_flux(p, 1e-6) ≈ p.k_d * p.h_r^3 * 1e-6 rtol = 1e-6
        # The film saturates at 10-300 m2/a.
        @test 10 <= FastHydrology.film_saturation_flux(p, 100.0) * YR <= 300
        # Canal: 1.5 cm deep at D_s = 2 mm, Psi = 100; N_s^e = 0.17 MPa at 1e3 m2/a and 0.04 MPa at 1e5 m2/a.
        @test canal_depth(p, 100.0, 2e-3) ≈ 0.0146 rtol = 0.01
        @test isapprox(canal_N_for_flux(p, 1e3 / YR, 100.0, A, 2e-3), 0.17e6; rtol = 0.1)
        @test isapprox(canal_N_for_flux(p, 1e5 / YR, 100.0, A, 2e-3), 0.04e6; rtol = 0.15)
        # Hewitt's critical discharge (laminar): rho_i L_w u_b h_r / (2 l_r Psi), 1e6-1e8 m2/a at 100 m/a.
        q_A = critical_flux(p, 100.0, u_b)
        @test 1e6 <= q_A * YR <= 1e8
        p_lam = aguamenti_model(grid1; Re_c = 1e30).params
        @test critical_flux(p_lam, 100.0, u_b) ≈ p.rho_i * p.L_w * u_b * p.h_r / (2 * p.l_r * 100.0) rtol = 1e-8
        # q_on = q_h,sat unless u_b < 2 k_d h_r^2 l_r Psi^2 / (rho_i L_w) (0.4 m/a at Psi = 100), laminar.
        u_crit = 2 * p.k_d * p.h_r^2 * p.l_r * 100.0^2 / (p.rho_i * p.L_w)
        @test isapprox(u_crit * YR, 0.4; rtol = 0.05)
        @test onset_flux(p_lam, 100.0, 2u_crit) == sheet_saturation_flux(p_lam, 100.0)
        @test onset_flux(p_lam, 100.0, u_crit / 2) < sheet_saturation_flux(p_lam, 100.0)
        # Channel growth time at onset: 0.2 a (Psi = 1000), 6 a (100), 360 a (10).
        for (Psi, t) in ((1000.0, 0.2), (100.0, 6.0), (10.0, 360.0))
            q_on = onset_flux(p, Psi, u_b)
            @test isapprox(1 / channel_growth_rate(p, q_on, Psi, u_b) / YR, t; rtol = 0.25)
        end
    end

    @testset "element library: monotonicity, scalings, inverses" begin
        Ns = exp10.(range(3, 7; length = 30))
        qh = [sheet_flux(p, N, 100.0, u_b, A) for N in Ns]
        Qh = [channel_discharge(p, N, 100.0, A) for N in Ns]
        Qs = [canal_discharge(p, N, 100.0, A, 2e-3) for N in Ns]
        @test issorted(qh; rev = true) && all(>(0), qh)
        @test issorted(Qh)
        @test issorted(Qs; rev = true)
        @test sheet_flux(p, 1e-3, 100.0, u_b, A) ≈ sheet_saturation_flux(p, 100.0) rtol = 1e-9
        # Q_h ~ N^15 Psi^-7 and Q_s ~ N^-3 Psi^-1 (n = 3, alpha = 5/4, beta = 3/2)
        @test channel_discharge(p, 2e6, 100.0, A) / channel_discharge(p, 1e6, 100.0, A) ≈ 2^15 rtol = 1e-9
        @test channel_discharge(p, 1e6, 200.0, A) / channel_discharge(p, 1e6, 100.0, A) ≈ 2.0^-7 rtol = 1e-9
        @test canal_discharge(p, 2e5, 100.0, A, 2e-3) / canal_discharge(p, 1e5, 100.0, A, 2e-3) ≈ 2.0^-3 rtol = 1e-9
        @test canal_discharge(p, 1e5, 200.0, A, 2e-3) / canal_discharge(p, 1e5, 100.0, A, 2e-3) ≈ 0.5 rtol = 1e-9
        # Inverses (Eqs. 12b, 14b, 17d)
        N = 2e6
        @test channel_N_for_flux(p, channel_discharge(p, N, 100.0, A) / p.l_h, 100.0, A) ≈ N rtol = 1e-9
        @test canal_N_for_flux(p, canal_discharge(p, N, 100.0, A, 2e-3) / p.l_s, 100.0, A, 2e-3) ≈ N rtol = 1e-9
        @test sheet_N_for_flux(p, sheet_flux(p, N, 100.0, u_b, A), 100.0, u_b, A) ≈ N rtol = 1e-6
        @test sheet_N_for_flux(p, 2 * sheet_saturation_flux(p, 100.0), 100.0, u_b, A) == 0
        # A channel at 1e4 m2/a and Psi = 100 has an effective pressure of order MPa (Fig. 4b)
        @test 0.5e6 < channel_N_for_flux(p, 1e4 / YR, 100.0, A) < 3e6
        # No sliding: closed cavities, no sheet flux for N > 0
        @test sheet_flux(p, 1e5, 100.0, 0.0, A) == 0
    end

    @testset "flux balance: roots and branches (Fig. 7)" begin
        Psi, p_i = 100.0, 917 * 9.81 * 2000
        q_sat = sheet_saturation_flux(p, Psi)
        q_sn, N_m = saddle_node_flux(p, 0.0, Psi, u_b, A, 2e-3)
        @test 0 < q_sn <= q_sat
        # q_sat / q_sn is 1.0, 1.7 and 80 at Psi = 10, 100 and 1000 (Appendix B4)
        for (Psi_, ratio) in ((10.0, 1.0), (100.0, 1.7), (1000.0, 80.0))
            qs, _ = saddle_node_flux(p, 0.0, Psi_, u_b, A, 2e-3)
            @test isapprox(sheet_saturation_flux(p, Psi_) / qs, ratio; rtol = 0.3)
        end
        # Sheet only, q = 0.25 q_sat: a unique pressurized root
        q = 0.25q_sat
        N_star, chan = flux_balance_root(p, q, 0.0, 0.0, Psi, u_b, A, 2e-3, p_i, q_sn)
        @test !chan && 0 < N_star < p_i
        @test total_flux(p, N_star, 0.0, 0.0, Psi, u_b, A, 2e-3) ≈ q rtol = 1e-6
        # Above saturation without channels: no root (uplift); with channels: the channelized root
        q = 1.6q_sat
        N0, _ = flux_balance_root(p, q, 0.0, 0.0, Psi, u_b, A, 2e-3, p_i, q_sn)
        @test N0 == 0
        N1, chan = flux_balance_root(p, q, 1.0, 0.0, Psi, u_b, A, 2e-3, p_i, q_sn)
        @test chan && N1 > N_m
        @test total_flux(p, N1, 1.0, 0.0, Psi, u_b, A, 2e-3) ≈ q rtol = 1e-6
        @test regime_pressure(p, N0, p_i, Inf) == (p.eps_u * p_i, AGUAMENTI_UPLIFT)
        # Bistable band: the channelized root lies above the sheet root
        if q_sn < 0.7q_sat
            q = 0.7q_sat
            Ns, _ = flux_balance_root(p, q, 0.0, 0.0, Psi, u_b, A, 2e-3, p_i, q_sn)
            Nc, chan = flux_balance_root(p, q, 1.0, 0.0, Psi, u_b, A, 2e-3, p_i, q_sn)
            @test chan && Nc > Ns > 0
        end
        # A decaying channel (q < q_sn) uses the sheet branch
        _, chan = flux_balance_root(p, 0.5q_sn, 1.0, 0.0, Psi, u_b, A, 2e-3, p_i, q_sn)
        @test !chan
        # Soft bed: always a root, decreasing in q
        Na, _ = flux_balance_root(p, 1e2 / YR, 0.0, 1.0, Psi, u_b, A, 2e-3, p_i, 0.0)
        Nb, _ = flux_balance_root(p, 1e4 / YR, 0.0, 1.0, Psi, u_b, A, 2e-3, p_i, 0.0)
        @test 0 < Nb < Na
        # Regimes and the strength bound (P2)
        @test strength_bound(p, 0.6, 20e3) ≈ 1e5
        @test strength_bound(p, 0.4, 20e3) == Inf
        @test regime_pressure(p, 5e5, 2e7, 1e5) == (2e5, AGUAMENTI_UPLIFT)    # floor eps_u p_i above N_0
        @test regime_pressure(p, 5e5, 2e6, 1e5) == (1e5, AGUAMENTI_STRENGTH)
        @test regime_pressure(p, 3e6, 2e6, Inf) == (2e6, AGUAMENTI_FREE_SURFACE)
        @test regime_pressure(p, 1e6, 2e6, Inf) == (1e6, AGUAMENTI_PRESSURIZED)
        # Grounding-line correction: 0 at flotation, -> N_inf far upstream
        hf = 1000.0; bf = -p.rho_i * hf / p.rho_sw
        @test grounding_line_correction(p, 1e6, hf, bf) ≈ 0 atol = 1e-6
        @test grounding_line_correction(p, 1e6, 2000.0, 0.0) ≈ 1e6 rtol = 1e-6
    end

    @testset "channel indicator (Eq. 22)" begin
        L0 = 1 / YR
        # growth from 0 at Lambda_0 for one e-folding time
        @test advance_channel_indicator(p, 0.0, 2.0, 1.0, 0.5, L0, 1e6, A, YR) ≈ 1 - exp(-1) rtol = 1e-12
        # decay at the creep rate c_e A N^n
        b = p.c_e * A * 1e6^3
        @test advance_channel_indicator(p, 1.0, 0.1, 1.0, 0.5, L0, 1e6, A, 1 / b) ≈ exp(-1) rtol = 1e-12
        # bistable band: unchanged
        @test advance_channel_indicator(p, 0.3, 0.7, 1.0, 0.5, L0, 1e6, A, YR) == 0.3
        @test advance_channel_indicator(p, 1e-3, 0.1, 1.0, 0.5, L0, 1e6, A, 100 / b) == 0
        @test steady_channel_indicator(0.0, 2.0, 1.0, 0.5) == 1
        @test steady_channel_indicator(1.0, 0.1, 1.0, 0.5) == 0
        @test steady_channel_indicator(1.0, 0.7, 1.0, 0.5) == 1
        @test steady_channel_indicator(0.0, 0.7, 1.0, 0.5) == 0
    end

    @testset "till reservoir (Sect. 2.5)" begin
        @test till_void_ratio(p, 0.2e6) ≈ 0.41 atol = 0.01
        @test till_stress(p, till_void_ratio(p, 3e5)) ≈ 3e5 rtol = 1e-12
        # T_t = 2.6 a (k_t = 1e-16) and 260 a (1e-18) for h_t = 5 m, N_t = 0.1 MPa
        @test isapprox(till_consolidation_time(p, 1e5, 5.0, 1e-16) / YR, 2.6; rtol = 0.1)
        @test isapprox(till_consolidation_time(p, 1e5, 5.0, 1e-18) / YR, 260; rtol = 0.1)
        # Fig. 6d: 1.3 a at N_t = 0.2 MPa
        @test isapprox(till_consolidation_time(p, 2e5, 5.0, 1e-16) / YR, 1.3; rtol = 0.1)
        Nmin, Nmax = till_bounds(p, 1e7)
        @test till_void_ratio(p, Nmin) ≈ p.e_max && till_void_ratio(p, Nmax) ≈ p.e_min
        @test till_bounds(p, 1e5)[2] == 1e5
        # Relaxation towards N, exponential with T_t; water absorbed when N_t > N
        T_t = till_consolidation_time(p, 3e5, 5.0, 1e-16)
        N_new, demand, N_u, storage = till_relax(p, 3e5, 1e5, T_t, 5.0, 1e-16, 1e7, 0.0)
        @test N_new ≈ 1e5 + 2e5 * exp(-1) rtol = 1e-12
        @test demand > 0
        @test till_from_exchange(p, N_u, demand, T_t, storage, 1e7) ≈ N_new rtol = 1e-10
        # The exchange rate at the start matches Eq. (20a) for a short step
        dt = 1e-4 * T_t
        _, d, _, _ = till_relax(p, 3e5, 1e5, dt, 5.0, 1e-16, 1e7, 0.0)
        @test d ≈ FastHydrology.till_exchange_rate(p, 3e5, 1e5, 5.0, 1e-16) rtol = 1e-3
        # Undrained response: B = 0.8, 1 m/a thinning lowers N_t by 1.8 kPa/a, without exchanging water
        N_new, demand, N_u, _ = till_relax(p, 3e5, 3e5, YR, 5.0, 1e-16, 1e7, -1 / YR)
        @test 3e5 - N_u ≈ 0.2 * p.rho_i * p.g rtol = 1e-10
        @test isapprox(3e5 - N_u, 1.8e3; rtol = 0.01)
        # Stable for a step far longer than T_t: the till reaches N
        N_new, _, _, _ = till_relax(p, 3e5, 1e5, 1e4 * T_t, 5.0, 1e-16, 1e7, 0.0)
        @test N_new ≈ 1e5 rtol = 1e-10
    end

    @testset "steady state: hard, soft and mixed beds" begin
        grid, state = aguamenti_margin()
        g = (2:grid.Nx-1, :)
        for kappa in (0.0, 0.3, 0.7, 1.0)
            m = aguamenti_model(grid; kappa = kappa, max_iters = 50)
            @test FastHydrology.solve_aguamenti!(m, grid, state, FastHydrology.SteadyMode())
            @test all(isfinite, state.N) && all(isfinite, state.W)
            p_i = m.p_i
            @test all(0 .<= state.N .<= p_i .+ 1e-6)
            @test all(state.N[end, :] .== 0)
            @test all(m.regime[end, :] .== AGUAMENTI_NOT_GROUNDED)
            # Upstream of the boundary layer N is N_inf, which respects the floor and the bounds
            @test all(m.N_inf[g...] .>= p.eps_u .* p_i[g...] .- 1e-6)
            if kappa >= p.kappa_c
                @test all(m.N_inf[g...] .<= max.(p.F_b * 20e3, p.eps_u .* p_i[g...]) .+ 1e-6)
            end
            @test all(m.mdot_t .== 0)                               # till in equilibrium
            wet = (m.dry .== 0) .& (state.mask .== GROUNDED)
            @test all(m.N_t[I] ≈ clamp(m.N_wet[I], till_bounds(p, p_i[I])...) for I in findall(wet))
            # the flux balance holds where the root is pressurized
            pr = findall(==(AGUAMENTI_PRESSURIZED), m.regime)
            for I in pr
                qt = total_flux(m.params, m.N_star[I], m.chi[I], m.kappa[I], m.Psi[I], m.abs_v_b[I], m.A_visc[I], m.D_s[I])
                @test qt ≈ m.q[I] rtol = 1e-5
            end
        end
    end

    @testset "steady state: channels form in uplift and keep their hysteresis" begin
        grid, state = aguamenti_margin()
        m = aguamenti_model(grid; G = 20.0)    # strong melt on a small domain: fluxes above saturation
        update_steady_state!(m, grid, state)
        @test any(m.chi .== 1)
        @test all(m.chi[(m.q .>= m.q_on) .& (state.mask .== GROUNDED)] .== 1)
        @test all(m.chi[(m.q .< m.q_sn) .& (state.mask .== GROUNDED)] .== 0)
        # starting from chi = 1 everywhere, cells in the bistable band keep their channel
        m1 = aguamenti_model(grid; G = 20.0, chi0 = 1.0)
        update_steady_state!(m1, grid, state)
        band = (m.q .>= m.q_sn) .& (m.q .< m.q_on) .& (state.mask .== GROUNDED)
        @test all(m1.chi[band] .>= m.chi[band])
    end

    @testset "time steps: mass conservation, till, dry and cold cells" begin
        grid, state = aguamenti_margin()
        Nx, Ny = grid.Nx, grid.Ny
        cold = zeros(Nx, Ny); cold[1, 1] = 1; cold[3, 2] = 1
        m = aguamenti_model(grid; kappa = 0.8, k_t = 1e-17, cold = cold, N_t0 = 2e6)
        for _ in 1:3
            step!(m, grid, state, 0.5YR)
        end
        r = m.routing
        area = grid.dx * grid.dy
        # Water budget of the step: supply = outflow to the ocean + till uptake + refreezing
        supply = sum(r.mdot_total[i, j] / p.rho_w for i in 1:Nx-1, j in 1:Ny if cold[i, j] == 0) * area
        outflow = sum(r.psi_out[Nx - 1, :])
        sinks = sum(m.mdot_t[1:Nx-1, :]) * area + sum(m.refreeze) * area
        @test supply ≈ outflow + sinks rtol = 1e-8
        # Cold cells: dry, no flux, the dry N
        @test m.regime[1, 1] == AGUAMENTI_COLD && m.dry[1, 1] == 1 && m.q[1, 1] == 0
        @test state.N[3, 2] == min(m.N_t[3, 2], m.p_i[3, 2])
        # inflow from upstream refreezes in a cold cell (hard bed, so the till takes none of it)
        mc = aguamenti_model(grid; cold = cold)
        step!(mc, grid, state, YR)
        @test mc.refreeze[3, 2] > 0 && mc.q[3, 2] == 0 && state.N[3, 2] == mc.p_i[3, 2]
        supply = sum(mc.mdot_total[i, j] / p.rho_w for i in 1:Nx-1, j in 1:Ny if cold[i, j] == 0) * area
        @test supply ≈ sum(mc.psi_out[Nx - 1, :]) + sum(mc.refreeze) * area rtol = 1e-8
        # An over-consolidated till absorbs water (positive exchange) and softens
        @test any(m.mdot_t .> 0)
        @test all(isfinite, m.N_t) && all(isfinite, state.N)
        Nmin = till_bounds(p, 0.0)[1]
        @test all(Nmin - 1e-9 .<= m.N_t[1:Nx-1, :] .<= m.p_i[1:Nx-1, :] .+ 1e-6)

        # A permeable till fed by enough water reaches N within a few steps
        m2 = aguamenti_model(grid; kappa = 1.0, k_t = 1e-13, N_t0 = 3e5, G = 5.0)
        for _ in 1:4
            step!(m2, grid, state, YR)
        end
        wet = (m2.dry .== 0) .& (state.mask .== GROUNDED)
        @test count(wet) > 0
        @test maximum(abs.(m2.N_t[wet] .- clamp.(state.N[wet], Nmin, Inf)) ./ state.N[wet]) < 1e-3
        # ... while a supply below the till's demand is absorbed entirely (dry cells), and N_t falls towards N
        m5 = aguamenti_model(grid; kappa = 1.0, k_t = 1e-13, N_t0 = 5e5, G = 0.5)
        Nt_prev = copy(m5.N_t)
        for _ in 1:3
            step!(m5, grid, state, YR)
            @test all(m5.N_t[1:Nx-1, :] .< Nt_prev[1:Nx-1, :])
            Nt_prev .= m5.N_t
        end
        @test all(m5.dry[1:Nx-1, :] .== 1) && all(m5.q .== 0)
        @test m5.mdot_t[1:Nx-1, :] ≈ m5.mdot_total[1:Nx-1, :] ./ p.rho_w rtol = 1e-10

        # A dry cell: a till that can absorb more than the supply takes all of it
        m3 = aguamenti_model(grid; kappa = 1.0, k_t = 1e-14, N_t0 = 9e6, G = 0.02)
        step!(m3, grid, state, 0.1YR)
        @test any(m3.dry .== 1)
        d = findall((m3.dry .== 1) .& (m3.cold .== 0) .& (state.mask .== GROUNDED))
        @test all(state.N[d] .== min.(m3.N_t[d], m3.p_i[d]))
        @test all(m3.regime[d] .== AGUAMENTI_DRY)

        # Undrained response under thinning, no exchange possible (impermeable till)
        m4 = aguamenti_model(grid; kappa = 1.0, k_t = 1e-30, N_t0 = 3e5)
        dH = fill(-1 / YR, Nx, Ny)
        step!(m4, grid, state, YR; dH_dt = dH)
        @test m4.N_t[2, 2] ≈ 3e5 - 0.2 * p.rho_i * p.g rtol = 1e-6
        @test_throws ArgumentError step!(m4, grid, state, -1.0)
    end

    @testset "time steps: channel growth and decay" begin
        grid, state = aguamenti_margin()
        m = aguamenti_model(grid; G = 20.0)
        step!(m, grid, state, 1.0)
        grow = findall((m.q .>= m.q_on) .& (state.mask .== GROUNDED))
        @test !isempty(grow)
        chi1 = copy(m.chi)
        for _ in 1:20
            step!(m, grid, state, 5YR)
        end
        @test all(m.chi[grow] .>= chi1[grow])
        @test maximum(m.chi) > 0.5
        # switch the supply off: channels can no longer be sustained and decay
        set_aguamenti_inputs!(m; G = fill(1e-4, grid.Nx, grid.Ny))
        chi_before = copy(m.chi)
        for _ in 1:50
            step!(m, grid, state, 10YR)
        end
        @test sum(m.chi) < sum(chi_before)
    end

    @testset "routing options, sliding coupling and N_from_ub!" begin
        grid, state = aguamenti_margin()
        mh = aguamenti_model(grid; routing_potential = HydraulicRouting(), max_iters = 200)
        @test FastHydrology.solve_aguamenti!(mh, grid, state, FastHydrology.SteadyMode())
        Nh = copy(state.N)
        mg = aguamenti_model(grid)
        update_steady_state!(mg, grid, state)
        @test all(isfinite, Nh)
        # with a regularized Coulomb law the frictional heat depends on N: solved in the same loop
        ms = aguamenti_model(grid; sliding_law = RegularizedCoulombSlidingLaw(c_till = 0.5), max_iters = 100)
        @test FastHydrology.solve_aguamenti!(ms, grid, state, FastHydrology.SteadyMode())
        @test any(ms.Q_b .> 0)
        @test N_responds_to_ub(ms)
        N0 = copy(state.N)
        N_from_ub!(ms, grid, state, fill(200 / YR, grid.Nx, grid.Ny))
        pr = findall(==(AGUAMENTI_PRESSURIZED), ms.regime)
        @test !isempty(pr)
        @test all(state.N[pr] .>= N0[pr])    # faster sliding opens cavities: the same flux at higher N
        @test_throws ArgumentError aguamenti_model(grid; kappa = 1.5)
        @test_throws ArgumentError aguamenti_model(grid; routing_options = (psi_out_algorithm = RecursivePsiOut(),))
    end

    @testset "review fixes" begin
        Psi, p_i = 100.0, 917 * 9.81 * 2000
        q_sat = sheet_saturation_flux(p, Psi)
        q_sn, _ = saddle_node_flux(p, 0.0, Psi, u_b, A, 2e-3)
        # A channel developing from uplift keeps the cell at the floor until it is developed
        q = 1.6q_sat
        N_dev, _ = flux_balance_root(p, q, 0.1, 0.0, Psi, u_b, A, 2e-3, p_i, q_sn; developing = true)
        @test N_dev == 0
        N_full, _ = flux_balance_root(p, q, 1.0, 0.0, Psi, u_b, A, 2e-3, p_i, q_sn; developing = true)
        N_young, _ = flux_balance_root(p, q, 0.1, 0.0, Psi, u_b, A, 2e-3, p_i, q_sn)    # bistable band: not growing
        @test N_full > 0 && N_young >= N_full

        grid, state = aguamenti_margin()
        Nx, Ny = grid.Nx, grid.Ny
        # Dry cells of a mixed bed no longer flip between wet and dry within a step
        m = aguamenti_model(grid; kappa = 0.3, G = 10.0, k_t = 1e-14, N_t0 = 1.7e6, max_iters = 60)
        @test FastHydrology.solve_aguamenti!(m, grid, state, FastHydrology.StepMode(0.1YR, nothing))
        # ... nor do whole flow lines on a longer margin (the till uptake is solved cell by cell in routing order)
        gl = ArrayHydroGrid(80, 4, (0.0, 400e3), (0.0, 20e3))
        hl = [600.0 + 2400.0 * sqrt(max(0.0, 1 - (i - 0.5) / 80)) for i in 1:80, j in 1:4]
        bl = [-100.0 - 200.0 * (i - 0.5) / 80 for i in 1:80, j in 1:4]
        ml = fill(GROUNDED, 80, 4); ml[end, :] .= OCEAN
        sl = HydroState(gl, HydroMask(ml), hl, bl)
        mm = aguamenti_model(gl; kappa = 0.4, G = 0.3, k_t = 1e-15, max_iters = 60)
        update_steady_state!(mm, gl, sl)
        mm.N_t .= 1.7e6
        @test FastHydrology.solve_aguamenti!(mm, gl, sl, FastHydrology.StepMode(0.05YR, nothing))

        # Freeze-on: no release where there is no till, and no more than the till holds (Eq. 9b)
        qT = zeros(Nx, Ny); qT[1:3, :] .= 0.3
        for (kappa, Nt0) in ((0.0, nothing), (1.0, 1e9))
            mf = AguamentiHydroModel(grid, fill(kappa, Nx, Ny), fill(u_b, Nx, Ny), fill(A, Nx, Ny), fill(0.05, Nx, Ny), qT; N_t0 = Nt0)
            step!(mf, grid, state, YR)
            @test mf.dry[2, 2] == 1
            @test mf.mdot_t[2, 2] ≈ 0 atol = 1e-20
            @test mf.freeze_deficit[2, 2] > 0
        end
        # The exchange equals the change of stored till water
        mt = aguamenti_model(grid; kappa = 1.0, k_t = 1e-15, N_t0 = 1e6, G = 0.5)
        e0 = till_void_ratio.(Ref(p), copy(mt.N_t))
        step!(mt, grid, state, YR)
        g = 1:Nx-1
        @test mt.mdot_t[g, :] .* YR ≈ mt.storage[g, :] .* (till_void_ratio.(Ref(p), mt.N_t[g, :]) .- e0[g, :]) rtol = 1e-8

        # The dissipation melt is q Psi >= 0, also in a hollow of the potential
        gridb, stateb = aguamenti_margin(bump = 30.0)
        md = aguamenti_model(gridb; G = 1.0)
        update_steady_state!(md, gridb, stateb)
        @test all(>=(0), md.Q_diss)

        # Steady chi depends on the indicator before the solve and the converged q, not on the iteration path
        ma = aguamenti_model(grid; G = 20.0)
        update_steady_state!(ma, grid, state)
        mb = aguamenti_model(grid; G = 20.0)
        state.N .= 5e6
        update_steady_state!(mb, grid, state)
        @test ma.chi == mb.chi
        @test all(ma.chi .== steady_channel_indicator.(0.0, ma.q_in, ma.q_on, ma.q_sn) .* (stateb.mask .== GROUNDED))

        # Diagnostic element fluxes: the channel term is zero off the channelized branch
        @test all(ma.q_ch[ma.chi .== 0] .== 0)
        # Keywords of K24 that Aguamenti does not use are rejected
        @test_throws ArgumentError aguamenti_model(grid; routing_options = (sensible_heat = true,))
        @test aguamenti_model(grid; routing_options = (q_max = 1.0,)) isa AguamentiHydroModel
    end

    @testset "Float32" begin
        grid = ArrayHydroGrid(8, 4, (0.0, 40e3), (0.0, 20e3); T = Float32)
        h = Float32[1500 - 100 * i for i in 1:8, j in 1:4]
        b = Float32[-100 - 10 * i for i in 1:8, j in 1:4]
        mask = fill(GROUNDED, 8, 4); mask[end, :] .= OCEAN
        state = HydroState(grid, HydroMask(mask), h, b)
        m = AguamentiHydroModel(grid, fill(0.5f0, 8, 4), fill(Float32(100 / YR), 8, 4), fill(2.4f-24, 8, 4),
                                fill(0.1f0, 8, 4), zeros(Float32, 8, 4); rtol = 1e-4)
        update_steady_state!(m, grid, state)
        @test eltype(state.N) == Float32
        @test all(isfinite, state.N)
        step!(m, grid, state, Float32(YR))
        @test all(isfinite, m.N_t)
    end
end
