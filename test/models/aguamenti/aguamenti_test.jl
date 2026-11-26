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

# The same parameters with some fields changed (no model needed).
aguamenti_remake(p; kw...) = typeof(p)((haskey(kw, f) ? kw[f] : getfield(p, f) for f in fieldnames(typeof(p)))...)

# Deterministic draws in [0, 1) for the random combinations of Table B1 (an additive recurrence, one
# irrational step per parameter), and their log-uniform scaling.
aguamenti_draw(n, k) = mod(n * sqrt((2, 3, 5, 7, 11, 13, 17, 19, 23, 29)[k]), 1.0)
aguamenti_logu(u, lo, hi) = lo * (hi / lo)^u

@testset "Aguamenti" begin

    grid1 = ArrayHydroGrid(3, 3, (0.0, 3e3), (0.0, 3e3))
    p = aguamenti_model(grid1).params
    A = 2.4e-24
    u_b = 100 / YR

    @testset "element library: values quoted in the paper" begin
        # Saturation flux of the sheet at Psi = 100 Pa/m: 2e4 m2/a (Sect. 3.4.1), laminar limit k_d h_r^3 Psi.
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
        # Inverses (Eqs. 11b, 13b, 16d)
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
        # q_sat / q_sn is 1.0, 1.7 and 80 at Psi = 10, 100 and 1000 (Appendix D4)
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
        # A closing channel (q < q_sn): below the minimum of q_tot(N; chi), held at its minimiser; without a
        # channel, the sheet branch
        Nc, chan = flux_balance_root(p, 0.5q_sn, 1.0, 0.0, Psi, u_b, A, 2e-3, p_i, q_sn)
        @test chan
        @test Nc ≈ N_m rtol = 1e-12
        _, chan = flux_balance_root(p, 0.5q_sn, 0.0, 0.0, Psi, u_b, A, 2e-3, p_i, q_sn)
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

    @testset "channel indicator (Eq. 21)" begin
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

    @testset "closing channels (Sect. 3.6): a fraction chi of the channel flux, N continuous" begin
        FH = FastHydrology
        Psi, p_i, D_s = 100.0, 917 * 9.81 * 1500, 0.0
        q_sn, N_sn = saddle_node_flux(p, 0.0, Psi, u_b, A, D_s)
        q_on = onset_flux(p, Psi, u_b)
        sheet(q) = flux_balance_root(p, q, 0.0, 0.0, Psi, u_b, A, D_s, p_i, q_sn)[1]
        # (b) chi = 1 and q >= q_sn: the channelized branch, as before (the larger root of q_tot(N; 1))
        for q in (q_sn, 1.2q_sn, 0.9q_on)
            N, chan = flux_balance_root(p, q, 1.0, 0.0, Psi, u_b, A, D_s, p_i, q_sn; N_sn)
            @test chan && N >= N_sn * (1 - 1e-6) && N == FH.channelized_root(p, q, 1.0, 0.0, Psi, u_b, A, D_s, p_i; minimum = (N_sn, q_sn))
        end
        # the closing root: q_tot(N; chi) = q on the sheet side of its minimum, or the minimiser; the sheet
        # root as chi -> 0 (so the chi_min cutoff causes no jump)
        for (q, chi) in ((0.6q_sn, 0.5), (0.9q_sn, 0.1), (0.95q_sn, 0.9), (0.3q_sn, 0.02))
            f = N -> total_flux(p, N, chi, 0.0, Psi, u_b, A, D_s)
            N_m, q_m = FH.log_brent_min(f, p.N_lo, p.N_hi, 2 * p.root_iters)
            N, chan = flux_balance_root(p, q, chi, 0.0, Psi, u_b, A, D_s, p_i, q_sn; N_sn)
            @test chan
            if q_m >= q
                @test N ≈ min(N_m, p_i) rtol = 1e-12
            else
                @test f(N) ≈ q rtol = 1e-8
                @test N <= N_m && N >= sheet(q)
            end
        end
        @test FH.closing_root(p, 0.6q_sn, p.chi_min, 0.0, Psi, u_b, A, D_s, p_i) ≈ sheet(0.6q_sn) rtol = 1e-5
        # a developing channel (q >= q_on) keeps the uplift floor until chi_dev
        @test flux_balance_root(p, 1.01q_on, 0.5, 0.0, Psi, u_b, A, D_s, p_i, q_sn; developing = true, N_sn) == (0.0, true)
        # (a) one cell, q falling from above q_sn to below it while chi decays by creep (closure time 0.065 a):
        # N has no jump (the largest change per step shrinks with dt, like that of the fold of the channelized
        # branch it leaves), and ends on the sheet root once chi reaches 0
        function closing_run(dt; closing = true)
            chi, N, Ns = 1.0, 1.05N_sn, Float64[]
            for k in 1:round(Int, 3YR / dt)
                q = q_sn * max(1.05 - 0.15 * k * dt / YR, 0.8)
                chi = advance_channel_indicator(p, chi, q, q_on, q_sn, channel_growth_rate(p, q, Psi, u_b), N, A, dt)
                N_star, _ = flux_balance_root(p, q, closing || q >= q_sn ? chi : 0.0, 0.0, Psi, u_b, A, D_s, p_i, q_sn;
                                              developing = q >= q_on, N_sn)
                N = regime_pressure(p, N_star, p_i, Inf)[1]
                push!(Ns, N)
            end
            return maximum(abs, diff(Ns)) / (maximum(Ns) - minimum(Ns)), N, chi
        end
        j3, N_end, chi_end = closing_run(1e-3YR)
        j4, _, _ = closing_run(1e-4YR)
        @test j4 < 0.05 && j4 < 0.5j3
        @test chi_end == 0
        @test N_end ≈ sheet(0.8q_sn) rtol = 1e-12
        # (the sheet branch at once, as before: a jump of ~0.55 of the range that does not shrink with dt)
        @test closing_run(1e-4YR; closing = false)[1] > 0.5
    end

    @testset "till reservoir (Sect. 3.5)" begin
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
        N_new, demand, dN_u, storage = till_relax(p, 3e5, 1e5, T_t, 5.0, 1e-16, 1e7, 0.0)
        @test N_new ≈ 1e5 + 2e5 * exp(-1) rtol = 1e-12
        @test demand > 0 && dN_u == 0
        @test till_from_exchange(p, 3e5, demand, T_t, storage, 1e7) ≈ N_new rtol = 1e-10
        # The exchange rate at the start matches Eq. (19a) for a short step
        dt = 1e-4 * T_t
        _, d, _, _ = till_relax(p, 3e5, 1e5, dt, 5.0, 1e-16, 1e7, 0.0)
        @test d ≈ FastHydrology.till_exchange_rate(p, 3e5, 1e5, 5.0, 1e-16) rtol = 1e-3
        # Undrained response: B = 0.8, 1 m/a thinning lowers N_t by 1.8 kPa/a, without exchanging water
        N_new, demand, dN_u, _ = till_relax(p, 3e5, 3e5, YR, 5.0, 1e-16, 1e7, -1 / YR)
        @test -dN_u ≈ 0.2 * p.rho_i * p.g rtol = 1e-10
        @test isapprox(-dN_u, 1.8e3; rtol = 0.01)
        @test N_new ≈ 3e5 + dN_u rtol = 1e-12
        @test demand == 0
        # Eq. (28a): the undrained term is added after the decay, which uses T_t at the N_t the step starts from
        dt = 0.7YR
        N_new, demand, dN_u, storage = till_relax(p, 3e5, 1e5, dt, 5.0, 1e-16, 1e7, -3 / YR)
        @test N_new ≈ 1e5 + 2e5 * exp(-dt / T_t) + dN_u rtol = 1e-12
        @test till_from_exchange(p, 3e5, demand, dt, storage, 1e7, dN_u) ≈ N_new rtol = 1e-10
        # Eq. (28b): the lagged exchange is the storage change of the Eq. (28a) step towards N^(j-1) in the
        # bounds; for a short step and a small N_t - N, the rate of Eq. (19a) averaged over the step
        FH = FastHydrology
        c = FH.till_relax_coefficients(p, 3e5, dt, 5.0, 1e-16, 1e7, 0.0)
        @test FH.till_lagged_exchange(p, c, 1e5) == FH.till_relax_to(p, c, 1e5)[2]
        @test FH.till_lagged_exchange(p, c, 0.0) ≈ FH.till_lagged_exchange(p, c, c.N_min) rtol = 1e-12
        @test FH.till_lagged_exchange(p, c, 3e5) == 0
        cs = FH.till_relax_coefficients(p, 3e5, 1e-3T_t, 5.0, 1e-16, 1e7, 0.0)
        @test FH.till_lagged_exchange(p, cs, 2.97e5) ≈
              FH.till_exchange_rate(p, 3e5, 2.97e5, 5.0, 1e-16) * T_t / cs.dt * (-expm1(-cs.dt / T_t)) rtol = 1e-2
        # ... so that, applied in full through the storage law, it takes the till to the Eq. (28a) value,
        # between N_t^(j) and N^(j-1), for any step (the linear rate overshot N^(j-1), up to N_max)
        for N_t in (2e4, 1e5, 1e6), N_lag in (0.0, 5e4, 2e5, 1e6, 3e6), k_t in (1e-18, 1e-16, 1e-14), dtk in (0.01YR, YR, 10YR)
            c = FH.till_relax_coefficients(p, N_t, dtk, 5.0, k_t, 1e7, 0.0)
            ex = FH.till_lagged_exchange(p, c, N_lag)
            N_new = till_from_exchange(p, N_t, ex, dtk, c.storage, 1e7)
            Nl = clamp(N_lag, c.N_min, c.N_max)
            @test min(N_t, Nl) * (1 - 1e-10) <= N_new <= max(N_t, Nl) * (1 + 1e-10)
            @test N_new ≈ FH.till_relax_to(p, c, N_lag)[1] rtol = 1e-10
        end
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
        step!(m, grid, state, 0.01YR)
        # (c) a closing channel (0 < chi, q < q_sn) carries the fraction chi of its flux, chi Q_h/l_h
        closing = findall((state.mask .== GROUNDED) .& (m.chi .> 0) .& (m.q_in .< m.q_sn))
        @test !isempty(closing)
        for c in closing
            @test m.q_ch[c] ≈ m.chi[c] * channel_discharge(p, max(m.N_wet[c], p.N_lo), m.Psi[c], m.A_visc[c]) / p.l_h rtol = 1e-12
            @test m.q_ch[c] > 0
        end
        for _ in 1:50
            step!(m, grid, state, 10YR)
        end
        @test sum(m.chi) < sum(chi_before)
    end

    @testset "dissipation rule: the flux of the dissipation melt held within a solve" begin
        @test aguamenti_model(aguamenti_margin()[1]).dissipation_rule isa HeldDissipation
        @test HeldDissipation().after == 10
        @test_throws ArgumentError HeldDissipation(; after = 1)
        # a solve that converges before `after` sweeps is the same with either rule
        g2, s_free = aguamenti_margin(); _, s_held = aguamenti_margin()
        mf = aguamenti_model(g2; G = 20.0, dissipation_rule = FreeDissipation())
        mh = aguamenti_model(g2; G = 20.0)
        for dt in (1.0, 5YR, 5YR)
            step!(mf, g2, s_free, dt); step!(mh, g2, s_held, dt)
            @test mf.info.converged && mh.info.converged && mh.info.iters < 10
            @test !mh.info.dissipation_held && isnan(mh.info.res_diss) && !mf.info.dissipation_held
            @test s_free.N == s_held.N && mf.chi == mh.chi
        end
        # the failure it fixes: with till (kappa = 0.5) the first step of the margin keeps moving at the
        # 1e-6 level with the free rule; the default holds the dissipation flux at sweep 10 and converges
        for kap in (0.5, 1.0)
            gf, sf = aguamenti_margin(); gh, sh = aguamenti_margin()
            mf1 = aguamenti_model(gf; kappa = kap, G = 20.0, dissipation_rule = FreeDissipation())
            mh1 = aguamenti_model(gh; kappa = kap, G = 20.0)
            step!(mf1, gf, sf, 1.0); step!(mh1, gh, sh, 1.0)
            @test !mf1.info.converged && mf1.info.iters == mf1.max_iters
            @test mh1.info.converged && mh1.info.iters == HeldDissipation().after
            @test mh1.info.dissipation_held && mh1.info.res_diss < 1e-4
            @test maximum(abs.(sh.N .- sf.N)) <= 1e-4 * maximum(abs, sf.N)
        end
        # held from sweep 2: the dissipation melt is that of the held flux, and the gap is reported
        g3, s3 = aguamenti_margin()
        m2 = aguamenti_model(g3; G = 20.0, dissipation_rule = HeldDissipation(; after = 2))
        for dt in (1.0, 5YR)
            step!(m2, g3, s3, dt)
            @test m2.info.converged
            if m2.info.iters >= 2
                @test m2.info.dissipation_held && 0 <= m2.info.res_diss < 1
            end
            @test all(isfinite, s3.N)
        end
    end

    @testset "routing options, sliding coupling and N_from_ub!" begin
        grid, state = aguamenti_margin()
        mh = aguamenti_model(grid; routing_potential = HydraulicRouting(), max_iters = 200)
        @test FastHydrology.solve_aguamenti!(mh, grid, state, FastHydrology.SteadyMode())
        Nh = copy(state.N)
        mg = aguamenti_model(grid)
        update_steady_state!(mg, grid, state)
        @test all(isfinite, Nh)
        # with a regularized Coulomb law the frictional heat depends on N: solved in the same loop (joint),
        ms = aguamenti_model(grid; sliding_law = RegularizedCoulombSlidingLaw(c_till = 0.5), max_iters = 100,
                             lagged_friction = false)
        @test FastHydrology.solve_aguamenti!(ms, grid, state, FastHydrology.SteadyMode())
        @test any(ms.Q_b .> 0)
        N_joint = copy(state.N)
        # or (default) from the N the solve starts from, held during the solve: repeated calls reach the same state
        ml = aguamenti_model(grid; sliding_law = RegularizedCoulombSlidingLaw(c_till = 0.5))
        @test ml.lagged_friction
        state.N .= 0
        update_steady_state!(ml, grid, state)
        @test all(ml.tau_b[state.mask .== GROUNDED] .== 0)        # N = 0 going in: no friction this call
        for _ in 1:30
            update_steady_state!(ml, grid, state)
        end
        @test maximum(abs, state.N .- N_joint) <= 1e-4 * maximum(N_joint)
        tau_held = copy(ml.tau_b)
        update_steady_state!(ml, grid, state)
        @test ml.info.converged
        @test ml.tau_b == tau_held || maximum(abs, ml.tau_b .- tau_held) <= 1e-4 * maximum(tau_held)
        @test N_responds_to_ub(ms)
        N0 = copy(state.N)
        N_from_ub!(ms, grid, state, fill(200 / YR, grid.Nx, grid.Ny))
        pr = findall(==(AGUAMENTI_PRESSURIZED), ms.regime)
        @test !isempty(pr)
        @test all(state.N[pr] .>= N0[pr])    # faster sliding opens cavities: the same flux at higher N
        # a host that refreshes the thickness in place between solves: the overburden follows it
        h0 = copy(state.h)
        state.h .*= 0.9
        N_from_ub!(ms, grid, state, fill(200 / YR, grid.Nx, grid.Ny))
        @test ms.p_i ≈ ms.params.rho_i * ms.params.g .* state.h
        @test all(state.N[state.mask .== GROUNDED] .<= ms.p_i[state.mask .== GROUNDED] .* (1 + 1e-12))
        state.h .= h0
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

    @testset "weighted routing (Eq. 26)" begin
        FH = FastHydrology
        @test aguamenti_model(grid1).routing_potential isa GeometricRouting
        @test WeightedRouting().gamma0 == 0.5 && WeightedRouting(0.2) == WeightedRouting(; gamma0 = 0.2)
        @test_throws ArgumentError WeightedRouting(-0.1)
        # gamma0 = 0 is GeometricRouting
        grid, sa = aguamenti_margin(bump = 20.0)
        _, sb = aguamenti_margin(bump = 20.0)
        mg = aguamenti_model(grid; kappa = 0.3, G = 20.0)
        mw = aguamenti_model(grid; kappa = 0.3, G = 20.0, routing_potential = WeightedRouting(0.0))
        for _ in 1:3
            step!(mg, grid, sa, 2YR)
            step!(mw, grid, sb, 2YR)
        end
        @test sa.N == sb.N && mg.q == mw.q && mg.chi == mw.chi
        @test mw.info.routing_builds == 1 && mw.info.G_gamma == 0 && mw.info.clipped == 0
        # A step builds the graph once, from chi^(j) and N^(j-1); steady solves rebuild it every sweep
        m = aguamenti_model(grid; kappa = 0.3, G = 20.0, routing_potential = WeightedRouting(), max_iters = 200)
        # (here channels form in uplift and the graph cycles: it is held after max_iters ÷ 2 sweeps, the
        # fluxes and N then settle on the held graph, but the routing has not, so the solve reports that it
        # did not converge: res_route = 0.34 > rtol, against res_q, res_N ~ 1.6e-7)
        @test !FH.solve_aguamenti!(m, grid, sb, FH.SteadyMode())
        @test !m.info.converged && m.info.iters < m.max_iters
        @test m.info.routing_builds == m.max_iters ÷ 2 && m.info.res_route > 0.1
        @test m.info.res_q <= m.rtol && m.info.res_N <= m.rtol
        @test all(isfinite, sb.N)
        # with a weaker feedback the graph settles on the N it produces
        _, sc = aguamenti_margin(bump = 20.0)
        mc = aguamenti_model(grid; kappa = 0.3, G = 5.0, routing_potential = WeightedRouting(), max_iters = 100)
        @test FH.solve_aguamenti!(mc, grid, sc, FH.SteadyMode())
        @test mc.info.converged && mc.info.routing_builds == mc.info.iters
        @test mc.routing_relaxation * mc.info.res_route <= mc.rtol
        @test isnan(mg.info.res_route)
        # gamma0 > 0 changes the graph: the step's correction is Eq. (26) at chi^(j) and N^(j-1), and the
        # routed flux differs from the geometric one (by most of max q on this bumpy margin)
        _, sg = aguamenti_margin(bump = 20.0)
        _, sw = aguamenti_margin(bump = 20.0)
        mg2 = aguamenti_model(grid; kappa = 0.3, G = 20.0)
        mw2 = aguamenti_model(grid; kappa = 0.3, G = 20.0, routing_potential = WeightedRouting(0.5))
        for k in 1:3
            step!(mg2, grid, sg, 2YR)
            step!(mw2, grid, sw, 2YR)
            C = copy(mw2.route_corr)
            FH.weighted_routing_correction!(mw2, grid, sw, 0.5, mw2.chi_start, mw2.N_lag)
            @test C == mw2.route_corr
            # (the first step from a fresh state has N^(j-1) = state.N = 0, so no correction)
            @test k == 1 ? all(C .== 0) : any(C .> 0)
        end
        @test maximum(abs.(mg2.q .- mw2.q)) > 0.1 * maximum(mg2.q)
        for _ in 1:3
            step!(m, grid, sb, 2YR)
            @test m.info.routing_builds == 1 && isfinite(m.info.G_gamma)
        end
        @test all(isfinite, sb.N)
        # The clip (Eq. 26c): below gamma0 (1 - chi) N, 0 where chi = 1, every face within min(Psi) d / sqrt(2)
        Nx, Ny = grid.Nx, grid.Ny
        chi = zeros(Nx, Ny); chi[3:5, 2:3] .= 1
        Nr = [p_i * (0.2 + 0.8 * isodd(i + j)) for (p_i, i, j) in ((m.p_i[i, j], i, j) for i in 1:Nx, j in 1:Ny)]
        FH.weighted_routing_correction!(m, grid, sb, 0.5, chi, Nr)
        C, P, g = m.route_corr, m.Psi, sb.mask .== GROUNDED
        @test m.info.G_gamma > 1 && m.info.clipped > 0
        @test all(C .<= 0.5 .* (1 .- chi) .* Nr .* g .+ 1e-9) && all(C[chi .== 1] .== 0) && all(C[.!g] .== 0)
        s = 1 / sqrt(2)
        for j in 1:Ny, i in 1:Nx
            g[i, j] || continue
            i < Nx && g[i + 1, j] && @test abs(C[i + 1, j] - C[i, j]) <= min(P[i, j], P[i + 1, j]) * s * grid.dx * (1 + 1e-12)
            j < Ny && g[i, j + 1] && @test abs(C[i, j + 1] - C[i, j]) <= min(P[i, j], P[i, j + 1]) * s * grid.dy * (1 + 1e-12)
        end
        # ... and the largest such field: every clipped cell is held by a neighbour at the bound
        for j in 1:Ny, i in 1:Nx
            (g[i, j] && C[i, j] < 0.5 * (1 - chi[i, j]) * Nr[i, j] - 1e-6) || continue
            held = false
            for (a, b, d) in ((i - 1, j, grid.dx), (i + 1, j, grid.dx), (i, j - 1, grid.dy), (i, j + 1, grid.dy))
                (1 <= a <= Nx && 1 <= b <= Ny && g[a, b]) || continue
                held |= C[i, j] ≈ C[a, b] + min(P[i, j], P[a, b]) * s * d
            end
            @test held
        end
        # the gradient the routing sees never exceeds Psi, with either of K24's stencils: the face one
        # (the default with Warner) and the centred one (each centred component within Psi/sqrt(2))
        r = m.routing
        phi0 = copy(r.phi0)
        r.phi0 .= C
        @test r.gradient_discretization isa FH.FaceGradient
        for gd in (FH.FaceGradient(), FH.CentredGradient())
            FH.update_true_gradient_magnitude!(r, grid, gd)
            @test all(r.abs_grad_phi0[g] .<= P[g] .* (1 + 1e-9))
        end
        r.phi0 .= phi0
        FH.update_true_gradient_magnitude!(r, grid, r.gradient_discretization)
        alloc(f, args...) = (f(args...); @allocated f(args...))
        @test alloc(FH.weighted_routing_correction!, m, grid, sb, 0.5, chi, Nr) == 0
    end

    @testset "channel indicator before the flux balance (Sect. 4)" begin
        grid, state = aguamenti_margin()
        m = aguamenti_model(grid; G = 20.0)
        # the first step from chi = 0 already uses the channels it grows: the channel term is in the balance
        step!(m, grid, state, 2YR)
        @test any(m.chi .> 0) && any(m.q_ch .> 0)
        for _ in 1:2
            chi0, N0 = copy(m.chi), copy(state.N)
            dt = 3YR
            step!(m, grid, state, dt)
            for c in findall(state.mask .== GROUNDED)
                q = m.q[c]
                chi = advance_channel_indicator(m.params, chi0[c], q, m.q_on[c], m.q_sn[c],
                                                channel_growth_rate(m.params, q, m.Psi[c], m.abs_v_b[c]), N0[c], m.A_visc[c], dt)
                @test m.chi[c] ≈ chi rtol = 1e-12 atol = 1e-14
            end
            @test m.chi_start == chi0 && m.N_lag == N0
        end
    end

    @testset "till exchange: implicit (default) and lagged (Eqs. 27-28)" begin
        grid, state = aguamenti_margin()
        Nx, Ny = grid.Nx, grid.Ny
        area = grid.dx * grid.dy
        @test aguamenti_model(grid).till_exchange isa ImplicitTillExchange
        # Implicit: a wet cell's till follows Eq. (28a) towards the N of the end of the step, with the
        # undrained term after the decay
        mi = aguamenti_model(grid; kappa = 1.0, k_t = 1e-14, N_t0 = 6e5, G = 5.0)
        update_steady_state!(mi, grid, state)
        mi.N_t .= 6e5
        dH = fill(-2 / YR, Nx, Ny)
        dt = 0.5YR
        Nt0 = copy(mi.N_t)
        step!(mi, grid, state, dt; dH_dt = dH)
        wet = findall((mi.dry .== 0) .& (state.mask .== GROUNDED))
        @test !isempty(wet)
        for c in wet
            T_t = till_consolidation_time(p, Nt0[c], mi.h_t[c], mi.k_t[c])
            Nmin, Nmax = till_bounds(p, mi.p_i[c])
            N = clamp(mi.N_wet[c], Nmin, Nmax)
            @test mi.N_t[c] ≈ N + (Nt0[c] - N) * exp(-dt / T_t) + (1 - p.B) * p.rho_i * p.g * dH[c] * dt rtol = 1e-6
        end

        # Lagged: Eq. (28b) at N^(j-1), bounded by the water available and by the water the till holds
        for (Nt_start, G) in ((1.5e6, 0.3), (1.5e6, 5.0), (2e4, 5.0))
            ml = aguamenti_model(grid; kappa = 1.0, k_t = 1e-14, N_t0 = Nt_start, G = G, till_exchange = LaggedTillExchange())
            @test ml.till_exchange isa LaggedTillExchange
            update_steady_state!(ml, grid, state)
            ml.N_t .= Nt_start
            N_lag, Nt0 = copy(state.N), copy(ml.N_t)
            @test FastHydrology.solve_aguamenti!(ml, grid, state, FastHydrology.StepMode(dt, nothing))
            gr = findall((state.mask .== GROUNDED))
            for c in gr
                co = FastHydrology.till_relax_coefficients(p, Nt0[c], dt, ml.h_t[c], ml.k_t[c], ml.p_i[c], 0.0)
                @test ml.demand[c] ≈ FastHydrology.till_lagged_exchange(p, co, N_lag[c]) rtol = 1e-12
                @test ml.mdot_t[c] >= -ml.release_max[c] * (1 + 1e-12)
                if ml.dry[c] == 0
                    @test ml.mdot_t[c] ≈ max(ml.demand[c], -ml.release_max[c]) rtol = 1e-12
                else
                    @test ml.mdot_t[c] <= ml.demand[c]       # all the water available, no more than asked for
                end
            end
            # the water budget closes and the till stores exactly the exchange
            supply = sum(ml.mdot_total[i, j] / p.rho_w for i in 1:Nx-1, j in 1:Ny) * area
            @test supply ≈ sum(ml.psi_out[Nx - 1, :]) + sum(ml.mdot_t[1:Nx-1, :]) * area rtol = 1e-8
            if Nt_start < 1e5
                @test any(ml.mdot_t[1:Nx-1, :] .< 0)        # a loose till releases water
            elseif G < 1
                @test any(ml.dry[1:Nx-1, :] .== 1)         # a dense one takes all of a small supply
            end
        end
        # the step itself, and the bounded release within the till bounds
        ml = aguamenti_model(grid; kappa = 1.0, k_t = 1e-14, N_t0 = 2e4, G = 5.0, till_exchange = LaggedTillExchange())
        for _ in 1:3
            step!(ml, grid, state, dt)
        end
        Nmin = till_bounds(p, 0.0)[1]
        @test all(Nmin - 1e-9 .<= ml.N_t[1:Nx-1, :] .<= ml.p_i[1:Nx-1, :] .+ 1e-6)
        @test all(ml.freeze_deficit .>= 0)
        # the till stores the exchange through its storage law: storage (e(N_t^(j+1)) - e(N_t^(j))) = mdot_t dt
        for Nt_start in (1.5e6, 2e4)
            _, sl = aguamenti_margin()
            ml = aguamenti_model(grid; kappa = 1.0, k_t = 1e-14, N_t0 = Nt_start, G = 5.0, till_exchange = LaggedTillExchange())
            update_steady_state!(ml, grid, sl)
            ml.N_t .= Nt_start
            Nt0 = copy(ml.N_t)
            step!(ml, grid, sl, dt)
            @test any(ml.mdot_t .!= 0)
            for c in findall(sl.mask .== GROUNDED)
                Nmin, Nmax = till_bounds(p, ml.p_i[c])
                Nmin < ml.N_t[c] < Nmax || continue
                @test ml.storage[c] * (till_void_ratio(p, ml.N_t[c]) - till_void_ratio(p, Nt0[c])) ≈ ml.mdot_t[c] * dt rtol = 1e-10 atol = 1e-15
            end
        end
        # for short steps the two exchanges agree: N^(j-1) and the N of the end of the step differ little
        res = map((ImplicitTillExchange(), LaggedTillExchange())) do te
            _, s1 = aguamenti_margin()
            m1 = aguamenti_model(grid; kappa = 1.0, k_t = 1e-17, N_t0 = 1e6, G = 5.0, till_exchange = te)
            update_steady_state!(m1, grid, s1)
            m1.N_t .= 1e6
            step!(m1, grid, s1, 0.01YR)
            copy(m1.N_t), copy(m1.mdot_t)
        end
        gr = findall(state.mask .== GROUNDED)
        @test maximum(abs.(res[1][1][gr] .- res[2][1][gr])) <= 1e-5 * maximum(abs.(res[1][1][gr] .- 1e6))
        @test maximum(abs.(res[1][2][gr] .- res[2][2][gr])) <= 1e-5 * maximum(abs.(res[1][2][gr]))
        @test maximum(abs.(res[1][1][gr] .- 1e6)) > 0
        # the lagged uptake functor of the replay does not allocate
        lu = FastHydrology.till_uptake(ml.till_exchange, ml, state, FastHydrology.StepMode(dt, nothing), area)
        @test lu isa FastHydrology.LaggedUptake
        lu(2, 2, 1.0)
        @test (@allocated lu(2, 2, 1.0)) == 0
        # the implicit uptake holds chi^(j) by default (StepStartChi), the latest chi with LatestChi
        @test ImplicitTillExchange().chi isa StepStartChi
        ms = aguamenti_model(grid)
        @test FastHydrology.till_uptake(ms.till_exchange, ms, state, FastHydrology.StepMode(dt, nothing), area).chi === ms.chi_start
        mlc = aguamenti_model(grid; till_exchange = ImplicitTillExchange(; chi = LatestChi()))
        @test FastHydrology.till_uptake(mlc.till_exchange, mlc, state, FastHydrology.StepMode(dt, nothing), area).chi === mlc.chi

        # Implicit, with channels forming along mixed-bed flow lines: the till of a wet cell relaxes
        # towards the N of the cell and does not overshoot it. (With the channel indicator advanced inside
        # the uptake root at q(W - u dx dy), a release opened channels and raised N, a spurious root:
        # N_t jumped 1.2 MPa past N.) The tolerance covers the branch jumps of N(q). With the default
        # StepStartChi the till relaxes towards the N at chi^(j): where a channel opens within the step
        # (chi from 0), it lags the final N by that step, so there the bound applies with LatestChi only.
        for chiopt in (StepStartChi(), LatestChi())
            grid, state = aguamenti_margin(16, 8; bump = 20.0)
            Nx, Ny = grid.Nx, grid.Ny
            kappa = [j <= 4 ? 0.0 : (j <= 6 ? 0.4 : 1.0) for i in 1:Nx, j in 1:Ny]
            G = [j <= 4 ? 12.0 : 0.4 for i in 1:Nx, j in 1:Ny]
            mc = AguamentiHydroModel(grid, kappa, fill(100 / YR, Nx, Ny), fill(2.4e-24, Nx, Ny), G, zeros(Nx, Ny);
                                     k_t = 1e-15, N_t0 = 1.2e6, till_exchange = ImplicitTillExchange(; chi = chiopt))
            for _ in 1:4
                Nt0, chi0 = copy(mc.N_t), copy(mc.chi)
                step!(mc, grid, state, 0.25YR)
                @test mc.info.converged
                for c in findall((state.mask .== GROUNDED) .& (mc.dry .== 0) .& (mc.kappa .> 0))
                    chiopt isa StepStartChi && chi0[c] == 0 && mc.chi[c] > 0 && continue
                    @test min(Nt0[c], state.N[c]) - 2e5 <= mc.N_t[c] <= max(Nt0[c], state.N[c]) + 2e5
                end
            end
            @test any(mc.chi[:, 5:6] .> 0)
        end
    end

    @testset "root finders" begin
        FH = FastHydrology
        # Brent: smooth root, to the tolerance; a jump across zero (no root): the g >= 0 side of the jump
        g = x -> x^3 - 2
        @test FH.brent_root(g, 0.0, 2.0, g(0.0), g(2.0), 1e-13, 100) ≈ cbrt(2) atol = 1e-12
        step_g = x -> x < 0.3 ? 1.0 : -1.0
        for (a, b) in ((0.0, 1.0), (0.29, 0.31), (-5.0, 0.9))
            x = FH.brent_root(step_g, a, b, step_g(a), step_g(b), 1e-12, 200; nonnegative = true)
            @test step_g(x) >= 0
            @test 0.3 - x < 1e-10
        end
        # log-log root of the element laws, against the bisection it replaced
        f = N -> total_flux(p, N, 0.0, 0.3, 100.0, 30 / YR, 2.4e-24, 2e-3)
        for q in (1e-6, 1e-4, 1e-2)
            f(p.N_lo) > q > f(1e8) || continue
            Nb = FH.log_bisect(N -> f(N) - q, p.N_lo, 1e8, 200)
            Nr = FH.log_flux_root(f, q, p.N_lo, 1e8, 60)
            @test Nr ≈ Nb rtol = 1e-10
        end
        # Brent minimiser: the saddle node of the golden section it replaced
        fc = N -> total_flux(p, N, 1.0, 0.0, 100.0, 30 / YR, 2.4e-24, 2e-3)
        Ng, qg = FH.log_golden_min(fc, p.N_lo, p.N_hi, 200)
        Nm, qm = FH.log_brent_min(fc, p.N_lo, p.N_hi, 120)
        @test qm ≈ qg rtol = 1e-10
        @test Nm ≈ Ng rtol = 1e-4
        # the cached saddle node used by the flux balance is the one saddle_node_flux finds
        grid, state = aguamenti_margin()
        m = aguamenti_model(grid)
        update_steady_state!(m, grid, state)
        gr = findall((state.mask .== GROUNDED) .& (m.kappa .< 1))
        for c in gr[1:min(end, 20)]
            q_sn, N_sn = saddle_node_flux(m.params, m.kappa[c], m.Psi[c], m.abs_v_b[c], m.A_visc[c], m.D_s[c])
            @test m.q_sn[c] == q_sn
            @test m.N_sn[c] == N_sn
        end
    end


    # ---------------------------------------------------------------------------------------------
    # Verification of manuscript v2 (Sect. 5.1, Tables 4 and B1, Appendix A, Fig. 9) and its options
    # (Table B2: turbulent sheet law, channels disabled).
    # ---------------------------------------------------------------------------------------------

    @testset "v2 Sect. 5.1 / Table B1: single-element inverses, Eqs. (11b), (13b), (16d)" begin
        # 2000 draws over the ranges of Table B1; the numerical root of q(N) = q (log-space Brent, as in the
        # flux balance) reproduces the closed-form inverse to a relative error below 2e-10 (v2: Newton in
        # ln N; achieved here ~5e-13 for the sheet, ~4e-15 for channel and canal).
        F = FastHydrology
        for law in (LaminarTurbulentSheetLaw(), TurbulentSheetLaw())
            p0 = aguamenti_model(grid1; sheet_law = law).params
            err_sheet = err_channel = err_canal = 0.0
            for n in 1:2000
                d(k) = aguamenti_draw(n, k)
                Psi = aguamenti_logu(d(1), 10.0, 1e3)
                ub = aguamenti_logu(d(2), 1.0, 1e3) / YR
                pn = aguamenti_remake(p0; k_d = aguamenti_logu(d(3), 1e-3, 1e-1), h_r = aguamenti_logu(d(4), 0.05, 0.5),
                                      l_r = aguamenti_logu(d(5), 1.0, 10.0), l_h = aguamenti_logu(d(6), 2e3, 2e4),
                                      l_s = aguamenti_logu(d(7), 1e2, 1e3))
                # sheet: 1e-3 to 0.98 of its saturation flux, Eq. (11b)
                q = aguamenti_logu(d(8), 1e-3, 0.98) * sheet_saturation_flux(pn, Psi)
                N_c = sheet_N_for_flux(pn, q, Psi, ub, A)
                N_n = F.log_flux_root(N -> sheet_flux(pn, N, Psi, ub, A), q, pn.N_lo, pn.N_hi, pn.root_iters)
                err_sheet = max(err_sheet, abs(N_n / N_c - 1))
                # channel, Eq. (13b), and canal, Eq. (16d): 1e-6 to 1e-1 m2/s
                q = aguamenti_logu(d(9), 1e-6, 1e-1)
                N_n = F.log_flux_root(N -> channel_discharge(pn, N, Psi, A) / pn.l_h, q, pn.N_lo, pn.N_hi, pn.root_iters)
                err_channel = max(err_channel, abs(N_n / channel_N_for_flux(pn, q, Psi, A) - 1))
                q = aguamenti_logu(d(10), 1e-6, 1e-1)
                N_n = F.log_flux_root(N -> canal_discharge(pn, N, Psi, A, 2e-3) / pn.l_s, q, pn.N_lo, pn.N_hi, pn.root_iters)
                err_canal = max(err_canal, abs(N_n / canal_N_for_flux(pn, q, Psi, A, 2e-3) - 1))
            end
            @test err_sheet < 2e-10
            @test err_channel < 2e-10
            @test err_canal < 2e-10
        end
    end

    # Laminar hard-bed scaling of Appendix A2-A3: nu = N/N*_h with N*_h = (u_b/(l_r c_d A))^(1/n), and
    # J = (1 + nu^3)^-3 + chi K nu^15 with K = Q_h(N*_h)/(l_h q_h,sat). The saddle node (Eq. A3):
    # K = (3/5) nu_m^-12 (1 + nu_m^3)^-4 and J_sn = (1 + (8/5) nu_m^3) (1 + nu_m^3)^-4.
    sn_K(nu) = 0.6 * nu^-12 / (1 + nu^3)^4
    sn_J(nu) = (1 + 1.6 * nu^3) / (1 + nu^3)^4
    p_lam = aguamenti_model(grid1; Re_c = 1e30).params
    N_star_h = (u_b / (p_lam.l_r * p_lam.c_d * A))^(1 / p_lam.n)

    @testset "v2 Appendix A3: saddle-node flux against the parametric solution, Eq. (A3)" begin
        # 25 values of nu_m in [0.1, 10] (Table B1), i.e. K from 6e11 to 6e-25 (36 orders of magnitude),
        # set through l_h (the 36 orders of magnitude are those of the formula, not a check). v2 quotes a
        # relative error of 1e-14 with a minimisation tolerance of 1e-12. The code's minimiser
        # (log_brent_min, Brent in x = ln N) stops at the tolerance sqrt(eps) (1 + |x|), about 2e-7 at
        # N ~ 2 MPa (|x| ~ 14.5), not at sqrt(eps): N_m is found to ~1e-7 (about 7 sqrt(eps)) and q_sn, flat
        # there, to ~1e-12 (worst 7.4e-13).
        Psi = 100.0
        q_sat = sheet_saturation_flux(p_lam, Psi)
        @test N_star_h ≈ 2.07e6 rtol = 0.01
        for nu in exp10.(range(-1, 1; length = 25))
            K = sn_K(nu)
            pK = aguamenti_remake(p_lam; l_h = channel_discharge(p_lam, N_star_h, Psi, A) / (K * q_sat))
            q_sn, N_m = saddle_node_flux(pK, 0.0, Psi, u_b, A, 2e-3)
            @test q_sn / q_sat ≈ sn_J(nu) rtol = 2e-12
            @test N_m / N_star_h ≈ nu rtol = 1e-6
        end
    end

    @testset "v2 Appendix A3: hysteresis ratios q_h,sat/q_sn" begin
        # The gradients 10, 100, 1000 Pa/m of Fig. 9b correspond to K = 1.6e9, 16, 1.6e-7 (K ~ Psi^-8) ...
        nu_of_K(K) = exp(FastHydrology.brent_root(x -> log(sn_K(exp(x))) - log(K), log(1e-2), log(1e2),
                                                  log(sn_K(1e-2)) - log(K), log(sn_K(1e2)) - log(K), 1e-14, 200))
        for (Psi, K_v2, ratio_lam, ratio_re) in ((10.0, 1.6e9, 1.0, 1.0), (100.0, 16.0, 2.0, 1.7), (1000.0, 1.6e-7, 240.0, 80.0))
            q_sat = sheet_saturation_flux(p_lam, Psi)
            K = channel_discharge(p_lam, N_star_h, Psi, A) / (p_lam.l_h * q_sat)
            @test K ≈ K_v2 rtol = 0.02                       # 1.623e9, 16.23, 1.623e-7
            # ... and to hysteresis ratios 1.0, 2.0 and 240 in the laminar limit (Eq. A3) ...
            @test 1 / sn_J(nu_of_K(K)) ≈ ratio_lam rtol = 0.03     # 1.010, 2.047, 240.2
            q_sn, _ = saddle_node_flux(p_lam, 0.0, Psi, u_b, A, 2e-3)
            @test q_sat / q_sn ≈ 1 / sn_J(nu_of_K(K)) rtol = 1e-10
            # ... reduced to 1.7 and 80 by the Reynolds correction (defaults, u_b = 100 m/a). The code gives
            # 1.010, 1.722 and 83.4: the last is 4 % above v2's rounded 80 (regression values pinned).
            q_sn, _ = saddle_node_flux(p, 0.0, Psi, u_b, A, 2e-3)
            @test sheet_saturation_flux(p, Psi) / q_sn ≈ ratio_re rtol = (Psi == 1000.0 ? 0.05 : 0.02)
            @test sheet_saturation_flux(p, Psi) / q_sn ≈ Dict(10.0 => 1.009771, 100.0 => 1.721896, 1000.0 => 83.37205)[Psi] rtol = 1e-5
        end
        # Channel growth time at onset for Psi = 1000 Pa/m: v2 quotes 0.2 a, the code gives 0.156 a
        # (22 % below; the element-library test accepts it at rtol 0.25). Pinned as a regression value.
        @test 1 / channel_growth_rate(p, onset_flux(p, 1000.0, u_b), 1000.0, u_b) / YR ≈ 0.156365 rtol = 1e-5
    end

    @testset "v2 Sect. 3.6 / Fig. 9a: hard-bed bifurcation at Psi = 100 Pa/m, u_b = 100 m/a" begin
        Psi, p_i, D_s = 100.0, 1e7, 2e-3
        q_sat = sheet_saturation_flux(p, Psi)
        q_sn, N_sn = saddle_node_flux(p, 0.0, Psi, u_b, A, D_s)
        q_on = onset_flux(p, Psi, u_b)
        @test q_sat * YR ≈ 2e4 rtol = 0.15       # "2e4 m2/a" (code: 2.26e4)
        @test q_sn * YR ≈ 1.3e4 rtol = 0.02      # "1.3e4 m2/a" (code: 1.311e4)
        @test q_on == q_sat                      # q_Lambda lies above q_h,sat (Fig. 9 caption)
        @test critical_flux(p, Psi, u_b) > q_sat
        @test 1 / channel_growth_rate(p, q_on, Psi, u_b) / YR ≈ 6 rtol = 0.05   # "1/Lambda_0 = 6 a" (code: 5.8)
        # beyond q_h,sat the cell lies on the channelized branch "at about 1.6 MPa" (code: 1.633 MPa)
        N_ch, channelized = flux_balance_root(p, 1.0001q_sat, 1.0, 0.0, Psi, u_b, A, D_s, p_i, q_sn; N_sn)
        @test channelized
        @test N_ch ≈ 1.6e6 rtol = 0.03
        # The hysteresis loop (arrows of Fig. 9a), with the steady indicator of Eq. (21): the flux rises
        # from 0.5 q_sn to 1.5 q_h,sat and falls back.
        qs = q_sat .* exp10.(range(log10(0.5q_sn / q_sat), log10(1.5); length = 201))
        function sweep(qpath, chi)
            out = Tuple{Float64, Float64, Float64}[]
            for q in qpath
                chi = steady_channel_indicator(chi, q, q_on, q_sn)
                N, _ = flux_balance_root(p, q, chi, 0.0, Psi, u_b, A, D_s, p_i, q_sn; N_sn)
                push!(out, (q, chi, N))
            end
            return out, chi
        end
        up, chi = sweep(qs, 0.0)
        down, _ = sweep(reverse(qs), chi)
        down = reverse(down)
        for ((q, chi_u, N_u), (_, chi_d, N_d)) in zip(up, down)
            if q < q_sn                          # sheet only, both ways
                @test chi_u == chi_d == 0 && N_u == N_d && N_u > 0
            elseif q < q_on                      # bistable band: sheet going up, channels coming down
                @test chi_u == 0 && chi_d == 1
                @test N_d > N_u > 0
            else                                 # channelized both ways (uplift only while channels develop)
                @test chi_u == chi_d == 1 && N_u == N_d && N_u > 1e6
            end
        end
    end

    @testset "v2 Table B1: till reservoir under a periodic N, exponential step of Eq. (28a)" begin
        # T_t = 10 a, N(t) = 0.3 + 0.1 sin(2 pi t/P) MPa, P = 100 a, three periods, N_t(0) = 0.3 MPa. Exact
        # solution of dN_t/dt = -(N_t - N)/T_t: N_t = N_0 + a (sin wt - w T_t cos wt)/(1 + (w T_t)^2) + C e^(-t/T_t).
        T_t, P, a = 10YR, 100YR, 0.1e6
        w = 2pi / P
        N_of(t) = 0.3e6 + a * sin(w * t)
        part(t) = 0.3e6 + a * (sin(w * t) - w * T_t * cos(w * t)) / (1 + (w * T_t)^2)
        exact(t) = part(t) + (0.3e6 - part(0.0)) * exp(-t / T_t)
        err = Dict{Float64, Float64}()
        for dt in (0.1YR, 1YR, 10YR, 100YR)
            N_t, e, e_first, e_last = 0.3e6, 0.0, 0.0, 0.0
            for k in 1:round(Int, 3P / dt)
                N_t = FastHydrology.till_exponential_step(N_t, N_of(k * dt), exp(-dt / T_t))
                d = abs(N_t - exact(k * dt))
                e = max(e, d)
                k * dt <= P && (e_first = max(e_first, d))
                k * dt > 2P && (e_last = max(e_last, d))
            end
            err[dt / YR] = e / a
            # stable: the error does not grow over the three periods (at dt = P it is 0.45 in both, equal
            # to 5e-5 relative)
            @test e_last <= e_first * (1 + 1e-4)
        end
        # 0.3 % and 3 % of the amplitude at dt = T_t/100 and T_t/10 (code: 0.266 % and 2.62 %), first order
        @test err[0.1] ≈ 0.003 rtol = 0.2
        @test err[1.0] ≈ 0.03 rtol = 0.2
        @test err[0.1] ≈ 0.0026619 rtol = 1e-3
        @test err[1.0] ≈ 0.026217 rtol = 1e-3
        @test 8 < err[1.0] / err[0.1] < 12
        @test err[10.0] < 1 && err[100.0] < 1
    end

    @testset "v2 Appendix A4: steady undrained offset under thinning, (1 - B) rho_i g r T_t" begin
        # r = 1 m/a: 2.3, 24 and 236 kPa for k_t = 1e-16, 1e-17, 1e-18 (h_t = 5 m), with T_t of Eq. (19c) at
        # N_t = 0.2 MPa, the value of the Fig. 6 caption (code: 2.364, 23.64, 236.4 kPa; 0.1 or 0.3 MPa would
        # give about 2x or 0.7x). The first is 3 % above v2's 2.3 kPa (2.4 when rounded).
        r = 1 / YR
        rate = (1 - p.B) * p.rho_i * p.g * r
        for (k_t, offset_v2) in ((1e-16, 2.3e3), (1e-17, 24e3), (1e-18, 236e3))
            T_t = till_consolidation_time(p, 0.2e6, 5.0, k_t)
            @test rate * T_t ≈ offset_v2 rtol = 0.03
            @test rate * T_t ≈ 2.364128e3 * (1e-16 / k_t) rtol = 1e-6
            # the Eq. (28a) step (relaxation, then the undrained term) holds a discrete offset
            # rate dt/(1 - exp(-dt/T_t)) = rate T_t (1 + dt/(2 T_t) + ...), which tends to it for dt << T_t
            dt = T_t / 1000
            N, N_t = 0.3e6, 0.3e6
            for _ in 1:40_000
                N_t = FastHydrology.till_exponential_step(N_t, N, exp(-dt / T_t), -rate * dt)
            end
            @test N - N_t ≈ rate * dt / (-expm1(-dt / T_t)) rtol = 1e-9
            @test N - N_t ≈ rate * T_t rtol = 1e-3
        end
        # the full till step (T_t at the current N_t, bounds) settles at the same offset, and at the discrete
        # offset of the T_t it ends with
        k_t, dt = 1e-16, 0.01YR
        T_t = till_consolidation_time(p, 0.2e6, 5.0, k_t)
        N = 0.2e6 + rate * T_t
        N_t = N
        for _ in 1:5000
            N_t, _, _, _ = till_relax(p, N_t, N, dt, 5.0, k_t, 1e7, -r)
        end
        @test N_t ≈ 0.2e6 rtol = 2e-3
        T_end = till_consolidation_time(p, N_t, 5.0, k_t)
        @test (N - N_t) / (rate * dt) ≈ 1 / (-expm1(-dt / T_end)) rtol = 1e-6
        # N_t from 0.3 to 0.1 MPa (h_t = 5 m): v2 quotes 0.3 m of water exchanged. The code's storage law, the
        # water (1 - theta) h_t de of Appendix D3 with theta at the start, gives 0.21 m (taken up by the till,
        # a positive exchange: it softens); the change of theta h_t would be 0.14 m. Pinned as the code's
        # value (an open question for the author).
        e3, e1 = till_void_ratio(p, 0.3e6), till_void_ratio(p, 0.1e6)
        co = FastHydrology.till_relax_coefficients(p, 0.3e6, 1.0, 5.0, k_t, 1e7, 0.0)
        @test co.storage * (e1 - e3) ≈ 0.2055 rtol = 1e-3
        _, demand, _, _ = till_relax(p, 0.3e6, 0.1e6, 1e6 * YR, 5.0, k_t, 1e7, 0.0)
        @test demand * 1e6 * YR ≈ 0.2055 rtol = 1e-3
    end

    @testset "sheet law option: turbulent law of GlaDS (Table B2, Appendix D1)" begin
        F = FastHydrology
        law = TurbulentSheetLaw()
        @test law.k_s == 5e-3
        @test_throws ArgumentError TurbulentSheetLaw(; k_s = -1.0)
        pt = aguamenti_model(grid1; sheet_law = law).params
        @test pt.sheet_law isa TurbulentSheetLaw{Float64}
        @test p.sheet_law isa LaminarTurbulentSheetLaw
        # q = k_s h^(5/4) Psi^(1/2), and its saturation flux (Eq. 12a with this law)
        @test F.sheet_law_flux(pt, law, 0.05, 100.0) ≈ 5e-3 * 0.05^1.25 * 10 rtol = 1e-14
        @test sheet_saturation_flux(pt, 100.0) ≈ 5e-3 * pt.h_r^1.25 * 10 rtol = 1e-14
        @test F.film_saturation_flux(pt, 100.0) ≈ 5e-3 * pt.h_rs^1.25 * 10 rtol = 1e-14
        @test sheet_flux(pt, 1e-3, 100.0, u_b, A) ≈ sheet_saturation_flux(pt, 100.0) rtol = 1e-9
        # inverses: h(q) = (q/(k_s Psi^(1/2)))^(4/5) and N(q) of Eq. (11b)
        for q in (1e-6, 1e-4, 0.5 * sheet_saturation_flux(pt, 100.0))
            h = F.sheet_height_for_flux(pt, q, 100.0)
            @test F.sheet_law_flux(pt, law, h, 100.0) ≈ q rtol = 1e-13
            @test sheet_flux(pt, sheet_N_for_flux(pt, q, 100.0, u_b, A), 100.0, u_b, A) ≈ q rtol = 1e-12
        end
        @test sheet_N_for_flux(pt, 2 * sheet_saturation_flux(pt, 100.0), 100.0, u_b, A) == 0
        # elasticity d ln q/d ln h = 3 phi_1 (Eq. D1), for both laws; phi_2 = 3 phi_1 - 1
        for pp in (p, pt), q in (1e-6, 1e-3, 1e-1, 10.0)
            h = F.sheet_height_for_flux(pp, q, 100.0)
            d = 1e-6
            el = (log(F.sheet_law_flux(pp, pp.sheet_law, h * (1 + d), 100.0)) -
                  log(F.sheet_law_flux(pp, pp.sheet_law, h * (1 - d), 100.0))) / (log1p(d) - log1p(-d))
            phi1, phi2 = F.growth_factors(pp, pp.sheet_law, q)
            @test el ≈ 3phi1 rtol = 1e-7
            @test phi2 ≈ 3phi1 - 1 rtol = 1e-14
        end
        @test F.growth_factors(pt, law, 1.0) == (5 / 12, 1 / 4)
        # critical flux: phi_2 q Psi = rho_i L_w u_b h_r/l_r with phi_2 = 1/4, eight times Hewitt's laminar value
        c = pt.rho_i * pt.L_w * u_b * pt.h_r / (pt.l_r * 100.0)
        @test critical_flux(pt, 100.0, u_b) ≈ 4c rtol = 1e-14
        @test critical_flux(pt, 100.0, u_b) ≈ 8 * critical_flux(p_lam, 100.0, u_b) rtol = 1e-8
        @test onset_flux(pt, 100.0, u_b) == min(sheet_saturation_flux(pt, 100.0), 4c)
        # a pressurized sheet channelizes above q_Lambda (slow sliding, so that q_Lambda < q_h,sat)
        u_slow = 1e-3 * u_b
        q_L = critical_flux(pt, 100.0, u_slow)
        @test q_L < sheet_saturation_flux(pt, 100.0)
        @test channel_growth_rate(pt, 0.9q_L, 100.0, u_slow) == 0
        @test channel_growth_rate(pt, 1.1q_L, 100.0, u_slow) > 0
        q = 1.1q_L; h = F.sheet_height_for_flux(pt, q, 100.0)
        @test channel_growth_rate(pt, q, 100.0, u_slow) ≈
              (q * 100.0 / 4 - pt.rho_i * pt.L_w * u_slow * pt.h_r / pt.l_r) / (pt.rho_i * pt.L_w * h) rtol = 1e-12
        # uplift: Lambda_0 = (5/4) q Psi/(rho_i L_w h_w)
        q = 2 * sheet_saturation_flux(pt, 100.0); h = F.sheet_height_for_flux(pt, q, 100.0)
        @test channel_growth_rate(pt, q, 100.0, u_b) ≈ 1.25 * q * 100.0 / (pt.rho_i * pt.L_w * h) rtol = 1e-12
        # at the onset (q_on = q_h,sat here) the sheet is saturated, h = h_r: Lambda_0 is finite and positive,
        # (5/4) q_on Psi/(rho_i L_w h_r), so channels grow at a finite rate (no 1/0 at q = q_sat)
        for Psi in (10.0, 100.0, 1000.0)
            q_on = onset_flux(pt, Psi, u_b)
            @test q_on == sheet_saturation_flux(pt, Psi)
            L0 = channel_growth_rate(pt, q_on, Psi, u_b)
            @test isfinite(1 / L0) && L0 > 0
            @test L0 ≈ 1.25 * q_on * Psi / (pt.rho_i * pt.L_w * pt.h_r) rtol = 1e-12
        end
        # the saddle node exists and lies below the onset; type-stable evaluations
        q_sn, N_m = saddle_node_flux(pt, 0.0, 100.0, u_b, A, 2e-3)
        @test 0 < q_sn < onset_flux(pt, 100.0, u_b)
        @inferred sheet_flux(pt, 1e5, 100.0, u_b, A)
        @inferred channel_growth_rate(pt, 1e-3, 100.0, u_b)
        @inferred critical_flux(pt, 100.0, u_b)
        # a whole model with the turbulent law
        grid, state = aguamenti_margin()
        m = aguamenti_model(grid; G = 20.0, sheet_law = TurbulentSheetLaw(; k_s = 5e-3))
        update_steady_state!(m, grid, state)
        @test all(isfinite, state.N)
        @test m.info.converged
        step!(m, grid, state, YR)
        @test all(isfinite, m.N_t)
        # Float32: the law takes the model's float type
        g32 = ArrayHydroGrid(3, 3, (0.0f0, 3f3), (0.0f0, 3f3); T = Float32)
        m32 = AguamentiHydroModel(g32, zeros(Float32, 3, 3), fill(Float32(100 / YR), 3, 3), fill(2.4f-24, 3, 3),
                                  fill(0.1f0, 3, 3), zeros(Float32, 3, 3); sheet_law = TurbulentSheetLaw())
        @test m32.params.sheet_law isa TurbulentSheetLaw{Float32}
        @test sheet_saturation_flux(m32.params, 100.0f0) isa Float32
    end

    @testset "options combined: weighted routing, lagged exchange, turbulent sheet, no channels" begin
        for T in (Float64, Float32)
            Nx, Ny = 12, 6
            grid = ArrayHydroGrid(Nx, Ny, (zero(T), T(Nx * 5e3)), (zero(T), T(Ny * 5e3)); T = T)
            L = Nx * grid.dx
            h = [T(600 + 1400 * sqrt(max(0.0, 1 - (i - 0.5) * grid.dx / L))) for i in 1:Nx, j in 1:Ny]
            b = [T(-100 - 200 * (i - 0.5) * grid.dx / L) for i in 1:Nx, j in 1:Ny]
            mask = fill(GROUNDED, Nx, Ny); mask[end, :] .= OCEAN
            state = HydroState(grid, HydroMask(mask), h, b)
            m = AguamentiHydroModel(grid, fill(T(0.3), Nx, Ny), fill(T(100 / YR), Nx, Ny), fill(T(2.4e-24), Nx, Ny),
                                    fill(T(5), Nx, Ny), zeros(T, Nx, Ny); routing_potential = WeightedRouting(T(0.5)),
                                    till_exchange = LaggedTillExchange(), sheet_law = TurbulentSheetLaw(), channels = false,
                                    k_t = 1e-15, rtol = T == Float32 ? 1e-4 : 1e-6)
            @test m.params.sheet_law isa TurbulentSheetLaw{T} && m.routing_potential isa WeightedRouting{T}
            update_steady_state!(m, grid, state)
            @test eltype(state.N) == T && all(isfinite, state.N)
            m.N_t .= T(1e6)                       # out of equilibrium: the lagged exchange is not zero
            g = state.mask .== GROUNDED
            area = grid.dx * grid.dy
            for k in 1:3
                step!(m, grid, state, T(0.5YR))
                # (the till releases water on the first step; it then sits at N_max, below N, and stops)
                k == 1 && @test any(m.mdot_t .< 0)
                @test m.info.converged
                @test all(isfinite, state.N) && all(isfinite, m.N_t) && eltype(m.N_t) == T
                @test all(m.chi .== 0) && all(m.q_ch .== 0)
                # the step's graph is Eq. (26) at N^(j-1), with chi^(j) = 0: gamma0 N^(j-1), clipped
                C = copy(m.route_corr)
                @test all(C[g] .<= T(0.5) .* m.N_lag[g] .* (1 + 10eps(T)))
                FastHydrology.weighted_routing_correction!(m, grid, state, T(0.5), m.chi_start, m.N_lag)
                @test C == m.route_corr
                # the water budget closes: supply = outflow to the ocean + till uptake
                supply = sum(m.mdot_total[i, j] / m.params.rho_w for i in 1:Nx-1, j in 1:Ny) * area
                @test supply ≈ sum(m.psi_out[Nx - 1, :]) + sum(m.mdot_t[1:Nx-1, :]) * area + sum(m.refreeze) * area rtol = (T == Float32 ? 1e-4 : 1e-8)
            end
        end
    end

    @testset "channels = false (Table B2, numerical laboratory)" begin
        grid, state = aguamenti_margin()
        @test_throws ArgumentError aguamenti_model(grid; channels = false, chi0 = 0.5)
        m_on = aguamenti_model(grid; G = 20.0)
        update_steady_state!(m_on, grid, state)
        @test any(m_on.chi .== 1)                 # the same set-up forms channels when they are on
        _, state_off = aguamenti_margin()
        m = aguamenti_model(grid; G = 20.0, channels = false)
        update_steady_state!(m, grid, state_off)
        g = state.mask .== GROUNDED
        @test all(m.chi .== 0)
        @test all(m.q_ch .== 0)
        @test all(isinf, m.q_on[g]) && all(m.q_sn .== 0) && all(m.N_sn .== 0)
        @test all(m.Lambda0 .== 0)
        # without channels the uplifted cells stay uplifted: N no higher than with channels
        @test all(state_off.N[g] .<= state.N[g] .+ 1e-6)
        for _ in 1:5
            step!(m, grid, state_off, 10YR)
        end
        @test all(m.chi .== 0)
        @test all(isfinite, state_off.N)
        # the flux balance never takes the channelized branch, whatever chi it is handed
        pn = aguamenti_remake(p; channels = false)
        q = 2 * sheet_saturation_flux(p, 100.0)
        @test flux_balance_root(pn, q, 1.0, 0.0, 100.0, u_b, A, 2e-3, 1e7, 0.0) == (0.0, false)
        @test flux_balance_root(p, q, 1.0, 0.0, 100.0, u_b, A, 2e-3, 1e7, 0.0)[2]
        @test saddle_node_flux(pn, 0.0, 100.0, u_b, A, 2e-3) == (0.0, 0.0)
    end

    @testset "Table B2 parameters can be passed" begin
        grid, state = aguamenti_margin()
        nx, ny = grid.Nx, grid.Ny
        m = AguamentiHydroModel(grid, zeros(nx, ny), fill(1e-6, nx, ny), fill(2.5e-25, nx, ny), fill(0.1, nx, ny),
                                zeros(nx, ny); rho_i = 910.0, c_0 = 100e3, glen_n = 3, k_d = 0.01, h_r = 0.1, l_r = 2.0,
                                Re_c = 1000.0, l_h = 10e3, f = 0.1, h_rs = 0.02, l_rs = 0.5, D_s = 2e-3, S_c = 0.045,
                                drho_s = 1650.0, l_s = 300.0, F_b = 5.0, sheet_law = TurbulentSheetLaw(; k_s = 5e-3),
                                channels = false)
        @test m.params.rho_i == 910 && m.params.n == 3 && m.params.F_b == 5
        @test all(m.c_0 .== 100e3) && all(m.A_visc .== 2.5e-25) && all(m.abs_v_b .== 1e-6)
        @test m.params.sheet_law.k_s == 5e-3 && !m.params.channels
        update_steady_state!(m, grid, state)
        @test all(isfinite, state.N)
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
