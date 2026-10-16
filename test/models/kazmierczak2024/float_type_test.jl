# T = Float32 stays Float32 end to end: the FFT cache, the StaggeredFriction arrays and the smoothing kernel
# follow the grid's float type, and a Float32 model agrees with Float64 to single-precision accuracy.

@testset "Float type consistency (Float32 model)" begin

    Nx, Ny, dx = 40, 30, 2e3
    function build_model(T; scheme = Warner())
        grid = ArrayHydroGrid(Nx, Ny, (0.0, Nx * dx), (0.0, Ny * dx); T)
        x = [(i - 0.5) * dx for i in 1:Nx, j in 1:Ny]; y = [(j - 0.5) * dx for i in 1:Nx, j in 1:Ny]
        b = -2e-3 .* x .- 1e-3 .* y .+ 20 .* sin.(x ./ 3e3) .* cos.(y ./ 4e3)
        u = perYear2perSecond(100.0)
        ux = u .* (1 .+ 0.5 .* sin.(x ./ 5e3)); uy = 0.3u .* cos.(y ./ 7e3)
        z = zeros(Nx, Ny); G = fill(1e-6, Nx, Ny) .* FastHydrology.KAZMIERCZAK_DEFAULT_L_W
        model = KazmierczakHydroModel(grid, z, hypot.(ux, uy), z .+ 1e-24, G, zero(G); coupling_length_kamb86 = 2.0,
                    dissipation_melt = true, dissipation_verbose = false, qN_verbose = false, routing_scheme = scheme,
                    fill_algorithm = PriorityFloodFill(), sliding_law = RegularizedCoulombSlidingLaw(c_till = 0.5),
                    friction_discretization = StaggeredFriction(ux, uy))
        state = HydroState(grid, ones(Nx, Ny), fill(1000.0, Nx, Ny), b)
        update_steady_state!(model, grid, state)
        return grid, model, state
    end

    g64, m64, s64 = build_model(Float64)
    g32, m32, s32 = build_model(Float32)

    # unset Picard tolerances: the double-precision values, raised to 100 eps(T) where unreachable
    @test (m64.qN_rtol, m64.dissipation_rtol) == (1e-8, 1e-12)
    @test m32.qN_rtol === m32.dissipation_rtol === 100 * eps(Float32)
    @test eltype(s32.N) === Float32 && eltype(m32.q) === Float32
    @test g32.conv_cache[] isa FastHydrology.FFTConvCache{Float32}
    @test g64.conv_cache[] isa FastHydrology.FFTConvCache{Float64}
    fd = m32.friction_discretization
    @test fd isa StaggeredFriction{Float32} && eltype(fd.tx) === Float32 && fd.u_floor isa Float32
    @test eltype(FastHydrology.coupling_kernel(m32, g32, 1000.0)) === Float32
    @test eltype(FastHydrology.coupling_kernel(m64, g64, 1000.0)) === Float64

    relerr(a, b) = maximum(abs.(Float64.(a) .- b)) / maximum(abs.(b))
    @test relerr(s32.N, s64.N) < 1e-3
    @test relerr(m32.q, m64.q) < 1e-3

    # the 8-neighbour angle split (bare `pi` literals used to promote it to Float64) agrees as well
    _, _, t64 = build_model(Float64; scheme = ModifiedTarboton())
    _, tm32, t32 = build_model(Float32; scheme = ModifiedTarboton())
    @test eltype(tm32.routing_tape.w8) === Float32 && relerr(t32.N, t64.N) < 1e-3

    # a cache built for one float type is rebuilt, not reused, for another
    ref = Ref{Any}(nothing)
    c64 = FastHydrology.get_fft_conv_cache!(ref, Float64, 20, 15, 2, 1)
    c32 = FastHydrology.get_fft_conv_cache!(ref, Float32, 20, 15, 2, 1)
    @test c32 isa FastHydrology.FFTConvCache{Float32} && c32 !== c64
    # a model keeps its StaggeredFriction scratch: no per-sweep allocation of the face tractions
    FastHydrology.add_friction_term!(m32)
    @test (@allocated FastHydrology.add_friction_term!(m32)) < 2_000
end
