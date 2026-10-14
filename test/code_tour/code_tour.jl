import Pkg; Pkg.activate(@__DIR__) # own environment (test/code_tour/Project.toml): FastHydrology and Shakti developed from the local repositories, plus CairoMakie, NCDatasets, MAT, Preferences

# =============================================================================================
#                           F A S T H Y D R O L O G Y   C O D E   T O U R
# =============================================================================================
#
# A runnable walk through EVERY function, method, struct, constant and abstract type of
# FastHydrology.jl and its two package extensions, in the order the package loads its source files
# (the `include` list in src/FastHydrology.jl). PART n tours one file; the PART numbers are those of
# the companion guide (test/code_guide/fasthydrology_code_guide.tex, Sect. 6, where every
# subsection starts with "Tour PART n."), so the guide gives the background and this script shows
# the code doing it.
#
# How to read it
# --------------
# Each PART starts with a banner and a short prose block saying what the file is for and why it is
# built the way it is. Then every definition of that file is called (or constructed) on a small
# fixture, a compact result is printed, and a `check` compares it with something independent: a
# value computed by hand, an invariant (conservation of routed water, symmetry, a bound), or a
# second code path that must agree (the recursive, iterative and taped routing; the fused and the
# step-by-step effective pressure; the cached FFT convolution and ImageFiltering's). A check prints
# "ok" or "FAIL"; the script carries on after a FAIL and exits with an error at the end if any
# check failed.
#
# Some definitions need objects from later files: `update_Po!` (PART 6) needs a model, and the
# first model, `HABHydroModel`, is defined in PART 19. A small shared fixture is therefore built
# once, right after the helpers, with those later functions, and reused everywhere; the PART that
# tours a function used early says so.
#
# How to run it
# -------------
#     sbatch ~/build_jobs/fh_tour.sbatch          (julia -t 4 test/code_tour/code_tour.jl on a compute node)
# The tour runs on the CPU in Float64 (and Float32 where the float type matters). Shakti is
# loaded with its "Threads" backend and without CUDA (test/code_tour/LocalPreferences.toml), only
# to run the FastHydrologyShaktiExt extension. PART 25 runs the models end to end on the real
# Thwaites 2 km (Kazmierczak et al. 2024) and Antarctica 16 km (Yelmo restart) inputs under
# test/Kazmierczak et al 2024/input/; it is skipped, with a note, if they are absent. Output files
# (checkpoints, NetCDF series, figures) go to test/code_tour/output/.
#
# At the end a coverage summary parses every source file for its definitions (at run time, from
# the installed package's src/ and ext/) and lists any name the tour did not touch.
# =============================================================================================

using FastHydrology
import CairoMakie   # `import`, not `using`: activates FastHydrologyMakieExt without bringing Makie's exports into scope
import Shakti       # `import`, not `using`: FastHydrology and Shakti both export `run!` and `step!`
using NCDatasets, MAT
using Statistics: mean, median, quantile, cor
using LinearAlgebra: norm
import Test         # only for Test.collect_test_logs, to check that a warning is (or is not) emitted
const FH = FastHydrology

const TOUR_T0 = time()
const OUT = mkpath(joinpath(@__DIR__, "output"))
const DATA = joinpath(dirname(@__DIR__), "Kazmierczak et al 2024", "input")

# ---------------------------------------------------------------------------------------------
# Tour helpers (not part of FastHydrology)
# ---------------------------------------------------------------------------------------------
const NCHECK = Ref(0)
const FAILED = String[]
const CURFILE = Ref("")
const TOUCHED = Set{String}()              # "file.jl:name" of every definition exercised
const SKIPPED = Dict{String, String}()     # "file.jl:name" => reason, for definitions that cannot run here

"""Record a check: prints ok/FAIL and carries on."""
function check(cond::Bool, what::AbstractString)
    NCHECK[] += 1
    if cond
        println("    ok    ", what)
    else
        push!(FAILED, CURFILE[] * ": " * what)
        println("    FAIL  ", what)
    end
    return cond
end

"""Mark definitions of the current file (or of `file`) as exercised."""
covered!(names::AbstractString...; file = CURFILE[]) = foreach(n -> push!(TOUCHED, file * ":" * n), names)
skipped!(name::AbstractString, reason::AbstractString; file = CURFILE[]) = (SKIPPED[file * ":" * name] = reason)

function part(n, file, title)
    CURFILE[] = file
    println()
    println("="^100)
    loc = isempty(file) ? "" : "   (" * file * ")"
    println("PART $n -- $title", loc, "   [", round(time() - TOUR_T0; digits = 1), " s]")
    println("="^100)
end
sub(title) = println("\n--- ", title, " ", "-"^max(0, 90 - length(title)))
show1(label, x) = println("    ", rpad(label, 50), " = ", x)

"""True if `f()` throws an exception of type `T`."""
throws(f, T = Exception) = try f(); false catch e; e isa T end

"""Run `f()`, returning (result, the log records it emitted)."""
function with_logs(f)
    logs, val = Test.collect_test_logs(f; min_level = Base.CoreLogging.Info)
    return val, logs
end
warned(logs, pat) = any(l -> l.level == Base.CoreLogging.Warn && occursin(pat, string(l.message)), logs)

maxabs(x) = maximum(abs, x; init = 0.0)
relerr(a, b) = maxabs(a .- b) / max(maxabs(b), floatmin(Float64))
"""Quantiles of the finite entries only (a NaN would make `quantile` throw)."""
qfin(v, p) = quantile(filter(isfinite, v), p)
"""Seconds `f()` takes once compiled: one untimed call, then the best of three."""
tbest(f) = (f(); minimum(@elapsed(f()) for _ in 1:3))

# ---------------------------------------------------------------------------------------------
# Shared fixture: a 30 x 24 slab, dx != dy, with bumps in the bed and a floating strip
# ---------------------------------------------------------------------------------------------
# The geometric potential phi0 = rho_i g h + rho_w g b falls towards +x (ice thins, bed deepens),
# and a sin*cos bump of +-25 m on the bed makes the flow paths converge and diverge, so routing has
# real work to do. The bump is too weak to reverse the slope (its potential gradient is at most
# 98 Pa/m against the background's 174 Pa/m), so this potential has no pits; the filling checks of
# PART 14 use the same fixture with 80 m bumps (314 Pa/m), which has. The last three columns are floating (mask 0): water reaching them leaves the
# grounded system there (the grounding line). dx = 1000 m, dy = 1250 m, so every place where the
# code treats x and y separately is exercised.
const NX, NY, DX, DY = 30, 24, 1000.0, 1250.0
function k24_inputs(; Nx = NX, Ny = NY, dx = DX, dy = DY, nfloat = 3, bumps = 25.0, T = Float64)
    grid = ArrayHydroGrid(Nx, Ny, (0.0, Nx * dx), (0.0, Ny * dy); T = T)
    x = [(i - 0.5) * dx for i in 1:Nx, j in 1:Ny]
    y = [(j - 0.5) * dy for i in 1:Nx, j in 1:Ny]
    h = 900.0 .- 0.015 .* x .+ 30.0 .* sin.(y ./ 5e3)
    b = -150.0 .- 4e-3 .* x .+ bumps .* sin.(x ./ 2.5e3) .* cos.(y ./ 3.5e3)
    mask = ones(Nx, Ny); mask[Nx-nfloat+1:Nx, :] .= 0.0
    kappa = clamp.(x ./ (Nx * dx), 0.0, 1.0)                 # hard bed upstream, soft towards the grounding line
    abs_v_b = perYear2perSecond.(20.0 .+ 200.0 .* x ./ (Nx * dx))
    A_visc = fill(2.4e-24, Nx, Ny)                          # Glen's A [Pa^-3 s^-1]
    G = fill(0.07, Nx, Ny)                                  # geothermal heat [W m^-2]
    q_T = fill(0.03, Nx, Ny)                                # conduction into the ice [W m^-2]
    i_eb = zeros(Nx, Ny); i_eb[8, 12] = 2e-4                # a moulin-like point input [kg m^-2 s^-1]
    cv(a) = T.(a)
    return (; grid, x, y, h = cv(h), b = cv(b), mask = cv(mask), kappa = cv(kappa), abs_v_b = cv(abs_v_b),
              A_visc = cv(A_visc), G = cv(G), q_T = cv(q_T), i_eb = cv(i_eb))
end
const INP = k24_inputs()
const GRID = INP.grid

newstate(inp = INP) = HydroState(inp.grid, inp.mask, inp.h, inp.b)
newk24(inp = INP; kw...) = KazmierczakHydroModel(inp.grid, inp.kappa, inp.abs_v_b, inp.A_visc, inp.G, inp.q_T;
                                                 i_eb = inp.i_eb, coupling_length_kamb86 = 2.0,
                                                 dissipation_verbose = false, qN_verbose = false, kw...)

# A grid type that implements only the grid interface's required fields: it exercises the generic
# defaults of grid.jl (the error stubs, and the ImageFiltering fallback of convolve!).
struct BareGrid <: AbstractHydroGrid
    Nx::Int
    Ny::Int
    dx::Float64
    dy::Float64
end
struct DummyModel <: FH.AbstractHydroModel end
struct EmptyState <: FH.AbstractHydroState end

# =============================================================================================
# PART 0 -- FastHydrology.jl: the module, its dependencies and its exports
# =============================================================================================
# src/FastHydrology.jl loads eight dependencies, includes the 25 source files (11 common, 7 for the
# Kazmierczak et al. 2024 model, 3 for HAB, 1 for the Shakti wrapper) in a fixed order, and
# declares the exported names grouped by the file that defines them. There are no preferences, no
# backend switch and no compile-time constants: everything runs on plain `Array`s, and the float
# type is the grid's (`ArrayHydroGrid(...; T = Float32)` gives a Float32 model).
#
# Two optional dependencies are package extensions: CairoMakie (plotting, ext/FastHydrologyMakieExt.jl)
# and Shakti (the time-evolving model, ext/FastHydrologyShaktiExt.jl). Their entry points
# (`visualize_field`, `visualize_grid`, `step!`) are declared in src/ as functions with no methods,
# so the names belong to FastHydrology and the extension only adds methods.
part(0, "FastHydrology.jl", "THE MODULE: DEPENDENCIES, INCLUDE ORDER, EXPORTS")
exported = filter(n -> n !== :FastHydrology, names(FastHydrology))
show1("number of exported names", length(exported))
check(all(n -> isdefined(FastHydrology, n), exported), "every exported name is defined (no stale export)")
check(all(n -> n in exported, (:ArrayHydroGrid, :HydroState, :KazmierczakHydroModel, :HABHydroModel, :ShaktiHydroModel,
                               :SteadyStateSimulation, :TimeSimulation, :run!, :step!, :update_steady_state!, :N_from_ub!)),
      "the public API (grid, state, models, simulations, run!/step!, N_from_ub!) is exported")
check(!(:KazmierczakParams in exported) && !(:DissipationMeltOn in exported) && isdefined(FH, :KazmierczakParams),
      "internal types (KazmierczakParams, DissipationMeltOn, ...) exist but are not exported")
for dep in (:ImageFiltering, :FFTW, :OffsetArrays, :DocStringExtensions, :MAT, :NCDatasets)
    check(isdefined(FH, dep), "dependency $dep is loaded into the module")
end
check(Base.get_extension(FastHydrology, :FastHydrologyMakieExt) !== nothing, "importing CairoMakie activated FastHydrologyMakieExt")
check(Base.get_extension(FastHydrology, :FastHydrologyShaktiExt) !== nothing, "importing Shakti activated FastHydrologyShaktiExt")

# =============================================================================================
# PART 1 -- common/grid.jl: the grid interface and ArrayHydroGrid
# =============================================================================================
# Every model is written against `AbstractHydroGrid`, whose interface is: the fields Nx, Ny, dx,
# dy (read directly, no accessors), `alloc_field` (the only place that knows the field type), and
# a handful of field operations with plain-array defaults: `fill_halo!` (a no-op),
# `convolve!`, `minus_gradient_x!/y!`, `masked_mean`, `masked_max_abs(_diff)`, `overwrite_where!`.
# `ArrayHydroGrid` is the only concrete grid since v2.0.0 (the Oceananigans grid was removed). It
# is cell-centred: `xlims`/`ylims` are the outer EDGES of the domain, so dx = (x2 - x1)/Nx.
# The masked reductions fuse the mask test into one branchless pass (`ifelse`), which the source
# records as faster than collecting the grounded indices first.
part(1, "common/grid.jl", "THE GRID INTERFACE AND ArrayHydroGrid")
g = ArrayHydroGrid(4, 3, (0.0, 8.0), (10.0, 19.0))
show1("ArrayHydroGrid(4, 3, (0,8), (10,19)): Nx, Ny, dx, dy", (g.Nx, g.Ny, g.dx, g.dy))
check(g.dx == 2.0 && g.dy == 3.0 && g isa ArrayHydroGrid{Float64}, "cell-centred: dx = (8-0)/4, dy = (19-10)/3, element type Float64")
g32 = ArrayHydroGrid(4, 3, (0.0, 8.0), (0.0, 9.0); T = Float32)
check(g32.dx isa Float32 && eltype(alloc_field(g32)) === Float32, "T = Float32 gives Float32 spacings and fields")
check(throws(() -> ArrayHydroGrid(0, 3, (0, 1), (0, 1)), ArgumentError), "Nx = 0 is rejected with an ArgumentError")
check(alloc_field(g) == zeros(4, 3), "alloc_field(grid) is a zeroed Nx x Ny Array")
check(alloc_field(g, 1:12) == reshape(1.0:12.0, 4, 3) && eltype(alloc_field(g32, 1:12)) === Float32,
      "alloc_field(grid, data) reshapes and converts to the grid's float type")
bg = BareGrid(4, 3, 2.0, 3.0)
check(throws(() -> alloc_field(bg), ErrorException) && throws(() -> alloc_field(bg, 1:12), ErrorException),
      "a grid type without alloc_field methods gets the explanatory error stubs")
fld = rand(4, 3); fld0 = copy(fld)
check(fill_halo!(fld, g) === nothing && fld == fld0, "fill_halo! is a no-op on plain arrays")

sub("minus_gradient_x!/y!: centred inside, one-sided (full gradient) on the edges")
gg = ArrayHydroGrid(7, 5, (0.0, 7 * 300.0), (0.0, 5 * 400.0))
xc = [(i - 0.5) * gg.dx for i in 1:7, j in 1:5]; yc = [(j - 0.5) * gg.dy for i in 1:7, j in 1:5]
lin = 3.0 .* xc .- 2.0 .* yc
dgx = alloc_field(gg); dgy = alloc_field(gg)
minus_gradient_x!(gg, dgx, lin); minus_gradient_y!(gg, dgy, lin)
check(all(isapprox.(dgx, -3.0; atol = 1e-12)) && all(isapprox.(dgy, 2.0; atol = 1e-12)),
      "a linear field gets its exact -gradient in every cell, edges included (one-sided, not halved)")
quad = xc .^ 2
minus_gradient_x!(gg, dgx, quad)
check(all(isapprox.(dgx[2:end-1, :], -2 .* xc[2:end-1, :]; rtol = 1e-12)), "centred differences are exact for a quadratic in the interior")
check(isapprox(dgx[1, 1], -(quad[2, 1] - quad[1, 1]) / gg.dx; rtol = 1e-14), "edge cell i = 1 uses (f[2] - f[1])/dx")
d1 = ones(1, 5); FH.minus_gradient_x_kernel!(d1, rand(1, 5), 1, 5, 1.0)
check(all(iszero, d1), "a single-column domain has zero x-gradient")
FH.minus_gradient_y_kernel!(dgy, lin, 7, 5, gg.dy)
check(all(isapprox.(dgy, 2.0; atol = 1e-12)), "minus_gradient_y_kernel! called directly agrees")
covered!("AbstractHydroGrid", "ArrayHydroGrid", "alloc_field", "fill_halo!", "minus_gradient_x_kernel!", "minus_gradient_y_kernel!",
         "minus_gradient_x!", "minus_gradient_y!")

sub("masked reductions and overwrite_where!")
v = randn(6, 5); m = rand(0:1, 6, 5) .* 1.0; m[1, 1] = 1.0; m[2, 1] = 0.5   # 0.5 is not "grounded"
sel = m .== 1
check(isapprox(masked_mean(gg, v, m), mean(v[sel]); rtol = 1e-14), "masked_mean = mean over cells with mask == 1 (0.5 excluded)")
check(masked_max_abs(gg, v, m) == maximum(abs, v[sel]), "masked_max_abs = max |field| over mask == 1")
w2 = v .+ randn(6, 5)
check(masked_max_abs_diff(gg, v, w2, m) == maximum(abs.(v .- w2)[sel]), "masked_max_abs_diff = max |a - b| over mask == 1")
show1("masked_mean with no grounded cell", masked_mean(gg, v, zeros(6, 5)))
check(isnan(masked_mean(gg, v, zeros(6, 5))), "masked_mean of an empty mask is 0/0 = NaN (a pitfall, see the guide)")
dst = copy(v); overwrite_where!(gg, dst, m, ==(0.0), 7.0)
check(all(dst[m .== 0] .== 7.0) && dst[m .!= 0] == v[m .!= 0], "overwrite_where!(scalar): only cells where the predicate holds change")
dst = copy(v); src = rand(6, 5); overwrite_where!(gg, dst, m, ==(1.0), src; scale = 2.0)
check(dst[sel] == 2.0 .* src[sel] && dst[.!sel] == v[.!sel], "overwrite_where!(field; scale) writes scale*src where the predicate holds")
covered!("masked_mean", "masked_max_abs_diff", "masked_max_abs", "overwrite_where!")

sub("convolve!: generic ImageFiltering default versus ArrayHydroGrid's cached FFT")
srcf = randn(7, 5); kern = [1.0 2 1; 2 4 2; 1 2 1] ./ 16
a1 = zeros(7, 5); a2 = zeros(7, 5)
convolve!(BareGrid(7, 5, 300.0, 400.0), a1, srcf, kern)   # imfilter! path
convolve!(gg, a2, srcf, kern)                               # cached_fft_convolve! path
check(relerr(a2, a1) < 1e-12, "the cached FFT convolution of ArrayHydroGrid equals ImageFiltering.imfilter! (replicate border)")
covered!("convolve!")

# =============================================================================================
# PART 2 -- common/fft_convolution.jl: a plan-reusing FFT convolution
# =============================================================================================
# `update_smoothed_potential_gradients!` convolves two gradient fields with the Kamb & Echelmeyer
# coupling kernel on every solve. ImageFiltering's FFT path builds new FFTW plans and padded
# buffers on every call (~20 MB on a real grid, per the source), so this file keeps them in a
# cache (`FFTConvCache`, stored in the grid's `conv_cache` Ref) and rebuilds it only when the
# image size or kernel radius changes. Only point-symmetric kernels are supported (convolution
# equals correlation then). Padded sizes are rounded up to 7-smooth numbers (`fft_size`), because
# FFTW is slow for sizes with a large prime factor (Thwaites 2 km padded to 389 x 332, 389 prime).
part(2, "common/fft_convolution.jl", "CACHED FFT CONVOLUTION")
show1("fft_size(389), fft_size(332), fft_size(64)", (FH.fft_size(389), FH.fft_size(332), FH.fft_size(64)))
smooth7(n) = (for p in (2, 3, 5, 7); while n % p == 0; n ÷= p; end; end; n == 1)
check(all(n -> FH.fft_size(n) >= n && smooth7(FH.fft_size(n)), 1:500) && FH.fft_size(64) == 64,
      "fft_size(n) is the smallest 2,3,5,7-smooth integer >= n")
sp = rand(6, 4); pad = fill(NaN, 6 + 4, 4 + 2)
FH.fill_padded_src!(pad, sp, 2, 1)
check(pad[3:8, 2:5] == sp && pad[1, 2:5] == sp[1, :] && pad[10, 2:5] == sp[6, :] && pad[3:8, 1] == sp[:, 1] && pad[1, 1] == sp[1, 1] && pad[10, 6] == sp[6, 4],
      "fill_padded_src!: data in the middle, edge rows/columns and corners replicated outward")
k53 = rand(5, 3); pk = zeros(12, 9)
FH.fill_padded_kernel!(pk, k53, 2, 1, (12, 9))
check(pk[1, 1] == k53[3, 2] && pk[12, 1] == k53[2, 2] && pk[1, 9] == k53[3, 1] && pk[3, 2] == k53[5, 3] && count(!iszero, pk) == 15,
      "fill_padded_kernel!: kernel centre at [1,1], negative offsets wrapped to the far end")
cref = Ref{Any}(nothing)
c1 = FH.get_fft_conv_cache!(cref, 20, 15, 2, 1)
c2 = FH.get_fft_conv_cache!(cref, 20, 15, 2, 1)
c3 = FH.get_fft_conv_cache!(cref, 20, 15, 3, 1)
check(c1 isa FH.FFTConvCache && c1 === c2 && c3 !== c1 && cref[] === c3, "get_fft_conv_cache! reuses the cache, rebuilds it when the kernel radius changes")
check(c1.full_size == (FH.fft_size(24), FH.fft_size(17)) && all(isnan, FH.FFTConvCache(5, 5, 1, 1).kernel),
      "FFTConvCache: padded size from fft_size; stored kernel starts as NaN so the first call always transforms it")
# Independent reference: a brute-force correlation with replicate padding, rectangular kernel.
img = randn(20, 15)
kr = [exp(-((i - 3)^2 / 2 + (j - 2)^2)) for i in 1:5, j in 1:3]; kr ./= sum(kr)
function brute_conv(src, k)
    Nx, Ny = size(src); fx, fy = (size(k) .- 1) .÷ 2
    out = similar(src)
    for j in 1:Ny, i in 1:Nx
        s = 0.0
        for dj in -fy:fy, di in -fx:fx
            s += k[di + fx + 1, dj + fy + 1] * src[clamp(i + di, 1, Nx), clamp(j + dj, 1, Ny)]
        end
        out[i, j] = s
    end
    return out
end
dres = zeros(20, 15); cref2 = Ref{Any}(nothing)
FH.cached_fft_convolve!(cref2, dres, img, kr)
check(relerr(dres, brute_conv(img, kr)) < 1e-12, "cached_fft_convolve! equals a brute-force replicate-padded convolution (5x3 kernel)")
cache_before = cref2[]; fk_before = copy(cache_before.Fkern)
FH.cached_fft_convolve!(cref2, dres, img .* 2, kr)
check(cref2[] === cache_before && cache_before.Fkern == fk_before && relerr(dres, 2 .* brute_conv(img, kr)) < 1e-12,
      "a second call with the same kernel reuses plans and the kernel transform")
conv_alloc(cr, d, s, k) = @allocated FH.cached_fft_convolve!(cr, d, s, k)
conv_alloc(cref2, dres, img, kr)
show1("bytes allocated by a warm cached_fft_convolve!", conv_alloc(cref2, dres, img, kr))
check(conv_alloc(cref2, dres, img, kr) < 4096, "a warm call allocates (almost) nothing")
kr2 = copy(kr); kr2[3, 2] += 0.1; kr2 ./= sum(kr2)
FH.cached_fft_convolve!(cref2, dres, img, kr2)
check(relerr(dres, brute_conv(img, kr2)) < 1e-12, "a changed kernel (same size) is re-transformed")
covered!("FFTConvCache", "fft_size", "get_fft_conv_cache!", "fill_padded_src!", "fill_padded_kernel!", "cached_fft_convolve!")

# =============================================================================================
# PART 3 -- common/model.jl: AbstractHydroModel and the N_responds_to_ub trait
# =============================================================================================
# `AbstractHydroModel` is the supertype of the three models. `N_responds_to_ub(model)` tells a
# coupled ice-flow model whether N depends on the basal sliding speed it is iterating on, and so
# whether it should re-evaluate N inside its velocity solve (`N_from_ub!`, PART 16). The default
# is false (HAB: N from geometry only; Shakti: N is prognostic state); K24 overrides it.
part(3, "common/model.jl", "AbstractHydroModel AND N_responds_to_ub")
check(N_responds_to_ub(DummyModel()) == false, "default N_responds_to_ub is false")
check(N_responds_to_ub(HABHydroModel(GRID)) == false, "HAB: N does not depend on u_b")
covered!("AbstractHydroModel", "N_responds_to_ub")

# =============================================================================================
# PART 4 -- common/state.jl: HydroState, the fields every model shares
# =============================================================================================
# `HydroState` holds the inputs every model needs (mask, ice thickness h, bed b) and the two
# outputs every model writes (effective pressure N, water thickness W). Model-specific fields live
# on the model, not here, so one state type serves all models. The constructor copies the inputs
# through `alloc_field` (so the state never aliases the caller's arrays) and checks their sizes.
# mask is stored as a float field; "grounded" means mask == 1 exactly throughout the code.
part(4, "common/state.jl", "HydroState")
st = newstate()
check(st isa HydroState{Matrix{Float64}} && st isa FH.AbstractHydroState, "HydroState{Matrix{Float64}} <: AbstractHydroState")
check(st.h == INP.h && st.h !== INP.h && all(iszero, st.N) && all(iszero, st.W), "inputs copied (not aliased); N and W start at zero")
check(throws(() -> HydroState(GRID, INP.mask[1:end-1, :], INP.h, INP.b), ArgumentError), "a size mismatch is an ArgumentError naming the field")
st32 = HydroState(g32, ones(Int, 4, 3), rand(Float32, 4, 3), rand(Float32, 4, 3))
check(eltype(st32.mask) === Float32, "an integer mask is stored in the grid's float type")
covered!("AbstractHydroState", "HydroState")

# =============================================================================================
# PART 5 -- common/simulation.jl: steady-state and time-evolving simulations
# =============================================================================================
# Two simulation types. `SteadyStateSimulation(model, grid, state)` is for K24 and HAB: `run!`
# recomputes N and W from the current geometry, with no time stepping (the water is assumed to
# equilibrate much faster than the ice). `TimeSimulation(model)` wraps a model that owns its own
# time loop, today only `ShaktiHydroModel`; its `run!`/`step!` methods come from the Shakti
# extension (PART 24).
part(5, "common/simulation.jl", "SteadyStateSimulation AND TimeSimulation")
hab0 = HABHydroModel(GRID)
ss = SteadyStateSimulation(hab0, GRID, st)
check(ss isa FH.AbstractSimulation && ss.model === hab0 && ss.grid === GRID && ss.state === st, "SteadyStateSimulation stores model, grid and state")
ts = TimeSimulation(DummyModel())
check(ts isa TimeSimulation{DummyModel} && ts.model isa DummyModel, "TimeSimulation{M} wraps any AbstractHydroModel")
covered!("AbstractSimulation", "TimeSimulation", "SteadyStateSimulation")

# =============================================================================================
# PART 6 -- common/effective_pressure.jl: the overburden pressure
# =============================================================================================
# `update_Po!(model, grid, state)` writes Po = rho_i g h into `model.Po`; it is shared by every
# model that has `Po`, `rho_i` and `g` (HAB uses it; K24 computes Po inside its fused kernel,
# PART 16). Uses the HAB model of PART 19.
part(6, "common/effective_pressure.jl", "OVERBURDEN PRESSURE (uses HABHydroModel, PART 19)")
update_Po!(hab0, GRID, st)
check(hab0.Po == 917.0 .* 9.81 .* st.h, "Po = rho_i g h with HAB's rho_i = 917, g = 9.81")
covered!("update_Po!")

# =============================================================================================
# PART 7 -- common/run.jl: run! for steady states, and the step! stub
# =============================================================================================
# `run!(sim::SteadyStateSimulation)` is one call to `update_steady_state!(model, grid, state)`,
# which each model implements (PART 17, PART 21). `step!` is declared here with no methods: it
# exists so that the Shakti extension can add `step!(::TimeSimulation{<:ShaktiHydroModel})` to a
# name FastHydrology owns, for coupled drivers that run their own time loop.
part(7, "common/run.jl", "run! AND THE step! STUB")
st_a = newstate(); st_b = newstate()
run!(SteadyStateSimulation(HABHydroModel(GRID), GRID, st_a))
update_steady_state!(HABHydroModel(GRID), GRID, st_b)
check(st_a.N == st_b.N, "run!(SteadyStateSimulation) is exactly update_steady_state!")
mstep = methods(step!)
show1("methods of step! (all from extensions)", length(mstep))
check(all(mm -> parentmodule(mm) !== FastHydrology, mstep) && length(mstep) >= 1, "step! has no method in FastHydrology itself; the Shakti extension supplies one")
covered!("run!", "step!")

# =============================================================================================
# PART 8 -- common/checkpoint.jl: save and restore a state
# =============================================================================================
# `save_checkpoint(path, state, step, time)` writes every field of any AbstractHydroState
# (found with `fieldnames`, so new state types need no code here) to a NetCDF file, tagged with
# the step and time. It writes to `path.tmp` and renames, so a crash during the write leaves the
# previous checkpoint intact. `load_checkpoint!(state, path)` restores the fields in place and
# returns (step, time). K24 and HAB have no time-integrated state, so this exists for a coupled
# driver resuming its own loop. `_output_array` / `_copy_into!` are the two hooks a grid with
# non-Array fields would extend.
part(8, "common/checkpoint.jl", "CHECKPOINTS")
ckp = joinpath(OUT, "tour_checkpoint.nc")
stc = newstate(); stc.N .= rand(NX, NY); stc.W .= rand(NX, NY)
saved = deepcopy(stc)
save_checkpoint(ckp, stc, 7, 3.5e7)
check(isfile(ckp) && !isfile(ckp * ".tmp"), "the checkpoint exists and the temporary file was renamed away")
save_checkpoint(ckp, stc, 8, 3.6e7)     # overwriting an existing checkpoint
stc.N .= 0; stc.h .= 0
stepk, timek = load_checkpoint!(stc, ckp)
check((stepk, timek) == (8, 3.6e7) && stc.N == saved.N && stc.h == saved.h && stc.mask == saved.mask, "load_checkpoint! restores every field and returns (step, time)")
check(throws(() -> save_checkpoint(joinpath(OUT, "empty.nc"), EmptyState(), 1, 0.0), ErrorException), "a state type with no fields is an error")
oa = FH._output_array(stc.N)
check(oa == stc.N && oa !== stc.N, "_output_array returns a plain Array copy")
tgt = zeros(NX, NY); FH._copy_into!(tgt, saved.W)
check(tgt == saved.W, "_copy_into! copies data into an existing field in place")
covered!("_output_array", "_copy_into!", "save_checkpoint", "load_checkpoint!")

# =============================================================================================
# PART 9 -- common/output.jl: periodic NetCDF output that survives a killed run
# =============================================================================================
# `NetCDFOutputWriter(path, grid, field_names)` creates a file with an unlimited time dimension;
# `write_output!(writer, step, time, (N = ..., W = ...))` appends one slice and syncs to disk at
# once, so a run killed by the wall-time limit keeps every slice written before. The motivating
# incident (in the source) is a ~4000-step coupled Yelmo run that wrote only at the end. With
# `resume_step = k` the writer reopens the file and first drops slices with step > k
# (`_trim_output_after!`, a copy-and-rename, since an unlimited dimension cannot shrink in place).
part(9, "common/output.jl", "PERIODIC NETCDF OUTPUT")
outp = joinpath(OUT, "tour_series.nc")
wr = NetCDFOutputWriter(outp, GRID, [:N, :W])
check(wr isa FH.AbstractOutputWriter && wr.field_names == [:N, :W], "NetCDFOutputWriter <: AbstractOutputWriter")
for k in 1:5
    write_output!(wr, k, 10.0 * k, (N = fill(Float64(k), NX, NY), W = fill(-Float64(k), NX, NY), extra = zeros(2)))
end
NCDataset(outp, "r") do ds
    check(ds["step"][:] == 1:5 && ds["time"][:] == 10.0 .* (1:5) && all(ds["N"][:, :, 3] .== 3.0) && size(ds["W"]) == (NX, NY, 5),
          "the file holds 5 slices, readable while the writer is still open (synced after each write)")
end
check(throws(() -> write_output!(wr, 6, 60.0, (N = zeros(NX, NY),)), ErrorException), "a missing field in the NamedTuple is an error (extra fields are ignored)")
# The field check runs after step and time are appended, so the failed call leaves a partial slice:
NCDataset(outp, "r") do ds
    n6 = length(ds["step"])
    show1("slices after the failed write (step, W filled?)", (n6, ds["step"][end], ismissing(ds["W"][1, 1, end])))
    check(n6 == 6 && ds["step"][end] == 6, "a failed write_output! still appends step/time: a partial slice 6 (see the guide's open issues)")
end
close_output!(wr)
check(throws(() -> NetCDFOutputWriter(outp, GRID, [:N, :W]; overwrite = false), ErrorException), "overwrite = false refuses to replace an existing file")
wr2 = NetCDFOutputWriter(outp, GRID, [:N, :W]; resume_step = 3)
write_output!(wr2, 4, 40.5, (N = fill(44.0, NX, NY), W = zeros(NX, NY)))
close_output!(wr2)
NCDataset(outp, "r") do ds
    check(ds["step"][:] == [1, 2, 3, 4] && ds["time"][4] == 40.5 && all(ds["N"][:, :, 4] .== 44.0) && all(ds["N"][:, :, 2] .== 2.0),
          "resume_step = 3 dropped steps 4-5, kept 1-3, and appended the new step 4")
end
FH._trim_output_after!(outp, [:N, :W], 2)
NCDataset(outp, "r") do ds
    check(ds["step"][:] == [1, 2] && size(ds["N"], 3) == 2, "_trim_output_after! keeps only steps <= resume_step")
end
check(throws(() -> NetCDFOutputWriter(joinpath(OUT, "nope.nc"), GRID, [:N]; resume_step = 1), ErrorException), "resuming a file that does not exist is an error")
cp(outp, joinpath(OUT, "tour_series_copy.nc"); force = true)
check(throws(() -> NetCDFOutputWriter(joinpath(OUT, "tour_series_copy.nc"), GRID, [:N, :Z]; resume_step = 9), ErrorException),
      "resuming with a field the file does not have is an error")
covered!("AbstractOutputWriter", "NetCDFOutputWriter", "_trim_output_after!", "write_output!", "close_output!")

# =============================================================================================
# PART 10 -- common/utilities.jl: unit conversions and grid limits
# =============================================================================================
# The model works in SI throughout (m, s, Pa, kg). Ice-sheet data come in per-year and km units,
# hence `perYear2perSecond`, `perSecond2perYear` (a Julian year, 365.25 d) and `Km2m`.
# `compute_lims(xc, yc)` turns cell-centre coordinates into the edge-to-edge limits that
# `ArrayHydroGrid` takes (half a cell of padding on each side).
part(10, "common/utilities.jl", "UNITS AND GRID LIMITS")
check(FH.SECONDS_PER_YEAR == 365.25 * 86400 && FH.METERS_PER_KILOMETER == 1000, "SECONDS_PER_YEAR is the Julian year; METERS_PER_KILOMETER = 1000")
check(perYear2perSecond(31557600.0) == 1.0 && perSecond2perYear(perYear2perSecond(123.0)) ≈ 123.0 && Km2m(2.5) == 2500.0, "per year <-> per second, km -> m")
check(perYear2perSecond(1.0f0) isa Float32 && Km2m(2.0f0) isa Float32, "conversions keep the argument's float type")
xl, yl = compute_lims(collect(0.0:2.0:10.0), collect(5.0:5.0:20.0))
check(xl == (-1.0, 11.0) && yl == (2.5, 22.5), "compute_lims pads the centre range by half a cell on each side")
gl = ArrayHydroGrid(6, 4, xl, yl)
check(gl.dx == 2.0 && gl.dy == 5.0, "...so ArrayHydroGrid(Nx, Ny, compute_lims(xc, yc)...) recovers the data spacing")
covered!("SECONDS_PER_YEAR", "METERS_PER_KILOMETER", "compute_lims", "perYear2perSecond", "perSecond2perYear", "Km2m")

# =============================================================================================
# PART 11 -- common/plotting.jl: plotting stubs and mask_field
# =============================================================================================
# `visualize_field` and `visualize_grid` are declared without methods; the CairoMakie extension
# (PART 23) adds them, so loading FastHydrology never compiles Makie. `mask_field(field, mask,
# value)` needs no Makie: it returns a copy with every cell where mask != 1 set to `value`, e.g.
# NaN to blank the ocean in a plot.
part(11, "common/plotting.jl", "PLOTTING STUBS AND mask_field")
mf = mask_field(INP.h, INP.mask, NaN)
check(all(isnan, mf[INP.mask .!= 1]) && mf[INP.mask .== 1] == INP.h[INP.mask .== 1] && !any(isnan, INP.h), "mask_field blanks non-grounded cells in a copy")
check(all(mm -> parentmodule(mm) !== FastHydrology, methods(visualize_field)) && !isempty(methods(visualize_grid)), "visualize_field/visualize_grid methods come from the Makie extension")
covered!("visualize_field", "visualize_grid", "mask_field")

# =============================================================================================
# PART 12 -- models/kazmierczak2024/model.jl: the K24 model, its options and its sliding laws
# =============================================================================================
# The Kazmierczak et al. (2024) model routes the basal water source over the geometric potential
# phi0 = rho_i g h + rho_w g b (Le Brocq et al. 2006/2009 flux routing), turns the routed flux into
# a conduit discharge, and gets N from a far-field conduit balance N_inf and an erf transition to
# the grounding line. This file defines every choice as a type, so each choice is resolved by
# dispatch once, when the model is built:
#   AbstractDissipationMelt (On/Off), AbstractPsiOutAlgorithm (Taped/Recursive/Iterative/Topological),
#   AbstractRoutingScheme (Warner, GDSWarner, Quinn, Tarboton, ModifiedTarboton, GDSTarboton),
#   AbstractFillAlgorithm, AbstractQConversion, AbstractDissipationDiscretization,
#   AbstractFrictionDiscretization, AbstractWaterThicknessAlgorithm (+ AbstractGradientConvention),
#   AbstractSlidingLaw (+ AbstractPressureDependentSlidingLaw), AbstractDrainageMode.
# The model is split into `KazmierczakParams` (immutable constants and options) and
# `KazmierczakWorkspace` (all arrays), with `getproperty` forwarding so `model.q`, `model.rho_w`
# work. Defaults: Warner routing with priority-flood filling, face-averaged q and face dissipation,
# taped routing, Darcy-Weisbach W with a domain-mean gradient, no friction, no clamps (Wmin = 0,
# Wmax = Inf, sigmat = 0, q_max = Inf). `coupling_length_kamb86` has no safe default and warns.
part(12, "models/kazmierczak2024/model.jl", "KazmierczakHydroModel, OPTIONS AND SLIDING LAWS")
mdef, logs = with_logs(() -> KazmierczakHydroModel(GRID, INP.kappa, INP.abs_v_b, INP.A_visc, INP.G, INP.q_T))
check(warned(logs, "coupling_length_kamb86 not specified"), "leaving coupling_length_kamb86 unset warns and explains how to choose it")
check(mdef.longcoupwater == 5.0, "...and falls back to 10 ice thicknesses, i.e. longcoupwater = 10/2 = 5")
show1("default routing / fill / q / dissipation", (nameof(typeof(mdef.routing_scheme)), nameof(typeof(mdef.fill_algorithm)), nameof(typeof(mdef.q_conversion)), nameof(typeof(mdef.dissipation_discretization))))
check(mdef.routing_scheme isa Warner && mdef.fill_algorithm isa PriorityFloodFill && mdef.q_conversion isa QFromFaceAverage &&
      mdef.dissipation_discretization isa FaceDissipation && mdef.psi_out_algorithm isa TapedPsiOut,
      "defaults: Warner + PriorityFloodFill + QFromFaceAverage + FaceDissipation + TapedPsiOut")
check(mdef.water_thickness_algorithm isa DarcyWeisbachThickness{MeanGradient} && mdef.sliding_law isa NoFrictionSlidingLaw &&
      mdef.drainage_mode isa BothDrainage && mdef.dissipation_melt isa FH.DissipationMeltOn && mdef.friction_discretization isa CellCentredFriction,
      "defaults: DarcyWeisbachThickness(MeanGradient), NoFrictionSlidingLaw, BothDrainage, dissipation on, cell-centred friction")
check((mdef.Wmin, mdef.Wmax, mdef.sigmat, mdef.q_min, mdef.q_max) == (0.0, Inf, 0.0, 0.0, Inf), "no clamp by default: Wmin 0, Wmax Inf, sigmat 0, q in [0, Inf]")
check((mdef.rho_w, mdef.rho_i, mdef.g, mdef.L_w, mdef.n, mdef.h_b, mdef.alpha, mdef.beta, mdef.f, mdef.F_till, mdef.Q_c, mdef.H_0, mdef.l_c) ==
      (1000.0, 917.0, 9.81, 3.34e5, 3.0, 0.1, 1.25, 1.5, 0.1, 1.1, 1.0, 0.1, 1e4) && FH.KAZMIERCZAK_DEFAULT_L_W == 3.34e5,
      "physical defaults (rho_w, rho_i, g, L_w, n, h_b, alpha, beta, f, F_till, Q_c, H_0, l_c)")
check(mdef.K ≈ (2 / pi)^0.25 * sqrt((pi + 2) / (1000.0 * 0.1)) && mdef.eta_w ≈ perYear2perSecond(1.8e-3),
      "K = (2/pi)^(1/4) sqrt((pi+2)/(rho_w f)); eta_w = 1.8e-3/yr (KORI-ULB's units, not SI water viscosity)")
check((mdef.fill_iters, mdef.max_psi_out_calls, mdef.max_dissipation_iters, mdef.dissipation_rtol, mdef.max_qN_iters, mdef.qN_rtol) ==
      (10, 100_000, 20, 1e-12, 20, 1e-8), "iteration controls: fill_iters 10, max_psi_out_calls 1e5, 20 sweeps, rtol 1e-12 / 1e-8")
m0 = newk24()
check(m0.mdot_fixed ≈ (INP.G .- INP.q_T) ./ 3.34e5 && m0.i_eb == INP.i_eb, "mdot_fixed = (G - q_T)/L_w; i_eb copied from the keyword")
check(m0.rho_w === m0.params.rho_w && m0.q === m0.workspace.q && m0.params isa FH.KazmierczakParams && m0.workspace isa FH.KazmierczakWorkspace,
      "getproperty forwards model.<name> to params, then workspace")
pn = propertynames(m0)
check(:rho_w in pn && :q in pn && :params in pn && :workspace in pn && length(pn) == length(fieldnames(FH.KazmierczakParams)) + length(fieldnames(FH.KazmierczakWorkspace)) + 2,
      "propertynames lists every params and workspace field plus :params, :workspace")
show1("fields of KazmierczakParams / KazmierczakWorkspace", (length(fieldnames(FH.KazmierczakParams)), length(fieldnames(FH.KazmierczakWorkspace))))

sub("options follow the routing scheme; invalid combinations are rejected")
mg = newk24(; routing_scheme = GDSWarner())
check(mg.fill_algorithm isa JacobiFill && mg.q_conversion isa QFromOutflow && mg.dissipation_discretization isa CellCentredDissipation,
      "GDSWarner() alone gives the original K24 set-up: JacobiFill, QFromOutflow, CellCentredDissipation")
mq = newk24(; routing_scheme = Quinn())
check(mq.fill_algorithm isa PriorityFloodFill && mq.q_conversion isa QFromOutflow && mq.dissipation_discretization isa CellCentredDissipation,
      "an 8-neighbour scheme gets PriorityFloodFill, QFromOutflow, CellCentredDissipation")
check(newk24(; routing_scheme = GDSTarboton()).fill_algorithm isa JacobiFill, "GDSTarboton also defaults to JacobiFill")
check(throws(() -> newk24(; routing_scheme = Quinn(), psi_out_algorithm = RecursivePsiOut()), ArgumentError), "non-GDSWarner routing needs TapedPsiOut")
check(throws(() -> newk24(; routing_scheme = Quinn(), q_conversion = QFromFaceAverage()), ArgumentError), "face-based options need a 4-neighbour scheme")
check(throws(() -> newk24(; coupling_length_kamb86 = -1.0), ArgumentError), "a negative coupling length is an ArgumentError")
check(throws(() -> newk24(; friction_discretization = StaggeredFriction(zeros(3, 3), zeros(3, 3))), ArgumentError), "StaggeredFriction velocities must be Nx x Ny")
check(throws(() -> KazmierczakHydroModel(GRID, INP.kappa[1:2, :], INP.abs_v_b, INP.A_visc, INP.G, INP.q_T; coupling_length_kamb86 = 0.0), ArgumentError),
      "input fields of the wrong size are rejected")
check(throws(() -> newk24(; i_eb = zeros(2, 2)), ArgumentError), "an i_eb of the wrong size is rejected")
check(newk24(; dissipation_melt = false).dissipation_melt isa FH.DissipationMeltOff, "dissipation_melt = false becomes the DissipationMeltOff trait")
check(newk24(; coupling_length_kamb86 = 0.0).longcoupwater == 0.0, "coupling_length_kamb86 = 0 disables the smoothing (longcoupwater = 0)")

sub("option types and their constructors")
check(TopologicalPsiOut().allow_cycles == false && TopologicalPsiOut(allow_cycles = true).allow_cycles, "TopologicalPsiOut(; allow_cycles = false)")
check(Quinn().original == false && Quinn(original = true).original, "Quinn(; original = false)")
check(PriorityFloodFill().epsilon == 1.0, "PriorityFloodFill(; epsilon = 1 Pa)")
check(DarcyWeisbachThickness().gradient_convention isa MeanGradient && LaminarThickness(gradient_convention = LocalGradient()).gradient_convention isa LocalGradient,
      "DarcyWeisbachThickness/LaminarThickness(; gradient_convention = MeanGradient())")
sfd = StaggeredFriction(ones(NX, NY), zeros(NX, NY))
check(sfd.quadrature == false && sfd.u_floor ≈ perYear2perSecond(1e-3) && sfd.ux isa Matrix{Float64}, "StaggeredFriction(ux, uy; quadrature = false, u_floor = 1e-3 m/yr)")
check(FH.n_directions(Warner()) == 4 && FH.n_directions(GDSWarner()) == 4 && FH.n_directions(Quinn()) == 8 && FH.n_directions(Tarboton()) == 8 &&
      FH.n_directions(ModifiedTarboton()) == 8 && FH.n_directions(GDSTarboton()) == 8, "n_directions: 4 for (GDS)Warner, 8 for the others")
tp = FH.RoutingTape{Float64}(NX, NY)
check(!tp.valid && size(tp.w8) == (8, NX, NY) && size(tp.Fx) == (NX + 1, NY) && size(tp.Fy) == (NX, NY + 1) && isempty(tp.w),
      "RoutingTape: empty, invalid, w8 (8,Nx,Ny), face fluxes on staggered (Nx+1,Ny)/(Nx,Ny+1) arrays")
for T in (FH.AbstractQConversion, FH.AbstractDissipationDiscretization, FH.AbstractFrictionDiscretization, AbstractFillAlgorithm, FH.AbstractDrainageMode)
    check(isabstracttype(T), "$(nameof(T)) is an abstract option type")
end
check(LowestNeighbourFill() isa AbstractFillAlgorithm && ArealConduitThickness() isa AbstractWaterThicknessAlgorithm && LocalGradient() isa AbstractGradientConvention,
      "LowestNeighbourFill, ArealConduitThickness, LocalGradient are singleton option values")
check(EfficientOnly() isa FH.AbstractDrainageMode && InefficientOnly() isa FH.AbstractDrainageMode && FH.DissipationMeltOn() isa FH.AbstractDissipationMelt,
      "drainage modes and dissipation traits")
check(IterativePsiOut() isa AbstractPsiOutAlgorithm && RecursivePsiOut() isa AbstractPsiOutAlgorithm && ModifiedTarboton() isa AbstractRoutingScheme,
      "psi_out algorithms and routing schemes")

sub("sliding laws: constructors, N dependence, float conversion")
check(PowerPlasticSlidingLaw(c_till = 0.5) isa FH.AbstractPressureDependentSlidingLaw && !(WeertmanSlidingLaw(C = 1.0) isa FH.AbstractPressureDependentSlidingLaw) &&
      !(PrescribedFieldSlidingLaw(GRID, ones(NX, NY)) isa FH.AbstractPressureDependentSlidingLaw), "N-dependent laws are AbstractPressureDependentSlidingLaw; Weertman and prescribed are not")
pp = PowerPlasticSlidingLaw(c_till = 0.5); rc = RegularizedCoulombSlidingLaw(c_till = 0.5)
check(pp.q == 1.0 && pp.u0 ≈ perYear2perSecond(100.0) && rc.q == 1 / 3 && WeertmanSlidingLaw(C = 2).q == 1 / 3 && WeertmanSlidingLaw(C = 2).C isa Float64,
      "keyword defaults: PowerPlastic q = 1, u0 = 100 m/yr; RegularizedCoulomb and Weertman q = 1/3")
rcf = RegularizedCoulombFieldSlidingLaw(GRID, fill(0.4, NX, NY))
srl = ShaktiRegularizedCoulombSlidingLaw(GRID, INP.A_visc, 0.25)
check(rcf.c_till == fill(0.4, NX, NY) && srl.lambda ≈ 1.5 .* INP.A_visc && srl.inv_n ≈ 1 / 3 && srl.n == 3.0,
      "field laws wrap their fields with alloc_field; Shakti law: lambda = 1.5 A, inv_n = 1/n")
m32 = newk24(k24_inputs(; T = Float32); sliding_law = WeertmanSlidingLaw(C = 1e5))
check(m32.sliding_law.C isa Float32 && eltype(m32.q) === Float32 && m32.rho_w isa Float32, "a Float32 grid gives a Float32 model; convert_sliding_law converts the law's scalars")
pf = PrescribedFieldSlidingLaw(GRID, ones(NX, NY))
check(FH.convert_sliding_law(Float64, pf) === pf && FH.convert_sliding_law(Float32, pf).tau_b isa Matrix{Float32} &&
      FH.convert_sliding_law(Float32, rcf).c_till isa Matrix{Float32} && FH.convert_sliding_law(Float32, srl).lambda isa Matrix{Float32} &&
      FH.convert_sliding_law(Float32, srl).C isa Float32 && FH.convert_sliding_law(Float32, pp).c_till isa Float32 &&
      FH.convert_sliding_law(Float32, rc).u0 isa Float32 && FH.convert_sliding_law(Float32, NoFrictionSlidingLaw()) isa NoFrictionSlidingLaw &&
      FH.convert_sliding_law(Float64, rcf) === rcf && FH.convert_sliding_law(Float64, srl) === srl,
      "convert_sliding_law: field laws returned as-is when the eltype matches, every law converted to Float32 otherwise")

sub("set_basal_terms!: changing the source between solves")
mb = newk24()
r = set_basal_terms!(mb; G = fill(0.1, NX, NY))
check(r === mb && mb.G == fill(0.1, NX, NY) && mb.q_T == INP.q_T && mb.mdot_fixed ≈ (0.1 .- INP.q_T) ./ 3.34e5, "G replaced, q_T kept, mdot_fixed recomputed")
set_basal_terms!(mb; q_T = fill(0.1, NX, NY), i_eb = zeros(NX, NY))
check(all(iszero, mb.mdot_fixed) && all(iszero, mb.i_eb), "q_T = G gives no fixed melt; i_eb replaced")
covered!("AbstractDissipationMelt", "DissipationMeltOn", "DissipationMeltOff", "AbstractPsiOutAlgorithm", "RecursivePsiOut", "IterativePsiOut",
         "TopologicalPsiOut", "TapedPsiOut", "AbstractFrictionDiscretization", "CellCentredFriction", "StaggeredFriction", "AbstractRoutingScheme",
         "GDSWarner", "Warner", "Quinn", "Tarboton", "ModifiedTarboton", "GDSTarboton", "n_directions", "AbstractQConversion", "QFromOutflow",
         "QFromFaceAverage", "AbstractDissipationDiscretization", "CellCentredDissipation", "FaceDissipation", "AbstractFillAlgorithm", "JacobiFill",
         "LowestNeighbourFill", "PriorityFloodFill", "RoutingTape", "AbstractWaterThicknessAlgorithm", "AbstractGradientConvention", "LocalGradient",
         "MeanGradient", "ArealConduitThickness", "DarcyWeisbachThickness", "LaminarThickness", "AbstractSlidingLaw", "AbstractPressureDependentSlidingLaw",
         "NoFrictionSlidingLaw", "PrescribedFieldSlidingLaw", "WeertmanSlidingLaw", "PowerPlasticSlidingLaw", "RegularizedCoulombSlidingLaw",
         "RegularizedCoulombFieldSlidingLaw", "ShaktiRegularizedCoulombSlidingLaw", "convert_sliding_law", "AbstractDrainageMode", "BothDrainage",
         "EfficientOnly", "InefficientOnly", "KazmierczakParams", "KazmierczakWorkspace", "KazmierczakHydroModel", "getproperty", "propertynames",
         "KAZMIERCZAK_DEFAULT_L_W", "set_basal_terms!")

# =============================================================================================
# PART 13 -- models/kazmierczak2024/sliding_law.jl: basal shear stress and frictional heat
# =============================================================================================
# The frictional heat Q_b = tau_b u_b melts ice at the bed (Eq. 3 of K24). `calc_tau_b(law, N, u)`
# is the scalar formula of each law, for tests and diagnostics; `update_tau_b!(model, state, law)`
# is the field version used in the solve, written out with plain broadcasts. For velocities on a
# C-grid (Yelmo's acx/acy), `StaggeredFriction` forms the heat from face drag beta_face u_face^2
# (faces) or at 2x2 Gauss points (Yelmo's qb_method = 2).
part(13, "models/kazmierczak2024/sliding_law.jl", "SLIDING LAWS AND FRICTIONAL HEAT")
Ns, us = 2.0e6, perYear2perSecond(150.0)
wl = WeertmanSlidingLaw(C = 3e4, q = 1 / 3)
check(calc_tau_b(NoFrictionSlidingLaw(), Ns, us) == 0.0, "NoFriction: tau_b = 0")
check(calc_tau_b(wl, Ns, us) ≈ 3e4 * us^(1 / 3), "Weertman: C |u|^q, independent of N")
check(calc_tau_b(PowerPlasticSlidingLaw(c_till = 0.5, q = 1.0, u0 = us), Ns, us) ≈ 0.5 * Ns, "PowerPlastic at |u| = u0, q = 1: tau_b = c_till N")
check(isapprox(calc_tau_b(rc, Ns, 1e3), 0.5 * Ns; rtol = 1e-6) && calc_tau_b(rc, Ns, us) < 0.5 * Ns, "RegularizedCoulomb saturates at the Coulomb limit c_till N")
mt = newk24(); stt = newstate(); stt.N .= 1e6 .+ 1e4 .* INP.x ./ 1e3
for law in (NoFrictionSlidingLaw(), wl, pp, rc)
    update_tau_b!(mt, stt, law)
    check(mt.tau_b ≈ calc_tau_b.(Ref(law), stt.N, mt.abs_v_b), "update_tau_b!($(nameof(typeof(law)))) equals calc_tau_b cell by cell")
end
update_tau_b!(mt, stt, PrescribedFieldSlidingLaw(GRID, fill(5e4, NX, NY)))
check(all(==(5e4), mt.tau_b), "PrescribedFieldSlidingLaw copies its field")
update_tau_b!(mt, stt, RegularizedCoulombFieldSlidingLaw(GRID, fill(0.5, NX, NY)))
t_field = copy(mt.tau_b); update_tau_b!(mt, stt, rc)
check(t_field == mt.tau_b, "RegularizedCoulombFieldSlidingLaw with a uniform field equals the scalar law exactly")
update_tau_b!(mt, stt, srl)
check(mt.tau_b ≈ 0.25 .* stt.N .* (mt.abs_v_b ./ (mt.abs_v_b .+ abs.(stt.N) .^ 3 .* 1.5 .* INP.A_visc)) .^ (1 / 3),
      "ShaktiRegularizedCoulomb: C N (|u| / (|u| + |N|^n lambda))^(1/n), lambda = 1.5 A")

sub("StaggeredFriction: heat from C-grid velocities")
u0s = perYear2perSecond(80.0)
mst = newk24(; friction_discretization = StaggeredFriction(fill(u0s, NX, NY), zeros(NX, NY)))
mst.abs_v_b .= u0s; mst.tau_b .= 6e4; mst.mdot_total .= 0
FH.add_friction_term!(mst, mst.friction_discretization)
check(all(isapprox.(mst.Q_b, 6e4 * u0s; rtol = 1e-12)), "uniform flow: face heat beta u^2 averaged to the centre = tau_b u_b everywhere")
check(mst.mdot_total ≈ mst.Q_b ./ mst.L_w, "the heat is added to mdot_total as Q_b / L_w")
mstq = newk24(; friction_discretization = StaggeredFriction(fill(u0s, NX, NY), fill(0.5u0s, NX, NY); quadrature = true))
mstq.abs_v_b .= hypot(u0s, 0.5u0s); mstq.tau_b .= 6e4; mstq.mdot_total .= 0
FH.add_friction_term!(mstq, mstq.friction_discretization)
check(all(isapprox.(mstq.Q_b, 6e4 * hypot(u0s, 0.5u0s); rtol = 1e-12)), "Gauss-point (Yelmo qb_method 2) form also gives |tau||u| for a uniform oblique flow")
check(FH.stagger_beta(fill(1.0, 1, 1), zeros(1, 1), 1e-3, 1, 1) == 1e3, "stagger_beta floors |u| at u_floor")
check(FH.GQ_S3 == 1 / sqrt(3) && length(FH.GQ_PTS) == 4 && FH.gq_interp((1.0, 2.0, 3.0, 4.0), (0.0, 0.0)) == 2.5 &&
      FH.gq_interp((1.0, 2.0, 3.0, 4.0), (-1.0, -1.0)) == 1.0, "2x2 Gauss points at +-1/sqrt(3); bilinear corner interpolation")
F = reshape(1.0:16.0, 4, 4)
check(FH.acx_corners(F, 2, 2, 1, 1, 3) == ((F[1, 1] + F[1, 2]) / 2, (F[2, 1] + F[2, 2]) / 2, (F[2, 2] + F[2, 3]) / 2, (F[1, 2] + F[1, 3]) / 2) &&
      FH.acy_corners(F, 2, 2, 1, 3, 1) == ((F[1, 1] + F[2, 1]) / 2, (F[2, 1] + F[3, 1]) / 2, (F[2, 2] + F[3, 2]) / 2, (F[1, 2] + F[2, 2]) / 2),
      "acx_corners/acy_corners average face values to the SW, SE, NE, NW corners")
Qtmp = zeros(NX, NY)
FH.staggered_friction_kernel!(Qtmp, fill(6e4, NX, NY), fill(u0s, NX, NY), fill(u0s, NX, NY), zeros(NX, NY), NX, NY, 1e-12, false)
check(all(isapprox.(Qtmp, 6e4 * u0s; rtol = 1e-12)), "staggered_friction_kernel! called directly agrees")
covered!("calc_tau_b", "update_tau_b!", "add_friction_term!", "stagger_beta", "GQ_S3", "GQ_PTS", "gq_interp", "acx_corners", "acy_corners", "staggered_friction_kernel!")

# =============================================================================================
# PART 14 -- models/kazmierczak2024/water_flux.jl: from the potential to the distributed flux q
# =============================================================================================
# `update_q!` is the heart of K24:
#   1. phi0 = rho_i g h + rho_w g b                                  (update_phi0!)
#   2. routing potential and flow directions for the scheme          (prepare_routing!, PART 15):
#      fill the pits of phi0 (potential_filling!), take gradients (update_potential_gradients!),
#      smooth them with the Kamb & Echelmeyer (1986) cone (update_smoothed_potential_gradients!)
#   3. corfac = (|sx| dy + |sy| dx)/|s|, the flux cross-section that turns psi_out [m3/s] into q [m2/s]
#   4. resolve_q!: route the water source to every cell (psi_out), q = psi_out/corfac, and iterate
#      when the source depends on q (dissipation melt) or on N (N-dependent sliding law).
# Routing (psi_out at a cell = its own source plus the shares its upstream neighbours send it) is
# a graph traversal. Four algorithms compute it; the default `TapedPsiOut` records the traversal
# once per update_q! and replays it each Picard sweep. This PART runs the original GDSWarner
# pipeline, which uses every function of this file.
part(14, "models/kazmierczak2024/water_flux.jl", "POTENTIAL, FILLING, SMOOTHING, ROUTING, q AND W")
mw = newk24(; routing_scheme = GDSWarner()); sw = newstate()
FH.update_phi0!(mw, GRID, sw)
check(mw.phi0 ≈ 917.0 .* 9.81 .* INP.h .+ 1000.0 .* 9.81 .* INP.b, "update_phi0!: phi0 = rho_i g h + rho_w g b")

sub("filling the pits of the routing potential")
function is_strict_min(p, i, j)
    Nx, Ny = size(p)
    v = p[i, j]
    v < p[max(i - 1, 1), j] && v < p[min(i + 1, Nx), j] && v < p[i, max(j - 1, 1)] && v < p[i, min(j + 1, Ny)]
end
nmin(p) = count(ij -> is_strict_min(p, ij[1], ij[2]), Iterators.product(1:size(p, 1), 1:size(p, 2)))
# The shared fixture has no pits (see its comment); 80 m bumps put real ones into the potential.
inp_p = k24_inputs(; bumps = 80.0); sp = newstate(inp_p)
mwp = newk24(inp_p; routing_scheme = GDSWarner()); FH.update_phi0!(mwp, GRID, sp)
npit = nmin(mwp.phi0)
show1("strict minima of phi0 before filling (80 m bumps)", npit)
check(npit > 0, "the 80 m bump fixture has pits, so the filling checks below do real work")
function naive_jacobi(p0, iters, lowest)
    p = copy(p0); Nx, Ny = size(p)
    for pass in 1:iters
        t = copy(p); any = false
        for j in 1:Ny, i in 1:Nx
            im1, ip1, jm1, jp1 = max(i - 1, 1), min(i + 1, Nx), max(j - 1, 1), min(j + 1, Ny)
            v, p1, p2, p3, p4 = p[i, j], p[ip1, j], p[im1, j], p[i, jp1], p[i, jm1]
            if v < p1 && v < p2 && v < p3 && v < p4
                t[i, j] = lowest ? min(p1, p2, p3, p4) : (p1 + p2 + p3 + p4) / 4.0; any = true
            end
        end
        p = t
        any || break
    end
    return p
end
FH.potential_filling!(mwp, GRID, sp)
check(mwp.phi0_filled == naive_jacobi(mwp.phi0, mwp.fill_iters, false), "JacobiFill (candidate-list version) equals naive full-grid Jacobi passes bit for bit")
check(any(mwp.phi0_filled .> mwp.phi0) && all(mwp.phi0_filled .>= mwp.phi0) && mwp.phi0 ≈ 917.0 .* 9.81 .* inp_p.h .+ 1000.0 .* 9.81 .* inp_p.b,
      "filling raises the pits and nothing else goes down; the true phi0 is left untouched")
check(mwp.h ≈ (mwp.phi0_filled .- 1000 * 9.81 .* sp.b) ./ (917 * 9.81), "model.h holds the thickness consistent with the filled potential")
show1("strict minima after 10 Jacobi passes", nmin(mwp.phi0_filled))
FH.potential_filling!(mw, GRID, sw)     # the shared GDSWarner model, used below, on the pit-free fixture
mlow = newk24(inp_p; routing_scheme = GDSWarner(), fill_algorithm = LowestNeighbourFill()); FH.update_phi0!(mlow, GRID, sp)
FH.potential_filling!(mlow, GRID, sp)
show1("strict minima after 10 lowest-neighbour passes", nmin(mlow.phi0_filled))
check(mlow.phi0_filled == naive_jacobi(mlow.phi0, 10, true), "LowestNeighbourFill equals naive lowest-neighbour passes")
p3 = [5.0 5 5; 5 1 5; 5 5 5]; t3 = copy(p3); fl = Tuple{Int32, Int32}[]
FH.fill_if_strict_minimum!(p3, t3, fl, 2, 2, 3, 3)
FH.fill_if_strict_minimum!(p3, t3, fl, 1, 1, 3, 3)
check(t3[2, 2] == 5.0 && fl == [(Int32(2), Int32(2))], "fill_if_strict_minimum!: the centre pit is raised to its neighbour mean; a non-minimum is left")
cands = Int32[]; stamp = zeros(Int32, 4, 4)
FH.collect_fill_candidates!(cands, stamp, 1, [(Int32(2), Int32(2)), (Int32(2), Int32(3))], 4, 4)
check(sort(cands) == sort(Int32[6, 5, 7, 2, 10, 9, 11, 14]), "collect_fill_candidates!: filled cells and their 4-neighbours, without duplicates")
mpf = newk24(inp_p); FH.update_phi0!(mpf, GRID, sp); FH.potential_filling!(mpf, GRID, sp)   # default PriorityFloodFill on the true potential
seed(i, j) = (i == 1 || i == NX || j == 1 || j == NY)
lower_nb(p, i, j) = any(((ni, nj),) -> 1 <= ni <= NX && 1 <= nj <= NY && p[ni, nj] < p[i, j], ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)))
gr = INP.mask .== 1
show1("grounded cells raised by PriorityFloodFill", count(mpf.phi0_filled[gr] .> mpf.phi0[gr]))
check(any(mpf.phi0_filled[gr] .> mpf.phi0[gr]) && all(mpf.phi0_filled[gr] .>= mpf.phi0[gr]) && mpf.phi0_filled[.!gr] == mpf.phi0[.!gr],
      "PriorityFloodFill raises grounded cells only (and does raise some)")
check(all(lower_nb(mpf.phi0_filled, i, j) for i in 1:NX, j in 1:NY if gr[i, j] && !seed(i, j)), "after PriorityFloodFill every grounded non-edge cell has a strictly lower neighbour (no pits or flats)")
hp = Float64[]; hk = Int32[]; vals = rand(64)
foreach(k -> FH.heap_push!(hp, hk, vals[k], Int32(k)), 1:64)
popped = [FH.heap_pop!(hp, hk) for _ in 1:64]
check(first.(popped) == sort(vals) && last.(popped) == sortperm(vals), "heap_push!/heap_pop!: a binary min-heap pops in increasing order")
covered!("update_phi0!", "potential_filling!", "fill_potential!", "heap_push!", "heap_pop!", "fill_if_strict_minimum!", "collect_fill_candidates!")

sub("gradients: magnitude from the true potential, directions from the filled one")
FH.update_potential_gradients!(mw, GRID)
gx = alloc_field(GRID); gy = alloc_field(GRID)
minus_gradient_x!(GRID, gx, mw.phi0); minus_gradient_y!(GRID, gy, mw.phi0)
check(mw.abs_grad_phi0 ≈ hypot.(gx, gy), "abs_grad_phi0 = |grad phi0| of the TRUE potential")
minus_gradient_x!(GRID, gx, mw.phi0_filled)
check(mw.minus_grad_phi0_x == gx, "minus_grad_phi0_x is -d/dx of the FILLED potential")

sub("stress-gradient coupling: the Kamb & Echelmeyer cone")
hbar = max(masked_mean(GRID, mw.h, sw.mask), 10.0)
kern = FH.coupling_kernel(mw, GRID, hbar)
width = 4 * hbar * mw.longcoupwater
show1("mean thickness, cone radius 4 h lcw, kernel size", (round(hbar; digits = 1), round(width; digits = 1), size(kern)))
check(sum(kern) ≈ 1 && kern == reverse(kern; dims = 1) && kern == reverse(kern; dims = 2), "the kernel is normalised and point-symmetric")
check(size(kern) == (2 * round(Int, width / DX - 0.5) + 1, 2 * round(Int, width / DY - 0.5) + 1), "kernel size from dx and dy separately (rectangular when dx != dy)")
fx, fy = (size(kern) .- 1) .÷ 2
check(all(kern[i, j] == 0 for i in axes(kern, 1), j in axes(kern, 2) if hypot(DX * (i - fx - 1), DY * (j - fy - 1)) >= width), "zero at and beyond the cone radius (physical distance)")
mtiny = newk24(; routing_scheme = GDSWarner(), coupling_length_kamb86 = 0.01)
show1("kernel size for coupling_length_kamb86 = 0.01 (cone narrower than a cell)", size(FH.coupling_kernel(mtiny, GRID, hbar)))
check(size(FH.coupling_kernel(mtiny, GRID, hbar)) == (1, 1), "a cone narrower than a cell gives a 1x1 kernel (no smoothing; the 'bump to >= 1 cell' branch has no effect, see the guide)")
FH.update_smoothed_potential_gradients!(mw, GRID, sw)
ref = similar(gx); FH.imfilter!(ref, mw.minus_grad_phi0_x, FH.centered(kern))
check(relerr(mw.minus_grad_phi0_sx, ref) < 1e-10, "smoothed x-gradient = ImageFiltering convolution of the raw one with the cone")
check(mw.abs_grad_phi0_s == abs.(mw.minus_grad_phi0_sx) .+ abs.(mw.minus_grad_phi0_sy), "abs_grad_phi0_s is |sx| + |sy| (an L1 norm, not the Euclidean magnitude)")
m00 = newk24(; routing_scheme = GDSWarner(), coupling_length_kamb86 = 0.0); FH.update_phi0!(m00, GRID, sw); FH.potential_filling!(m00, GRID, sw)
FH.update_potential_gradients!(m00, GRID); FH.update_smoothed_potential_gradients!(m00, GRID, sw)
check(m00.minus_grad_phi0_sx == m00.minus_grad_phi0_x && m00.minus_grad_phi0_sy == m00.minus_grad_phi0_y, "longcoupwater = 0: the 'smoothed' gradient is the raw one")
covered!("update_potential_gradients!", "coupling_kernel", "update_smoothed_potential_gradients!")

sub("routing weights and corfac")
function weights_ok(sxr, syr, dxr, dyr)
    ws = [FH.routing_weight(sxr, syr, di, dj, dxr, dyr, eps()) for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))]
    return isapprox(sum(max.(ws, 0)), 1; atol = 1e-12) && ws[1] ≈ -ws[2] && ws[3] ≈ -ws[4]
end
ok_w = all(weights_ok(randn(), randn(), 500 + 1000rand(), 500 + 1000rand()) for _ in 1:200)
check(ok_w, "routing_weight: the positive shares to the 4 neighbours sum to 1; opposite faces get opposite signs")
FH.update_q!(mw, GRID, sw)
s_abs = hypot.(mw.minus_grad_phi0_sx, mw.minus_grad_phi0_sy)
check(mw.corfac ≈ (abs.(mw.minus_grad_phi0_sx) .* DY .+ abs.(mw.minus_grad_phi0_sy) .* DX) ./ (s_abs .+ eps()), "corfac = (|sx| dy + |sy| dx)/|s|: the cell's cross-section normal to the flow")

sub("four routing algorithms, one answer")
function set_source!(m)
    @. m.mdot_total = m.mdot_fixed + m.i_eb
    return m
end
set_source!(mw); FH.update_psi_out!(mw, GRID, sw); psi_rec = copy(mw.psi_out)
mw.psi_out .= -1; FH.update_psi_out_iterative!(mw, GRID, sw); psi_it = copy(mw.psi_out)
FH.invalidate_routing_tape!(mw); mw.psi_out .= -1; FH.update_psi_out_taped!(mw, GRID, sw); psi_tp = copy(mw.psi_out)
# Only grounded cells are routed: the others keep whatever psi_out held before, so compare on the mask.
gr_w = sw.mask .== 1
show1("total routed psi_out [m3/s], ops on the tape", (round(sum(psi_rec[gr_w]); sigdigits = 6), length(mw.routing_tape.w)))
check(psi_it[gr_w] == psi_rec[gr_w], "IterativePsiOut equals RecursivePsiOut bit for bit")
check(psi_tp[gr_w] == psi_rec[gr_w], "TapedPsiOut (record + replay) equals RecursivePsiOut bit for bit")
check(all(psi_it[.!gr_w] .== -1), "non-grounded cells are never written by the routing")
check(mw.routing_tape.valid, "recording marks the tape valid; invalidate_routing_tape! clears it")
mw.mdot_total .*= 2; FH.replay_routing_tape!(mw.psi_out, mw.mdot_total, sw.mask, mw.routing_tape, NX, NY, DX, DY, mw.rho_w)
check(relerr(mw.psi_out[gr_w], 2 .* psi_tp[gr_w]) < 1e-14, "replaying with a doubled source doubles psi_out (no re-recording needed)")
for (alg, f) in ((RecursivePsiOut(), FH.update_psi_out!), (IterativePsiOut(), FH.update_psi_out_iterative!))
    set_source!(mw); FH.route_psi_out!(mw, GRID, sw, alg)
    check(mw.psi_out[gr_w] == psi_rec[gr_w], "route_psi_out!(…, $(nameof(typeof(alg)))) dispatches to $(nameof(f))")
end
# Even smooth ground is not enough for an acyclic GDS graph: two cells either side of a valley line
# each have a centred gradient pointing at the other, a 2-cycle. The bump-free fixture (whose
# thickness varies as sin(y)) has such a valley:
inp_s = k24_inputs(; bumps = 0.0)
mts = newk24(inp_s; routing_scheme = GDSWarner(), coupling_length_kamb86 = 0.0); sts = newstate(inp_s)
FH.update_q!(mts, inp_s.grid, sts); set_source!(mts)
check(throws(() -> FH.route_psi_out!(mts, inp_s.grid, sts, TopologicalPsiOut())),
      "GDSWarner on a smooth bed with a valley: a 2-cycle across the valley, and the topological sort refuses it")
# A uniform flow direction is acyclic, so there the topological sort is exact.
mts.minus_grad_phi0_sx .= 1.0; mts.minus_grad_phi0_sy .= 0.5
FH.update_psi_out!(mts, inp_s.grid, sts); psi_r2 = copy(mts.psi_out)
topo_ok = try
    FH.route_psi_out!(mts, inp_s.grid, sts, TopologicalPsiOut()); true
catch e
    println("    (topological sort raised: ", sprint(showerror, e)[1:min(end, 120)], ")"); false
end
gr_s = sts.mask .== 1
check(topo_ok && relerr(mts.psi_out[gr_s], psi_r2[gr_s]) < 1e-12, "TopologicalPsiOut equals the recursion on an acyclic (uniform-direction) graph")
# A 2x2 loop of flow directions: a genuine cycle.
gc = ArrayHydroGrid(2, 2, (0.0, 2e3), (0.0, 2e3))
mc = KazmierczakHydroModel(gc, zeros(2, 2), zeros(2, 2), fill(1e-24, 2, 2), fill(0.07, 2, 2), zeros(2, 2); coupling_length_kamb86 = 0.0, routing_scheme = GDSWarner(), dissipation_verbose = false)
stc2 = HydroState(gc, ones(2, 2), fill(500.0, 2, 2), zeros(2, 2))
# flow directions (1,1) -> (2,1) -> (2,2) -> (1,2) -> (1,1): every cell has one upstream neighbour
mc.minus_grad_phi0_sx .= [1.0 0.0; 0.0 -1.0]; mc.minus_grad_phi0_sy .= [0.0 -1.0; 1.0 0.0]
mc.mdot_total .= 1e-6
check(throws(() -> FH.update_psi_out_topological!(mc, gc, stc2, false), ErrorException), "TopologicalPsiOut on a cycle: an error by default")
_, lg = with_logs(() -> FH.update_psi_out_topological!(mc, gc, stc2, true))
check(warned(lg, "unprocessed") && all(isfinite, mc.psi_out), "allow_cycles = true: a warning and a finite (truncated) field")
FH.update_psi_out!(mc, gc, stc2)
check(all(isfinite, mc.psi_out) && all(>=(0), mc.psi_out), "the recursion does not fail on a cycle: it silently reads a partial value (see the guide)")
# The cap on cells per sweep, shared by the recursive, iterative and taped versions.
mcap = newk24(; routing_scheme = GDSWarner(), max_psi_out_calls = 25); FH.update_q!(mcap, GRID, sw); set_source!(mcap)
_, lg1 = with_logs(() -> FH.update_psi_out!(mcap, GRID, sw)); pc1 = copy(mcap.psi_out)
_, lg2 = with_logs(() -> FH.update_psi_out_iterative!(mcap, GRID, sw)); pc2 = copy(mcap.psi_out)
FH.invalidate_routing_tape!(mcap)
_, lg3 = with_logs(() -> FH.update_psi_out_taped!(mcap, GRID, sw)); pc3 = copy(mcap.psi_out)
check(warned(lg1, "max_psi_out_calls") && warned(lg2, "max_psi_out_calls") && warned(lg3, "max_psi_out_calls"), "max_psi_out_calls = 25: all three versions warn")
check(pc1 == pc2 == pc3 && all(>=(0), pc1), "...and cut the traversal off identically")
tp2 = FH.RoutingTape{Float64}(2, 2); FH.push_tape_op!(tp2, Int32(1), Int32(2), Int32(2), Int32(2), 0.5)
check(tp2.dst_i == [1] && tp2.src_j == [2] && tp2.w == [0.5], "push_tape_op! appends one 'dst += src * w' operation")
mrt = newk24(; routing_scheme = GDSWarner()); FH.update_q!(mrt, GRID, sw)
check(mrt.routing_tape.valid && length(mrt.routing_tape.w) > 0, "update_q! invalidates the tape, and resolve_q! re-records it on first use")
FH.record_routing_tape!(mrt.routing_tape, mrt, GRID, sw)
FH.record_routing_tape_kernel!(mrt.routing_tape, sw.mask, mrt.minus_grad_phi0_sx, mrt.minus_grad_phi0_sy, NX, NY, DX, DY, eps(), 10^6)
check(mrt.routing_tape.valid, "record_routing_tape!/record_routing_tape_kernel! can be called directly")
covered!("update_q!", "route_psi_out!", "invalidate_routing_tape!", "update_psi_out_taped!", "record_routing_tape!", "push_tape_op!", "record_routing_tape_kernel!",
         "replay_routing_tape!", "routing_weight", "accumulate_psi_out!", "update_psi_out!", "update_psi_out_iterative!", "update_psi_out_topological!")

sub("q from psi_out, and the three resolve_q! fixed points")
mqq = newk24(; routing_scheme = GDSWarner(), dissipation_melt = false); stq = newstate()
FH.update_q!(mqq, GRID, stq)
check(mqq.q ≈ clamp.(mqq.psi_out ./ (mqq.corfac .+ eps()), 0.0, Inf), "QFromOutflow: q = psi_out / (corfac + eps), clamped to [q_min, q_max]")
qcap = 0.3 * maximum(mqq.q)
mqq2 = newk24(; routing_scheme = GDSWarner(), dissipation_melt = false, q_max = qcap); FH.update_q!(mqq2, GRID, stq)
check(maximum(mqq2.q) == qcap && FH.update_q_from_psi_out!(mqq2) === nothing, "q_max caps q (update_q_from_psi_out!(model), the one-argument method)")
# (Off, NoFriction): one pass
mone = newk24(; routing_scheme = GDSWarner(), dissipation_melt = false); s1 = newstate(); FH.update_q!(mone, GRID, s1)
check(mone.mdot_total ≈ mone.mdot_fixed .+ mone.i_eb && all(iszero, mone.Q_b), "dissipation off, no friction: source = mdot_fixed + i_eb, one routing pass")
# (On, NoFriction): Picard on the dissipation melt
mon = newk24(; routing_scheme = GDSWarner()); s2 = newstate()
_, lgd = with_logs(() -> FH.update_q!(mon, GRID, s2))
qfix = copy(mon.q)
@. mon.mdot_total = mon.mdot_fixed + mon.i_eb; FH.add_friction_term!(mon); FH.add_dissipation_term!(mon, FH.DissipationMeltOn())
FH.route_psi_out!(mon, GRID, s2); FH.update_q_from_psi_out!(mon, GRID, s2)
show1("dissipation Picard: relative change of q in one more sweep", relerr(mon.q, qfix))
check(relerr(mon.q, qfix) < 1e-10, "dissipation on: the converged q is a fixed point of source(q) -> route -> q")
check(mon.Q_diss ≈ abs.(mon.q .* mon.abs_grad_phi0), "CellCentredDissipation: Q_diss = |q| |grad phi0| [W m^-2]")
FH.add_dissipation_term!(mon, FH.DissipationMeltOff())
check(all(iszero, mon.Q_diss), "add_dissipation_term!(…, DissipationMeltOff()) zeroes Q_diss")
mvb = newk24(; routing_scheme = GDSWarner(), dissipation_verbose = true)
_, lgv = with_logs(() -> FH.update_q!(mvb, GRID, newstate()))
check(any(l -> occursin("Water flux Picard loop: converged", string(l.message)), lgv), "dissipation_verbose = true logs the loop's convergence")
# (any, N-dependent law): joint (q, N) Picard
mqn = newk24(; routing_scheme = GDSWarner(), sliding_law = RegularizedCoulombSlidingLaw(c_till = 0.3)); s3 = newstate()
FH.update_q!(mqn, GRID, s3)
q1, N1 = copy(mqn.q), copy(s3.N)
FH.update_tau_b!(mqn, s3, mqn.sliding_law); @. mqn.mdot_total = mqn.mdot_fixed + mqn.i_eb
FH.add_friction_term!(mqn); FH.add_dissipation_term!(mqn, mqn.dissipation_melt)
FH.route_psi_out!(mqn, GRID, s3); FH.update_q_from_psi_out!(mqn, GRID, s3); update_N!(mqn, GRID, s3)
show1("(q, N) Picard: relative change in one more sweep (q, N)", (relerr(mqn.q, q1), relerr(s3.N, N1)))
check(relerr(mqn.q, q1) < 1e-6 && relerr(s3.N, N1) < 1e-6, "N-dependent law: (q, N) is a joint fixed point after resolve_q!")
check(maximum(mqn.Q_b) > 0 && mqn.Q_b ≈ mqn.tau_b .* mqn.abs_v_b, "the frictional heat tau_b |u_b| is part of the source")
mqv = newk24(; routing_scheme = GDSWarner(), sliding_law = rc, qN_verbose = true, max_qN_iters = 2)
_, lgq = with_logs(() -> FH.update_q!(mqv, GRID, newstate()))
check(any(l -> occursin("did NOT converge", string(l.message)), lgq), "max_qN_iters = 2: the loop stops and says it did not converge")
FH.resolve_q!(mone, GRID, s1, FH.DissipationMeltOff(), NoFrictionSlidingLaw())
covered!("add_dissipation_term!", "add_friction_term!", "update_q_from_psi_out!", "resolve_q!")

sub("state.W: three closures")
mW = newk24(; routing_scheme = GDSWarner(), dissipation_melt = false); sW = newstate(); update_steady_state!(mW, GRID, sW)
Wd = copy(sW.W); gm = masked_mean(GRID, mW.abs_grad_phi0, sW.mask)
check(Wd ≈ (0.1 .* 1000 .* mW.q .^ 2 ./ (4 .* gm .+ eps())) .^ (1 / 3), "DarcyWeisbach (MeanGradient): W = (f rho_w q^2 / (4 <|grad phi0|>))^(1/3)")
FH.update_W!(mW, GRID, sW, DarcyWeisbachThickness(gradient_convention = LocalGradient()))
check(sW.W ≈ (0.1 .* 1000 .* mW.q .^ 2 ./ (4 .* mW.abs_grad_phi0 .+ eps())) .^ (1 / 3), "DarcyWeisbach (LocalGradient) uses the local |grad phi0|")
FH.update_W!(mW, GRID, sW, LaminarThickness())
gs = masked_mean(GRID, mW.abs_grad_phi0_s, sW.mask)
check(sW.W ≈ (12 .* mW.eta_w .* mW.q ./ gs) .^ (1 / 3), "Laminar (MeanGradient): W = (12 eta_w q / <|grad phi0|_s>)^(1/3)")
FH.update_W!(mW, GRID, sW, LaminarThickness(gradient_convention = LocalGradient()))
check(sW.W ≈ (12 .* mW.eta_w .* mW.q ./ (mW.abs_grad_phi0_s .+ eps())) .^ (1 / 3), "Laminar (LocalGradient)")
FH.update_W!(mW, GRID, sW, ArealConduitThickness())
check(sW.W == mW.S_inf ./ mW.l_c, "ArealConduitThickness: W = S_inf / l_c, unclamped")
mWc = newk24(; routing_scheme = GDSWarner(), dissipation_melt = false, Wmin = 1e-3, Wmax = 2e-3); sWc = newstate(); update_steady_state!(mWc, GRID, sWc)
check(all(w -> 1e-3 <= w <= 2e-3, sWc.W), "Wmin/Wmax clamp the sheet closures")
FH.update_W_darcy_weisbach!(mW, GRID, sW, MeanGradient()); FH.update_W_laminar!(mW, GRID, sW, MeanGradient()); FH.update_W!(mW, GRID, sW)
check(sW.W == Wd, "update_W!(model, grid, state) dispatches on model.water_thickness_algorithm")
covered!("update_W!", "update_W_darcy_weisbach!", "update_W_laminar!")

sub("edge case: a domain with no grounded cell (masked_mean is 0/0)")
mfl = newk24(); sfl = HydroState(GRID, zeros(NX, NY), INP.h, INP.b)
err = try FH.update_q!(mfl, GRID, sfl); nothing catch e; e end
show1("update_q! with the smoothing on", err === nothing ? "ran" : string(typeof(err)))
check(err isa InexactError, "with smoothing, no grounded cell gives mean thickness NaN, and round(Int, NaN) in coupling_kernel throws (pitfall)")
mfl0 = newk24(; coupling_length_kamb86 = 0.0); sfl0 = HydroState(GRID, zeros(NX, NY), INP.h, INP.b)
update_steady_state!(mfl0, GRID, sfl0)
check(all(iszero, sfl0.N) && all(iszero, mfl0.q), "without smoothing it runs: N = 0, q = 0 everywhere")
check(all(isnan, sfl0.W), "...but W is NaN everywhere (MeanGradient divides by the 0/0 grounded mean)")

# =============================================================================================
# PART 15 -- models/kazmierczak2024/routing.jl: the Le Brocq et al. (2006) routing schemes
# =============================================================================================
# Every scheme other than GDSWarner is expressed as outflow fractions w8[d, i, j] (the share of
# cell (i,j)'s outflow sent to its neighbour in direction d of ROUTE_OFFSETS). `prepare_routing!`
# builds the routing surface (Warner/Quinn/Tarboton/ModifiedTarboton: smooth the potential, then
# fill it; GDS schemes: fill, then smooth the gradient), `compute_routing_weights!` fills w8,
# `record_routing_tape_weights_kernel!` records the traversal. With the 4-neighbour schemes the
# face fluxes Fx, Fy are formed (`update_face_fluxes!`), which give q at the cell centre
# (`QFromFaceAverage`) and the face-assembled dissipation (`FaceDissipation`). Warner (the
# default) only sends water downhill, so on a priority-flood-filled surface every drop of the
# source leaves the grounded domain: routing conserves water exactly.
# `freeze_on_capacity!` reports, for a coupled ice model, how much water reaches each cell.
part(15, "models/kazmierczak2024/routing.jl", "ROUTING SCHEMES, FACE FLUXES, FREEZE-ON CAPACITY")
check(all(d -> FH.ROUTE_OFFSETS[FH.ROUTE_OPPOSITE[d]] == .-FH.ROUTE_OFFSETS[d], 1:8), "ROUTE_OPPOSITE[d] points back along ROUTE_OFFSETS[d]")
check(all(((c, dg),) -> c <= 4 && dg > 4 && count(k -> FH.ROUTE_OFFSETS[c][k] == FH.ROUTE_OFFSETS[dg][k], 1:2) == 1, FH.TARBOTON_FACETS),
      "each Tarboton facet pairs a cardinal with an adjacent diagonal neighbour")
check(FH.in_domain(1, 1, 3, 3) && !FH.in_domain(0, 1, 3, 3) && !FH.in_domain(1, 4, 3, 3), "in_domain")
check(FH.needs_face_fluxes(newk24()) && !FH.needs_face_fluxes(newk24(; routing_scheme = GDSWarner())) &&
      FH.needs_face_fluxes(newk24(; q_conversion = QFromFaceAverage(), routing_scheme = GDSWarner())) &&
      !FH.needs_face_fluxes(newk24(; dissipation_melt = false, q_conversion = QFromOutflow())), "needs_face_fluxes: QFromFaceAverage, or FaceDissipation with the dissipation on")

sub("cell weights of each scheme on hand-made surfaces")
W8 = zeros(8, 3, 3); phi = zeros(3, 3); phi[2, 2] = 1.0
FH.cell_weights!(W8, Quinn(original = true), 2, 2, phi, phi, phi, 3, 3, 1e3, 1e3)
check(isapprox(W8[1:4, 2, 2], fill(1 / 6, 4); rtol = 1e-12) && isapprox(W8[5:8, 2, 2], fill(1 / 12, 4); rtol = 1e-12), "Quinn(original): cardinal 1/6, diagonal 1/12 for equal drops")
W8 .= 0; FH.cell_weights!(W8, Quinn(), 2, 2, phi, phi, phi, 3, 3, 1e3, 1e3)
check(all(W8[:, 2, 2] .≈ 1 / 8), "Quinn (Le Brocq Eq. 8): shares proportional to the drop only, 1/8 each")
W8 .= 0
phw = fill(5.0, 3, 3); phw[2, 2] = 4.0; phw[1, 2] = 3.0; phw[2, 1] = 2.0     # west drop 1, south drop 2
FH.cell_weights!(W8, Warner(), 2, 2, phw, phw, phw, 3, 3, 1e3, 2e3)
check(W8[1, 2, 2] ≈ (1 * 2) / (1 * 2 + 2 * 0.5) && W8[3, 2, 2] ≈ (2 * 0.5) / 3 && sum(W8[:, 2, 2]) ≈ 1, "Warner: drop x (face length / distance), normalised (dx = 1 km, dy = 2 km)")
W8 .= 0; FH.cell_weights!(W8, GDSWarner(), 2, 2, phw, fill(3.0, 3, 3), fill(-1.0, 3, 3), 3, 3, 1e3, 1e3)
check(W8[2, 2, 2] ≈ 0.75 && W8[3, 2, 2] ≈ 0.25 && W8[1, 2, 2] == 0, "GDSWarner: positive components of the smoothed gradient (here +x 3/4, -y 1/4)")
a, bb = 1.0, 0.4                           # plane phi = -(a x + b y) on a square grid: flow at atan(b/a)
php = [-(a * i + bb * j) * 1e3 for i in 1:3, j in 1:3]; W8 .= 0
FH.cell_weights!(W8, Tarboton(), 2, 2, php, php, php, 3, 3, 1e3, 1e3)
check(W8[8, 2, 2] ≈ atan(bb, a) / (pi / 4) && W8[2, 2, 2] ≈ 1 - atan(bb, a) / (pi / 4) && sum(W8[:, 2, 2]) ≈ 1,
      "Tarboton (D-infinity): a plane's flow angle splits between east and north-east by angle")
W8 .= 0; FH.angle_split!(W8, 2, 2, 1.0, 0.0, 1e3, 1e3)
check(W8[2, 2, 2] == 1.0 && sum(W8) == 1.0, "angle_split!: flow along +x goes entirely east")
W8 .= 0; FH.angle_split!(W8, 2, 2, a, bb, 1e3, 1e3)
check(W8[8, 2, 2] ≈ atan(bb, a) / (pi / 4) && W8[2, 2, 2] ≈ 1 - atan(bb, a) / (pi / 4), "angle_split! agrees with Tarboton on a plane")
W8 .= 0; FH.cell_weights!(W8, ModifiedTarboton(), 2, 2, php, fill(a, 3, 3), fill(bb, 3, 3), 3, 3, 1e3, 1e3)
W8b = zeros(8, 3, 3); FH.cell_weights!(W8b, GDSTarboton(), 2, 2, php, fill(a, 3, 3), fill(bb, 3, 3), 3, 3, 1e3, 1e3)
check(W8 == W8b && W8[8, 2, 2] ≈ atan(bb, a) / (pi / 4), "ModifiedTarboton and GDSTarboton split the given direction the same way")
covered!("ROUTE_OFFSETS", "ROUTE_OPPOSITE", "TARBOTON_FACETS", "in_domain", "needs_face_fluxes", "cell_weights!", "angle_split!")

sub("Warner on the fixture: weights, conservation, face fluxes")
mwr = newk24(; dissipation_melt = false); swr = newstate()
update_steady_state!(mwr, GRID, swr)
W8f = mwr.routing_tape.w8
sums = [sum(@view W8f[:, i, j]) for i in 1:NX, j in 1:NY]
check(all(s -> s == 0 || isapprox(s, 1; atol = 1e-12), sums[gr]) && all(iszero, sums[.!gr]), "outflow fractions sum to 1 (or 0 at a sink) on grounded cells, 0 elsewhere")
function exits(m, Nx, Ny, mask)
    W = m.routing_tape.w8; e = 0.0
    for j in 1:Ny, i in 1:Nx
        mask[i, j] == 1 || continue
        tot = sum(@view W[:, i, j])
        tot == 0 && (e += m.psi_out[i, j]; continue)
        for d in 1:8
            ni, nj = i + FH.ROUTE_OFFSETS[d][1], j + FH.ROUTE_OFFSETS[d][2]
            (FH.in_domain(ni, nj, Nx, Ny) && mask[ni, nj] == 1) || (e += m.psi_out[i, j] * W[d, i, j])
        end
    end
    return e
end
source = sum(mwr.mdot_total[gr]) * DX * DY / mwr.rho_w
show1("source vs water leaving the grounded domain [m3/s]", (source, exits(mwr, NX, NY, INP.mask)))
check(isapprox(exits(mwr, NX, NY, INP.mask), source; rtol = 1e-12), "Warner + PriorityFloodFill conserve water: everything routed leaves at the grounding line or edge")
Fx, Fy = mwr.routing_tape.Fx, mwr.routing_tape.Fy
div_ok = all(isapprox(Fx[i + 1, j] - Fx[i, j] + Fy[i, j + 1] - Fy[i, j], mwr.mdot_total[i, j] * DX * DY / mwr.rho_w; rtol = 1e-9, atol = 1e-14)
             for i in 1:NX, j in 1:NY if gr[i, j] && sums[i, j] ≈ 1)
check(div_ok, "face fluxes: net outflow through a cell's 4 faces equals its source (discrete divergence)")
ic, jc = 10, 9
check(mwr.q[ic, jc] ≈ hypot((Fx[ic, jc] + Fx[ic + 1, jc]) / (2DY), (Fy[ic, jc] + Fy[ic, jc + 1]) / (2DX)), "QFromFaceAverage: q = |face-normal fluxes averaged to the centre|")
check(all(iszero, mwr.q[.!gr]), "q is zero off the grounded mask")
covered!("prepare_routing!", "compute_routing_weights!", "routing_weights_kernel!", "record_routing_tape_weights_kernel!", "update_face_fluxes!",
         "face_fluxes_kernel!", "update_q_from_psi_out!", "q_from_psi_out!", "q_from_faces_kernel!")

sub("FaceDissipation: the energy released crossing each face")
mfd = newk24(); sfd2 = newstate(); update_steady_state!(mfd, GRID, sfd2)
t = mfd.routing_tape; ph = mfd.phi0
E_faces = sum(t.Fx[i, j] * (ph[i - 1, j] - ph[i, j]) * (gr[i - 1, j] + gr[i, j]) / 2 for i in 2:NX, j in 1:NY) +
          sum(t.Fy[i, j] * (ph[i, j - 1] - ph[i, j]) * (gr[i, j - 1] + gr[i, j]) / 2 for i in 1:NX, j in 2:NY)
E_cells = sum(t.diss) * DX * DY * mfd.L_w
show1("energy released on interior faces vs assembled into cells [W]", (E_faces, E_cells))
check(isapprox(E_cells, E_faces; rtol = 1e-10), "cells receive exactly the energy F dphi released on faces (half per side; a grounding-line face gives its grounded cell half)")
check(mfd.Q_diss ≈ t.diss .* mfd.L_w, "Q_diss = diss L_w (from the previous sweep; equal at convergence)")
show1("cells with negative face dissipation (water pushed up the true potential)", count(<(0), t.diss))
mcd = newk24(; dissipation_discretization = CellCentredDissipation(), q_conversion = QFromOutflow(), routing_scheme = GDSWarner()); scd = newstate(); update_steady_state!(mcd, GRID, scd)
check(mcd.Q_diss ≈ abs.(mcd.q .* mcd.abs_grad_phi0), "CellCentredDissipation (routing.jl method): |q grad phi0|")

sub("freeze_on_capacity!: the water a coupled ice model may refreeze")
C = zeros(NX, NY); freeze_on_capacity!(C, mwr, GRID, swr)
Psi_in = [gr[i, j] ? sum(begin
            ni, nj = i + FH.ROUTE_OFFSETS[d][1], j + FH.ROUTE_OFFSETS[d][2]
            (FH.in_domain(ni, nj, NX, NY) && gr[ni, nj]) ? mwr.psi_out[ni, nj] * W8f[FH.ROUTE_OPPOSITE[d], ni, nj] : 0.0
          end for d in 1:8) : 0.0 for i in 1:NX, j in 1:NY]
check(C ≈ max.(1000 .* Psi_in ./ (DX * DY) .+ mwr.i_eb, 0) ./ 917 .* gr, "C = (rho_w Psi_in/(dx dy) + i_eb)/rho_i, inflow from upstream neighbours only")
check(sum(Psi_in) ≈ sum(mwr.psi_out .* [sum(begin
            ni, nj = i + FH.ROUTE_OFFSETS[d][1], j + FH.ROUTE_OFFSETS[d][2]
            (FH.in_domain(ni, nj, NX, NY) && gr[ni, nj]) ? W8f[d, i, j] : 0.0 end for d in 1:8) for i in 1:NX, j in 1:NY]), "total inflow = total outflow sent to grounded neighbours")
mgc = newk24(; routing_scheme = GDSWarner(), dissipation_melt = false); sgc = newstate(); update_steady_state!(mgc, GRID, sgc)
C2 = zeros(NX, NY); freeze_on_capacity!(C2, mgc, GRID, sgc)
check(all(isfinite, C2) && all(>=(0), C2) && maximum(abs, mgc.routing_tape.w8) > 0, "GDSWarner: the weights are computed on demand for the capacity")
FH.freeze_on_capacity_kernel!(C2, mwr.psi_out, W8f, mwr.i_eb, INP.mask, NX, NY, DX, DY, 1000.0, 917.0)
check(C2 ≈ C, "freeze_on_capacity_kernel! called directly agrees")
covered!("add_dissipation_term!", "freeze_on_capacity!", "freeze_on_capacity_kernel!")

sub("all six schemes run on the fixture")
for sch in (Warner(), GDSWarner(), Quinn(), Quinn(original = true), Tarboton(), ModifiedTarboton(), GDSTarboton())
    m_ = newk24(; routing_scheme = sch, dissipation_melt = false); s_ = newstate(); update_steady_state!(m_, GRID, s_)
    show1("$(sch): sum q, mean N/Po (grounded)", (round(sum(m_.q); sigdigits = 4), round(mean(s_.N[gr] ./ m_.Po[gr]); digits = 3)))
    check(all(isfinite, m_.q) && all(isfinite, s_.N) && all(>=(0), m_.psi_out), "$(sch): finite q and N, psi_out >= 0")
end

# =============================================================================================
# PART 16 -- models/kazmierczak2024/effective_pressure.jl: from q to N
# =============================================================================================
# For each grounded cell: conduit discharge Q = q l_c; far-field conduit area S_inf from a
# Manning-Strickler law, S_inf = K^(-1/alpha) |grad phi0|^((1-beta)/alpha) Q^(1/alpha);
# conduit thickness H blending a hard bed (sqrt(S_inf)) and a soft bed (canal H_0 to film
# sqrt(S_inf)/F_till through exp(-Q/Q_c)) with kappa; far-field N_inf from opening (sliding over
# bumps, melt) = closing (creep, (N/n)^n A); and N = erf(sqrt(pi) phi0 / (2 N_inf)) N_inf, which
# goes to 0 at the grounding line (phi0 -> 0 at flotation) and to N_inf far inland. The erf argument
# uses rho_sw (keyword, default rho_w = the paper's Eq. 7) for the b term: flotation is decided with
# seawater, so with rho_w the argument at the GL is (rho_sw - rho_w) g |b| (~0.27 MPa per km of bed
# depth), not 0, and N stays there at ~0.87 N_inf on a soft bed (N_inf ~ 0.25 MPa). Set rho_sw to the
# ice model's flotation density (e.g. 1028) in coupled runs; routing and gradients keep phi0.
# `update_N!` does all of this in one fused, threaded pass over grounded cells; the standalone
# `update_Q!`, `update_S_inf!`, `update_H!`, `update_N_inf!` broadcasts compute the same values.
part(16, "models/kazmierczak2024/effective_pressure.jl", "EFFECTIVE PRESSURE")
mN = newk24(); sN = newstate(); update_steady_state!(mN, GRID, sN)
Nfused = copy(sN.N); Hf, Sf, Ninff = copy(mN.H), copy(mN.S_inf), copy(mN.N_inf)
FH.update_Q!(mN, GRID); FH.update_S_inf!(mN, GRID); FH.update_H!(mN, GRID); update_Po!(mN, GRID, sN); FH.update_N_inf!(mN, GRID)
Nstep = [mN.N_inf[i, j] == 0 ? 0.0 : max(0.0, FH.erf(sqrt(pi) * (mN.rho_i * mN.g * sN.h[i, j] + mN.rho_sw * mN.g * sN.b[i, j]) / (2 * mN.N_inf[i, j])) * mN.N_inf[i, j]) for i in 1:NX, j in 1:NY]
check(Nfused[gr] == Nstep[gr] && Hf[gr] == mN.H[gr] && Sf[gr] == mN.S_inf[gr] && Ninff[gr] == mN.N_inf[gr],
      "fused update_N! equals update_Q!, update_S_inf!, update_H!, update_Po!, update_N_inf!, erf step by step (bit for bit on grounded cells)")
check(all(iszero, Nfused[.!gr]) && all(iszero, Hf[.!gr]), "N and the conduit fields are zero off the grounded mask")
i0, j0 = 12, 7
Qh = mN.q[i0, j0] * mN.l_c
Sh = mN.K^(-1 / 1.25) * mN.abs_grad_phi0[i0, j0]^((1 - 1.5) / 1.25) * Qh^(1 / 1.25)
Hh = (1 - mN.kappa[i0, j0]) * sqrt(Sh) + mN.kappa[i0, j0] * (0.1 + (sqrt(Sh) / 1.1 - 0.1) * exp(-Qh / 1.0))
Ninfh = ((Hh / Sh)^2 * (917 * 3.34e5 * mN.abs_v_b[i0, j0] * 0.1 + Qh * mN.abs_grad_phi0[i0, j0]) / (2 * 3.0^-3 * 917 * 3.34e5 * mN.A_visc[i0, j0]))^(1 / 3)
Ninfh = clamp(Ninfh, 0.0, 917 * 9.81 * sN.h[i0, j0])
show1("cell (12,7): Q, S_inf, H, N_inf, N", (Qh, Sh, Hh, Ninfh, Nfused[i0, j0]))
check(isapprox(Sh, Sf[i0, j0]; rtol = 1e-12) && isapprox(Hh, Hf[i0, j0]; rtol = 1e-12) && isapprox(Ninfh, Ninff[i0, j0]; rtol = 1e-12),
      "S_inf, H and N_inf at one cell from the K24 formulas by hand")
check(all(0 .<= sN.N[gr] .<= mN.N_inf[gr] .<= mN.Po[gr]), "0 <= N <= N_inf <= Po on grounded cells")
Nloop = zeros(NX, NY); ws = mN.workspace
for j in 1:NY
    FH.update_N_grounded_column!(j, Nloop, copy(ws.Q), copy(ws.S_inf), copy(ws.H_hard), copy(ws.H_soft), copy(ws.H), copy(ws.Po), copy(ws.N_inf),
        mN.q, mN.abs_grad_phi0, mN.kappa, mN.abs_v_b, mN.A_visc, mN.phi0, sN.h, sN.mask, NX, NY, mN.l_c, mN.K^(-1 / mN.alpha), (1 - mN.beta) / mN.alpha,
        1 / mN.alpha, mN.H_0, mN.F_till, mN.Q_c, mN.rho_i, mN.g, mN.L_w, mN.h_b, 1.0, 1.0, 2.0 * mN.n^(-mN.n) * mN.rho_i * mN.L_w, 1.0 / mN.n, mN.sigmat, sqrt(pi), sN.b, mN.rho_sw)
end
update_N!(mN, GRID, sN)
check(Nloop == sN.N, "update_N_grounded_kernel! ($(Threads.nthreads()) threads) equals a serial loop over update_N_grounded_column!")
check(FH.effective_Q_c(BothDrainage(), 2.0) == 2.0 && FH.effective_Q_c(EfficientOnly(), 2.0) == 0.0 && FH.effective_Q_c(InefficientOnly(), 2.0) == Inf,
      "effective_Q_c: Q_c, 0 (efficient), Inf (inefficient) -- the corrected Sect. 3.2 assignment")
check(FH.opening_coefficients(BothDrainage(), Float64) == (1.0, 1.0) && FH.opening_coefficients(EfficientOnly(), Float64) == (0.0, 1.0) &&
      FH.opening_coefficients(InefficientOnly(), Float64) == (1.0, 0.0), "opening_coefficients: (sliding, melt) terms kept per drainage mode")
mE = newk24(; drainage_mode = EfficientOnly()); sE = newstate(); update_steady_state!(mE, GRID, sE)
mI = newk24(; drainage_mode = InefficientOnly()); sI = newstate(); update_steady_state!(mI, GRID, sI)
pos = gr .& (mE.Q .> 0)
check(all(mE.H_soft[pos] .== mE.H_0), "EfficientOnly: the soft-bed conduit is the canal, H_soft = H_0, wherever Q > 0")
check(mI.H_soft[gr] ≈ sqrt.(mI.S_inf[gr]) ./ mI.F_till, "InefficientOnly: H_soft = sqrt(S_inf)/F_till (inter-clastic film)")
check(N_responds_to_ub(newk24()) && !N_responds_to_ub(mE) && N_responds_to_ub(mI), "N_responds_to_ub: false only for EfficientOnly (no sliding term in N_inf)")

sub("N_from_ub!: N for a new sliding speed, routing held")
mU = newk24(; sliding_law = PrescribedFieldSlidingLaw(GRID, fill(5e4, NX, NY))); sU = newstate(); update_steady_state!(mU, GRID, sU)
q_before = copy(mU.q); N1u = copy(sU.N)
N2u = copy(N_from_ub!(mU, GRID, sU, 2 .* INP.abs_v_b))
check(mU.q == q_before && mU.abs_v_b == 2 .* INP.abs_v_b, "the routing (q) is held; only abs_v_b changes")
check(all(N2u[gr] .>= N1u[gr] .- 1e-9 .* mU.Po[gr]) && maximum(N2u[gr] .- N1u[gr]) > 0, "faster sliding opens cavities: N rises (x erf(c/x) is increasing in N_inf)")
mU2 = newk24(; sliding_law = PrescribedFieldSlidingLaw(GRID, fill(5e4, NX, NY))); sU2 = newstate(); update_steady_state!(mU2, GRID, sU2)
mU2.abs_v_b .= 2 .* INP.abs_v_b; update_N!(mU2, GRID, sU2)
check(sU2.N == N2u, "N_from_ub! is update_N! after setting abs_v_b")
mz = newk24(); sz = newstate(); mz.q .= 0; mz.abs_grad_phi0 .= 0
FH.update_Q!(mz, GRID); FH.update_S_inf!(mz, GRID); update_Po!(mz, GRID, sz); FH.update_H!(mz, GRID); FH.update_N_inf!(mz, GRID)
check(all(iszero, mz.S_inf) && mz.N_inf == mz.Po, "Q = 0 and zero gradient: S_inf = 0 (not 0^-x * 0 = NaN) and N_inf = Po")
mze = newk24(; drainage_mode = EfficientOnly()); mze.Q .= 0; mze.S_inf .= 0; FH.update_H!(mze, GRID)
check(all(iszero, mze.H_soft), "EfficientOnly with Q = 0: the 0/0 in exp(-Q/Q_c) resolved to H_soft = 0")
covered!("update_N!", "N_responds_to_ub", "N_from_ub!", "update_N_grounded_kernel!", "update_N_grounded_column!", "effective_Q_c", "update_H!",
         "update_S_inf!", "opening_coefficients", "update_N_inf!", "update_Q!")

# =============================================================================================
# PART 17 -- models/kazmierczak2024/run.jl: one steady-state update
# =============================================================================================
# `update_steady_state!(model, grid, state)` = update_q!, update_N!, update_W!, in that order: W
# needs S_inf (ArealConduitThickness), which update_N! keeps current, and nothing reads W.
part(17, "models/kazmierczak2024/run.jl", "update_steady_state!")
ma = newk24(); sa = newstate(); update_steady_state!(ma, GRID, sa)
mb2 = newk24(); sb2 = newstate(); FH.update_q!(mb2, GRID, sb2); update_N!(mb2, GRID, sb2); FH.update_W!(mb2, GRID, sb2)
check(sa.N == sb2.N && sa.W == sb2.W && ma.q == mb2.q, "update_steady_state! = update_q! then update_N! then update_W!")
sr = newstate(); run!(SteadyStateSimulation(newk24(), GRID, sr))
check(sr.N == sa.N, "run!(SteadyStateSimulation(K24 model, ...)) gives the same N")
show1("N/Po on grounded cells: min, median, max", round.((minimum(sa.N[gr] ./ ma.Po[gr]), median(sa.N[gr] ./ ma.Po[gr]), maximum(sa.N[gr] ./ ma.Po[gr])); digits = 3))
covered!("update_steady_state!")

# =============================================================================================
# PART 18 -- models/kazmierczak2024/data_loaders.jl: real inputs
# =============================================================================================
# Two loaders return the same 13-tuple (Nx, Ny, xlims, ylims, mask, h, b, abs_v_b, A_visc, G, q_T,
# tau_b, kappa) in SI units: `load_Kazmierczak` reads KORI-ULB's Thwaites .mat output (per-year
# ub and A converted to per-second, q_T from the two lowest ice temperature levels, x and y
# swapped in the file), `load_yelmox` a Yelmo restart NetCDF (km -> m, m/yr -> m/s, mW -> W).
# `initialize_κ` sets the bed type from the bed elevation. `load_Kazmierczak_melt` returns
# KORI-ULB's own melt rate for comparison only.
part(18, "models/kazmierczak2024/data_loaders.jl", "DATA LOADERS")
bk = [-2000.0 -1500 -1000; -600 -500 0]
check(FH.initialize_κ(2, 3, bk; bed_rheology = :hard) == zeros(2, 3) && FH.initialize_κ(2, 3, bk; bed_rheology = :soft) == ones(2, 3),
      "initialize_κ: :hard -> 0, :soft -> 1")
check(FH.initialize_κ(2, 3, bk; bed_rheology = :mixed) == [1.0 1 0; 0 0 0], ":mixed: soft where b < -1000 m")
check(FH.initialize_κ(2, 3, bk; bed_rheology = :mixed_smooth) ≈ [1.0 1 0.5; 0.1 0 0], ":mixed_smooth: 0 above -500 m, 1 below -1500 m, linear between")
check(throws(() -> FH.initialize_κ(2, 3, bk; bed_rheology = :clay), ErrorException), "an unknown bed_rheology is an error")
covered!("initialize_κ")
const THW = joinpath(DATA, "Kazmierczak2024", "THWAITES2km_m3_HAB_toto.mat")
const YX16 = joinpath(DATA, "yelmox", "16km", "1990-2019", "restart-0.000-kyr", "yelmo_restart.nc")
if isfile(THW)
    thw = load_Kazmierczak(THW; bed_rheology = :mixed_smooth)
    Nxt, Nyt, xlt, ylt, maskt, ht, bt, ubt, At, Gt, qTt, taubt, kapt = thw
    gt = ArrayHydroGrid(Nxt, Nyt, xlt, ylt)
    show1("Thwaites: Nx, Ny, dx, dy, grounded cells", (Nxt, Nyt, gt.dx, gt.dy, count(==(1), maskt)))
    check(gt.dx ≈ 2000 && gt.dy ≈ 2000, "load_Kazmierczak: 2 km grid, km converted to m")
    check(0 < perSecond2perYear(maximum(ubt)) < 1e4 && 1e-26 < median(At) < 1e-22, "u_b and A converted from per-year: max u_b a few km/yr, A ~ 1e-24 Pa^-3 s^-1")
    check(0.05 < median(Gt) < 0.2 && median(abs.(qTt[maskt .== 1])) < 0.5, "G ~ 0.1 W m^-2; q_T a fraction of a W m^-2")
    mel = load_Kazmierczak_melt(THW)
    show1("KORI-ULB melt on grounded cells: median [kg m^-2 s^-1], [mm/yr w.e.]", (median(mel[maskt .== 1]), perSecond2perYear(median(mel[maskt .== 1]))))
    check(size(mel) == (Nxt, Nyt) && all(isfinite, mel), "load_Kazmierczak_melt: KORI-ULB's melt as kg m^-2 s^-1")
    covered!("load_Kazmierczak", "load_Kazmierczak_melt")
else
    skipped!("load_Kazmierczak", "the Thwaites .mat file is not under test/Kazmierczak et al 2024/input")
    skipped!("load_Kazmierczak_melt", "the Thwaites .mat file is not under test/Kazmierczak et al 2024/input")
end
if isfile(YX16)
    yx = load_yelmox(YX16)
    Nxy, Nyy, xly, yly, masky, hy, by, uby, Ay, Gy, qTy, tauby, kapy = yx
    gy = ArrayHydroGrid(Nxy, Nyy, xly, yly)
    show1("Yelmo 16 km: Nx, Ny, dx, grounded cells", (Nxy, Nyy, gy.dx, count(==(1), masky)))
    check(isapprox(gy.dx, 16000; rtol = 1e-6) && all(m -> m == 0 || m == 1, masky), "load_yelmox: 16 km (to 1e-6: the restart's coordinates are not exact), mask = f_ice f_grnd > 0 as 0/1")
    check(0.02 < median(Gy[masky .== 1]) < 0.2 && maximum(kapy) <= 1 && minimum(kapy) >= 0, "G from mW m^-2 to W m^-2; kappa in [0, 1] (:mixed_smooth)")
    covered!("load_yelmox")
else
    skipped!("load_yelmox", "the Yelmo 16 km restart is not under test/Kazmierczak et al 2024/input")
end

# =============================================================================================
# PART 19 -- models/hab/model.jl: the height-above-buoyancy model
# =============================================================================================
# HAB (Kazmierczak et al. 2022, Sec. 2.1.1) assumes a direct hydraulic connection to the ocean:
# the water pressure is a fraction P_w = 0.96 of the ocean pressure at the bed. It has five fixed
# constants (rho_sw 1027, rho_i 917, g 9.81, P_w 0.96, sigmat 0.02; no keywords) and two arrays.
part(19, "models/hab/model.jl", "HABHydroModel")
hm = HABHydroModel(GRID)
check((hm.rho_sw, hm.rho_i, hm.g, hm.P_w, hm.sigmat) == (1027.0, 917.0, 9.81, 0.96, 0.02) && size(hm.Po) == (NX, NY), "constants and Po/p_w buffers")
check(HABHydroModel(g32).P_w isa Float32, "the float type follows the grid")
covered!("HABHydroModel")

# =============================================================================================
# PART 20 -- models/hab/effective_pressure.jl: p_w and N
# =============================================================================================
# p_w = -P_w rho_sw g b where the bed is below sea level; P_w rho_i g h under non-grounded ice; 0 where
# b >= 0 (overriding the previous rule). N = max(Po - p_w, sigmat Po).
part(20, "models/hab/effective_pressure.jl", "HAB WATER PRESSURE AND N")
gh = ArrayHydroGrid(2, 2, (0.0, 2.0), (0.0, 2.0))
sh = HydroState(gh, [1.0 1.0; 0.0 0.0], [540.0 1000.0; 300.0 50.0], [-500.0 100.0; -400.0 -900.0])
hm2 = HABHydroModel(gh); update_p_w!(hm2, gh, sh)
check(hm2.p_w[1, 1] ≈ 0.96 * 1027 * 9.81 * 500 && hm2.p_w[1, 2] == 0 && hm2.p_w[2, 1] ≈ 0.96 * 917 * 9.81 * 300 && hm2.p_w[2, 2] ≈ 0.96 * 917 * 9.81 * 50,
      "p_w: ocean pressure fraction (grounded, b < 0), 0 (b >= 0), overburden fraction (not grounded)")
update_N!(hm2, gh, sh)
Po2 = 917 * 9.81 .* sh.h
check(sh.N ≈ max.(Po2 .- hm2.p_w, 0.02 .* Po2) && sh.N[1, 1] == 0.02 * Po2[1, 1], "N = max(Po - p_w, 0.02 Po); here cell (1,1) is near flotation and hits the floor")
covered!("update_N!", "update_p_w!")

# =============================================================================================
# PART 21 -- models/hab/run.jl
# =============================================================================================
part(21, "models/hab/run.jl", "HAB update_steady_state!")
sh2 = HydroState(gh, sh.mask, sh.h, sh.b); update_steady_state!(HABHydroModel(gh), gh, sh2)
check(sh2.N == sh.N && all(iszero, sh2.W), "update_steady_state! is update_N!; HAB writes no W")
covered!("update_steady_state!")

# =============================================================================================
# PART 22 -- models/shakti/model.jl: the Shakti wrapper
# =============================================================================================
# `ShaktiHydroModel(sim)` holds a `Shakti.Simulation`, nothing else; Shakti owns its own grid,
# state and parameters. The type is defined without depending on Shakti (its field type is a
# parameter); the methods that run it come from the extension (PART 24).
part(22, "models/shakti/model.jl", "ShaktiHydroModel")
check(ShaktiHydroModel(42) isa ShaktiHydroModel{Int} && ShaktiHydroModel(42).sim == 42, "ShaktiHydroModel{S} wraps any value; S is the Simulation type in practice")
covered!("ShaktiHydroModel")

# =============================================================================================
# PART 23 -- ext/FastHydrologyMakieExt.jl: plotting
# =============================================================================================
# Loaded when CairoMakie is. `visualize_field(x, y, data)` draws a heatmap with a colorbar and a
# robust colour range (all-NaN, flat and zero fields get a sensible range instead of an error);
# `visualize_field(grid, field)` uses cell-centre coordinates from the grid's lower-left corner;
# `visualize_grid(grid)` draws the cell centres and edges of the four corners.
part(23, "FastHydrologyMakieExt.jl", "PLOTTING EXTENSION")
mkext = Base.get_extension(FastHydrology, :FastHydrologyMakieExt)
check(mkext._xcenters(GRID) ≈ ((1:NX) .- 0.5) .* DX && mkext._ycenters(GRID) ≈ ((1:NY) .- 0.5) .* DY, "_xcenters/_ycenters: cell centres from the lower-left corner")
pngN = joinpath(OUT, "tour_K24_N.png")
fig = visualize_field(GRID, mask_field(sa.N ./ 1e6, INP.mask, NaN); plot_title = "K24 fixture: N [MPa]", display_flag = false, savefig_path = pngN, colormap = :viridis)
check(fig isa CairoMakie.Figure && isfile(pngN), "visualize_field(grid, field) returns a Figure and saves it")
check(visualize_field(1:3, 1:2, fill(NaN, 3, 2); display_flag = false) isa CairoMakie.Figure &&
      visualize_field(1:3, 1:2, zeros(3, 2); display_flag = false) isa CairoMakie.Figure &&
      visualize_field(1:3, 1:2, fill(4.0, 3, 2); display_flag = false, transpose_data = false) isa CairoMakie.Figure &&
      visualize_field(1:2, 1:3, rand(3, 2); display_flag = false, transpose_data = true) isa CairoMakie.Figure,
      "all-NaN, all-zero and flat fields get a valid colour range; transpose_data works")
pngG = joinpath(OUT, "tour_grid.png")
fg = visualize_grid(GRID); CairoMakie.save(pngG, fg)
check(fg isa CairoMakie.Figure && isfile(pngG), "visualize_grid draws the four 5x5 corners")
covered!("_xcenters", "_ycenters", "visualize_field", "visualize_grid")

# =============================================================================================
# PART 24 -- ext/FastHydrologyShaktiExt.jl: running Shakti through FastHydrology
# =============================================================================================
# Loaded when Shakti is. `run!(::TimeSimulation{<:ShaktiHydroModel}; kwargs...)` is
# `Shakti.run!(sim; kwargs...)` (kwargs such as checkpoint_every pass through). `step!` is one
# `Shakti.step!` plus `total_time[] += dt[]`, which `Shakti.run!` would otherwise do: without it,
# time-dependent forcing (SeasonalMeltInput) would see the same time on every call.
part(24, "FastHydrologyShaktiExt.jl", "SHAKTI EXTENSION")
show1("Shakti backend / floattype / use_cudss", (Shakti.backend, Shakti.floattype, Shakti.use_cudss))
check(Shakti.backend == "Threads" && Shakti.use_cudss == false, "Shakti on the CPU, without CUDA (LocalPreferences.toml of the tour)")
function shakti_sim(tsteps)
    nx, ny = 4, 4
    grid = Shakti.Grid(nx, ny, 1e3, 1e3)
    state = Shakti.State(grid)
    p = Shakti.ModelParameters(e_v = 0.0)
    sl = Shakti.RegularizedCoulombSlidingLaw(0.25)
    mask = fill(Shakti.GROUNDED, nx, ny); mask[end, :] .= Shakti.OCEAN; mask[1, :] .= Shakti.LAND; mask[:, 1] .= Shakti.OTHER_BASIN
    zb = repeat(reshape(-0.02 .* grid.x, nx, 1), 1, ny)
    ieb = zeros(nx, ny); ieb[2, 2] = 3 / (grid.dx * grid.dy)
    Shakti.set_initial_conditions!(state, grid, p, sl, mask, fill(5e-25, nx, ny), zb, zb .+ 500.0, fill(0.01, nx, ny), fill(0.06, nx, ny),
                                   fill(1e-6, nx + 1, ny), zeros(nx, ny + 1), ieb, zeros(nx + 1, ny), zeros(nx, ny + 1))
    ps = Shakti.PicardSolver(500, 1e-6, Shakti.CholeskyDirectSolver(grid), grid)
    return Shakti.Simulation(grid, state, tsteps, 3600.0, p, "implicit", String[], Shakti.ConstantMeltInput(), sl; ps = ps)
end
tsim = TimeSimulation(ShaktiHydroModel(shakti_sim(3)))
ckpt_sh = joinpath(OUT, "tour_shakti_checkpoint.jld2")
run!(tsim; checkpoint_every = 1, checkpoint_path = ckpt_sh)
check(tsim.model.sim.total_time[] ≈ 3 * 3600 && all(isfinite, tsim.model.sim.state.N), "run!(TimeSimulation) ran Shakti's own loop for 3 steps")
check(isfile(ckpt_sh), "keyword arguments (checkpoint_every, checkpoint_path) pass through to Shakti.run!")
sA = shakti_sim(1); sB = shakti_sim(1); tA = TimeSimulation(ShaktiHydroModel(sA))
for k in 1:3
    sA.state.zs .+= 0.1; sB.state.zs .+= 0.1        # a coupled ice model moving the surface
    step!(tA)
    Shakti.step!(sB); sB.total_time[] += sB.dt[]
end
check(sA.total_time[] ≈ 3 * 3600 && sA.state.N == sB.state.N && sA.state.b == sB.state.b && sA.state.h == sB.state.h,
      "step!(TimeSimulation) = Shakti.step! + total_time += dt, bit for bit over 3 coupled steps")
covered!("run!", "step!")

# =============================================================================================
# PART 25 -- END TO END on real ice-sheet inputs
# =============================================================================================
# The pieces together, as a user would run them: load data, build the grid, state and model,
# solve, read back N, q, W, and use the coupling functions. Thwaites at 2 km with KORI-ULB's own
# basal shear stress as a prescribed friction law (the frictional heat KORI-ULB itself uses), then
# HAB on the same geometry; Antarctica at 16 km from a Yelmo restart, with the smoothing off
# (the coupling length is below one 16 km cell, as the constructor's warning explains).
part(25, "", "END TO END ON REAL DATA"); CURFILE[] = "PART 25 (end to end)"
if isfile(THW)
    sub("Thwaites 2 km, K24 (defaults: Warner routing, priority-flood filling, taped routing)")
    stT = HydroState(gt, maskt, ht, bt)
    mT = KazmierczakHydroModel(gt, kapt, ubt, At, Gt, qTt; sliding_law = PrescribedFieldSlidingLaw(gt, taubt), coupling_length_kamb86 = 5.0,
                               dissipation_verbose = false)
    tsolve = @elapsed run!(SteadyStateSimulation(mT, gt, stT))
    grT = maskt .== 1
    show1("first solve, including any compilation [s]", round(tsolve; digits = 2))
    nbad = (N = count(!isfinite, stT.N[grT]), Po0 = count(iszero, mT.Po[grT]), h0 = count(<=(0), ht[grT]),
            q = count(!isfinite, mT.q[grT]), W = count(!isfinite, stT.W[grT]))
    show1("grounded cells with non-finite N / Po = 0 / h <= 0 / bad q / bad W", nbad)
    okT = grT .& isfinite.(stT.N) .& (mT.Po .> 0)
    check(all(stT.N[grT .& (mT.Po .== 0)] .== 0), "the input's grounded cells with h = 0 have Po = 0 and get N = 0 (no NaN)")
    show1("N/Po (grounded, Po > 0): 5%, 50%, 95%", round.(quantile(stT.N[okT] ./ mT.Po[okT], (0.05, 0.5, 0.95)); digits = 3))
    show1("q [m2/yr] (grounded): 50%, 99%, max", round.(perSecond2perYear.((quantile(filter(isfinite, mT.q[grT]), 0.5), quantile(filter(isfinite, mT.q[grT]), 0.99), maximum(filter(isfinite, mT.q[grT])))); sigdigits = 3))
    show1("W [mm] (grounded): 50%, 99%", round.(1e3 .* quantile(filter(isfinite, stT.W[grT]), (0.5, 0.99)); sigdigits = 3))
    check(all(isfinite, stT.N) && all(isfinite, stT.W) && all(isfinite, mT.q), "Thwaites: N, W and q are finite everywhere")
    check(all(0 .<= stT.N[grT] .<= mT.Po[grT]) && all(>=(0), mT.q), "Thwaites: 0 <= N <= Po, q >= 0")
    melt_terms = (mT.mdot_fixed .+ mT.Q_b ./ mT.L_w)[grT]
    show1("melt from terms vs KORI-ULB melt: medians [mm/yr], correlation", (round(perSecond2perYear(median(melt_terms)); sigdigits = 3), round(perSecond2perYear(median(mel[grT])); sigdigits = 3), round(cor(melt_terms, mel[grT]); digits = 3)))
    visualize_field(gt, mask_field(stT.N ./ 1e6, maskt, NaN); plot_title = "Thwaites 2 km, K24: N [MPa]", display_flag = false,
                    savefig_path = joinpath(OUT, "tour_thwaites_N.png"), colormap = :viridis, transpose_data = false)
    Cfz = zeros(Nxt, Nyt); freeze_on_capacity!(Cfz, mT, gt, stT)
    show1("freeze-on capacity (grounded) median, max [m/yr ice]", round.(perSecond2perYear.((median(Cfz[grT]), maximum(Cfz[grT]))); sigdigits = 3))
    Nb = copy(stT.N); N_from_ub!(mT, gt, stT, 1.2 .* ubt)
    show1("N_from_ub! with u_b x 1.2: median change of N/Po", round(median((stT.N[okT] .- Nb[okT]) ./ mT.Po[okT]); sigdigits = 3))
    check(median(stT.N[grT] .- Nb[grT]) >= 0, "faster sliding raises N in the median cell (N_from_ub!)")
    sub("Thwaites 2 km, HAB")
    stH = HydroState(gt, maskt, ht, bt); run!(SteadyStateSimulation(HABHydroModel(gt), gt, stH))
    show1("HAB N/Po (grounded): 5%, 50%, 95%", round.(qfin(stH.N[grT] ./ (917 * 9.81 .* ht[grT]), (0.05, 0.5, 0.95)); digits = 3))
    check(all(isfinite, stH.N) && all(stH.N[grT] .>= 0.02 .* 917 .* 9.81 .* ht[grT] .- 1e-6), "HAB: finite, and N >= sigmat Po")
    # The two models share only the geometry: HAB has no routing, K24's N follows the routed flux. Their
    # correlation is a description of this input, not a property of the code, so it is shown, not checked.
    show1("correlation of HAB and K24 N (grounded)", round(cor(stH.N[grT], stT.N[grT]); digits = 3))
    sub("Thwaites 2 km, K24: where the time goes (one stage at a time, compiled, best of 3)")
    # Every stage is timed with tbest: one untimed call first, so a method this block calls but run!
    # does not is compiled before it is timed.
    t_phi = tbest(() -> FH.update_phi0!(mT, gt, stT))
    t_prep = tbest(() -> FH.prepare_routing!(mT, gt, stT, mT.routing_scheme))
    t_rec = tbest(() -> (FH.invalidate_routing_tape!(mT); FH.route_psi_out!(mT, gt, stT)))
    t_rep = tbest(() -> FH.route_psi_out!(mT, gt, stT))
    t_q = tbest(() -> FH.update_q_from_psi_out!(mT, gt, stT))
    t_N = tbest(() -> update_N!(mT, gt, stT))
    t_W = tbest(() -> FH.update_W!(mT, gt, stT))
    show1("phi0 / prepare_routing! (smooth, fill, gradients, w8) [ms]", round.(1e3 .* (t_phi, t_prep); digits = 1))
    show1("record tape + replay / replay only / q from psi_out [ms]", round.(1e3 .* (t_rec, t_rep, t_q); digits = 1))
    show1("update_N! ($(Threads.nthreads()) threads) / update_W! [ms]", round.(1e3 .* (t_N, t_W); digits = 1))
    show1("tape length (operations), grounded cells", (length(mT.routing_tape.w), count(grT)))
    _, lgT = with_logs(() -> FH.update_q!(KazmierczakHydroModel(gt, kapt, ubt, At, Gt, qTt; sliding_law = PrescribedFieldSlidingLaw(gt, taubt),
                                                                coupling_length_kamb86 = 5.0, dissipation_verbose = true), gt, HydroState(gt, maskt, ht, bt)))
    foreach(l -> println("    log: ", l.message), lgT)
    for (nm, alg) in (("GDSWarner, recursive", RecursivePsiOut()), ("GDSWarner, taped", TapedPsiOut()))
        mG = KazmierczakHydroModel(gt, kapt, ubt, At, Gt, qTt; routing_scheme = GDSWarner(), psi_out_algorithm = alg, coupling_length_kamb86 = 5.0,
                                   sliding_law = PrescribedFieldSlidingLaw(gt, taubt), dissipation_verbose = false)
        sG = HydroState(gt, maskt, ht, bt)
        tG = try
            round(1e3 * tbest(() -> run!(SteadyStateSimulation(mG, gt, sG))); digits = 1)
        catch e
            "failed: " * string(typeof(e))                                    # e.g. a StackOverflowError of the recursion
        end
        show1("full solve, $nm [ms]", tG)
    end
    tW = tbest(() -> run!(SteadyStateSimulation(mT, gt, stT)))
    show1("full solve, Warner default (compiled, warm start) [ms]", round(1e3 * tW; digits = 1))
else
    println("    (Thwaites input absent: Thwaites end-to-end runs skipped)")
end
if isfile(YX16)
    sub("Antarctica 16 km from a Yelmo restart, K24 without smoothing")
    stY = HydroState(gy, masky, hy, by)
    mY = KazmierczakHydroModel(gy, kapy, uby, Ay, Gy, qTy; sliding_law = PrescribedFieldSlidingLaw(gy, tauby), coupling_length_kamb86 = 0.0, dissipation_verbose = false)
    tY1 = @elapsed run!(SteadyStateSimulation(mY, gy, stY))
    tY = tbest(() -> run!(SteadyStateSimulation(mY, gy, stY)))
    show1("Yelmo 16 km solve: first call / compiled, best of 3 [ms]", round.(1e3 .* (tY1, tY); digits = 1))
    grY = masky .== 1
    show1("solve time [s]; N/Po (grounded) 5%, 50%, 95%", (round(tY; digits = 2), round.(qfin(stY.N[grY] ./ mY.Po[grY], (0.05, 0.5, 0.95)); digits = 3)))
    check(all(isfinite, stY.N) && all(0 .<= stY.N[grY] .<= mY.Po[grY]), "Antarctica 16 km: N finite and in [0, Po]")
    source_Y = sum(max.(mY.mdot_total[grY], 0)) * gy.dx * gy.dy / mY.rho_w
    show1("routed source (positive part) [m3/s]", round(source_Y; sigdigits = 4))
else
    println("    (Yelmo 16 km input absent: run skipped)")
end

# =============================================================================================
# COVERAGE: every definition of every source file versus what the tour exercised
# =============================================================================================
println("\n", "="^100, "\nCOVERAGE SUMMARY  (definitions parsed at run time from ", dirname(pathof(FastHydrology)), ")\n", "="^100)
const SRC_DIR = dirname(pathof(FastHydrology))
const EXT_DIR = joinpath(dirname(SRC_DIR), "ext")
const LOAD_ORDER = ["FastHydrology.jl", "common/grid.jl", "common/fft_convolution.jl", "common/model.jl", "common/state.jl", "common/simulation.jl",
                    "common/effective_pressure.jl", "common/run.jl", "common/checkpoint.jl", "common/output.jl", "common/utilities.jl", "common/plotting.jl",
                    "models/kazmierczak2024/model.jl", "models/kazmierczak2024/sliding_law.jl", "models/kazmierczak2024/water_flux.jl",
                    "models/kazmierczak2024/routing.jl", "models/kazmierczak2024/effective_pressure.jl", "models/kazmierczak2024/run.jl",
                    "models/kazmierczak2024/data_loaders.jl", "models/hab/model.jl", "models/hab/effective_pressure.jl", "models/hab/run.jl",
                    "models/shakti/model.jl"]
const EXT_FILES = ["FastHydrologyMakieExt.jl", "FastHydrologyShaktiExt.jl"]
# The include list of src/FastHydrology.jl must be LOAD_ORDER (minus the module file itself):
# if a file is added to the package, the tour notices.
const INCLUDED = [m.captures[1] for m in eachmatch(r"^include\(\"([^\"]+)\"\)"m, read(joinpath(SRC_DIR, "FastHydrology.jl"), String))]
CURFILE[] = "coverage"
check(INCLUDED == LOAD_ORDER[2:end], "the tour's PART order is the include order of src/FastHydrology.jl")
# A definition is: `function name` (including `function name end` stubs), a struct, an abstract
# type, a top-level const, or a top-level one-line method `name(args) = ...`. Docstrings and
# comments are skipped. Module prefixes of extended functions are dropped (Base.getproperty ->
# getproperty, FastHydrology.run! -> run!). One-line closures inside function bodies are indented
# and so not matched (`grounded` in fill_potential!).
const DEF_PATTERNS = [r"^\s*(?:@inline\s+|@noinline\s+)?function\s+([A-Za-z_][\w!.]*)",
                      r"^\s*(?:mutable\s+)?struct\s+([A-Za-z_]\w*)",
                      r"^\s*abstract\s+type\s+([A-Za-z_]\w*)",
                      r"^const\s+([A-Za-z_]\w*)",
                      r"^(?:@inline\s+|@noinline\s+)?([A-Za-z_][\w!.]*)\((?:[^()]|\([^()]*\))*\)(?:\s*where\s*[^=]*?)?\s*=(?!=)"]
function definitions(path)
    names = String[]
    indoc = false
    for line in eachline(path)
        nq = count("\"\"\"", line)
        if indoc
            isodd(nq) && (indoc = false)
            continue
        end
        if isodd(nq)
            indoc = true
            continue
        end
        startswith(lstrip(line), "#") && continue
        for re in DEF_PATTERNS
            m = match(re, line)
            if m !== nothing
                nm = String(last(split(m.captures[1], '.')))
                nm in names || push!(names, nm)
                break
            end
        end
    end
    return names
end
let ndef = 0, ntouch = 0, nskip = 0, missing_all = String[]
    for (k, f) in enumerate(vcat(LOAD_ORDER, EXT_FILES))
        path = f in EXT_FILES ? joinpath(EXT_DIR, f) : joinpath(SRC_DIR, f)
        defs = definitions(path)
        touched = filter(n -> (f * ":" * n) in TOUCHED, defs)
        skips = filter(n -> haskey(SKIPPED, f * ":" * n) && !((f * ":" * n) in TOUCHED), defs)
        miss = filter(n -> !(n in touched) && !(n in skips), defs)
        ndef += length(defs); ntouch += length(touched); nskip += length(skips)
        append!(missing_all, f .* ":" .* miss)
        println("  ", rpad("PART " * string(k - 1), 8), rpad(f, 46), lpad(length(defs), 3), " defined ", lpad(length(touched), 3), " exercised",
                isempty(skips) ? "" : "  skipped: " * join(skips, ", "), isempty(miss) ? "" : "  MISSING: " * join(miss, ", "))
    end
    println("\n  total: $ndef definitions, $ntouch exercised, $nskip skipped, $(length(missing_all)) missing")
    for (k, v) in sort(collect(SKIPPED))
        (k in TOUCHED) || println("  skipped  ", rpad(k, 60), v)
    end
    isempty(missing_all) || push!(FAILED, "coverage: missing " * join(missing_all, ", "))
end

println("\n", "="^100)
println("CODE TOUR FINISHED in ", round(time() - TOUR_T0; digits = 1), " s: ", NCHECK[], " checks, ", length(FAILED), " failed")
foreach(f -> println("  FAILED: ", f), FAILED)
println("="^100)
isempty(FAILED) || exit(1)
