```@meta
EditURL = "Synthetic.jl"
```

# [Synthetic ice sheet](@id Synthetic)
This example runs [`KazmierczakHydroModel`](@ref) end to end on a small synthetic ice-sheet geometry,
so unlike the [Kazmierczak et al 2024](@ref Kazmierczak2024) example (which needs the Thwaites
dataset) it executes live as part of the docs build.

````@example Synthetic
using FastHydrology
using CairoMakie

T = Float64
Nx, Ny = 50, 50
xlims, ylims = (0.0, 50_000.0), (0.0, 50_000.0) # 50 km x 50 km domain
````

Build the grid.

````@example Synthetic
grid = ArrayHydroGrid(Nx, Ny, xlims, ylims; T = T)
````

A synthetic sloped ice-sheet geometry: thickness decreasing and bed deepening from left to
right, uniform basal conditions, everywhere grounded.

````@example Synthetic
mask    = ones(T, Nx, Ny)
h       = [2000.0 - 15.0 * i for i in 1:Nx, j in 1:Ny]
b       = [-200.0 - 2.0 * j for i in 1:Nx, j in 1:Ny]
kappa   = zeros(T, Nx, Ny)                            # hard bed everywhere
abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), Nx, Ny)  # 100 m/a basal sliding speed
A_visc  = fill(1e-24, Nx, Ny)
G       = fill(1e-6 * 3.34e5, Nx, Ny)                 # geothermal heat [W m⁻²]: melts 1e-6 kg m⁻² s⁻¹
q_T     = zeros(Nx, Ny)                               # conductive heat into the ice [W m⁻²] (temperate bed)
````

`coupling_length_kamb86` is Kamb & Echelmeyer (1986)'s stress-gradient-coupling length, passed
directly as a multiple of ice thickness (~4-10 for ice sheets, ~1-3 for valley/mountain glaciers --
see the constructor's docstring). It has no safe default that works for every grid resolution, so it
must be passed explicitly or `KazmierczakHydroModel` warns and falls back to 10.0 (the upper edge of
the ice-sheet range). 10.0 is well resolved at this grid's 1 km spacing.

````@example Synthetic
model = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, G, q_T; coupling_length_kamb86 = 10.0, dissipation_verbose = false)
state = HydroState(grid, mask, h, b)

sim = SteadyStateSimulation(model, grid, state)
run!(sim)

N_plot = mask_field(state.N .* 1e-6, mask, NaN) # makes N [MPa]
fig_N = visualize_field(grid, N_plot; plot_title = "Effective pressure N [MPa]", colorrange = (0, 10))
````

