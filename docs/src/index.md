# FastHydrology.jl

Glacial hydrology models for ice-sheet modelling. Each model diagnoses the effective pressure `N` at
the ice base. Models:

* **K24**: steady state model of [Kazmierczak et al. (2024)](https://doi.org/10.5194/tc-18-5887-2024), the main one
* **HAB**: height above buoyancy, [Kazmierczak et al. (2022)](https://doi.org/10.5194/tc-16-4537-2022)
* **Shakti**: wrapper around [Shakti.jl](https://github.com/TakisAngelides/Shakti.jl), transient, [Sommers et al. (2018)](https://doi.org/10.5194/gmd-11-2955-2018)

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/TakisAngelides/FastHydrology.jl")
```

## Quick start

Every run has the same steps: grid, state, model, simulation, run.

```julia
using FastHydrology

Nx, Ny = 50, 50
grid = ArrayHydroGrid(Nx, Ny, (0.0, 50_000.0), (0.0, 50_000.0))

mask    = fill(GROUNDED, Nx, Ny)                        # grounded ice, solved
mask[end, :] .= OCEAN                                   # outlet: water leaves here (the other edges are closed)
h       = [2000.0 - 15.0 * i for i in 1:Nx, j in 1:Ny]  # ice thickness [m]
b       = [-200.0 - 2.0 * j for i in 1:Nx, j in 1:Ny]   # bed elevation [m]
kappa   = zeros(Nx, Ny)                                 # 0 = hard bed
abs_v_b = fill(perYear2perSecond(100.0), Nx, Ny)        # sliding speed [m/s]
A_visc  = fill(1e-24, Nx, Ny)                           # ice flow rate factor
G       = fill(1e-6 * 3.34e5, Nx, Ny)                   # geothermal flux [W/m2]
q_T     = zeros(Nx, Ny)                                 # conductive flux into the ice [W/m2]

state = HydroState(grid, HydroMask(mask), h, b)
model = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, G, q_T; coupling_length_kamb86 = 10.0)
run!(SteadyStateSimulation(model, grid, state))

state.N   # effective pressure [Pa]
state.W   # water thickness [m]
model.q   # water flux [m2/s]
```

## Where next

* [Tutorial](@ref): every option, from input data to output
* Examples: K24 and HAB on [Antarctica](@ref), [Greenland](@ref), [Thwaites](@ref), [Helheim](@ref), [Drang Drung](@ref), plus a small [Shakti](@ref Shakti-example) run
* [API Reference](@ref)
