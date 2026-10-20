# FastHydrology.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://TakisAngelides.github.io/FastHydrology.jl/dev/)
[![](https://img.shields.io/badge/license-GNU_GPL_3.0-green.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)
[![][ci-img]][ci-url]

[ci-img]: https://github.com/TakisAngelides/FastHydrology.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/TakisAngelides/FastHydrology.jl/actions

FastHydrology.jl is a Julia package of glacial hydrology models for ice-sheet modelling. Each model
diagnoses the effective pressure at the ice base, the quantity that couples subglacial water to
basal sliding. It currently includes:

- **K24**: the steady-state hydrology model of
  [Kazmierczak et al. (2024)](https://doi.org/10.5194/tc-18-5887-2024). It is the main model of this
  repository.
- **HAB**: the height-above-buoyancy parameterization, which assumes a direct
  hydraulic connection to the ocean at the grounding line
  (Sec. 2.1.1 of [Kazmierczak et al. 2022](https://doi.org/10.5194/tc-16-4537-2022)).
- **Shakti**: a wrapper around [Shakti.jl](https://github.com/TakisAngelides/Shakti.jl),
  the Julia implementation of the transient Shakti model of
  [Sommers et al. (2018)](https://doi.org/10.5194/gmd-11-2955-2018).

## Aim

The aim is to collect glacial hydrology models that can be coupled to an ice-flow model such as
[Yelmo](https://github.com/palma-ice/yelmo)
([Robinson et al. 2020](https://doi.org/10.5194/gmd-13-2805-2020)). The name of the repository is
indicative that we strive to keep the models included here fast, even for the transient model case of Shakti,
which is implemented in [Shakti.jl](https://github.com/TakisAngelides/Shakti.jl).

## The K24 model

K24 is a steady-state model. It takes the state of a glacier or ice sheet as input, calculates the
subglacial water flux with a routing algorithm, and diagnoses the effective pressure, defined as the
ice overburden pressure minus the water pressure.

This implementation of K24 includes all options for the routing algorithm that calculates the water flux that are discussed in [Le Brocq et al. (2006)](https://doi.org/10.1016/j.cageo.2006.05.003).

Because the model is steady state, the effective pressure is recomputed from the current geometry
and melt rate at each call, with no time stepping. This makes it cheap enough to couple to
multi-century ice-flow simulations. The code is also optimized such that one full solve on a real topography
takes tens of milliseconds when allowing for a feedback between melt rate and water flux, which
requires a Picard iteration, otherwise it is even faster. Some examples which include a Picard iteration are shown in the table below.

| Region | Grid (`nx x ny`) | Grounded cells | Time per solve |
|---|---|---|---|
| Thwaites Glacier, 2 km | 345 x 288 | 51,945 | 22 ms |
| Greenland Ice Sheet, 8 km | 211 x 361 | 27,492 | 12 ms |
| Antarctic Ice Sheet, 16 km | 381 x 381 | 47,886 | 23 ms |

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/TakisAngelides/FastHydrology.jl")
```

## Quick start

```julia
using FastHydrology

# A 50 km x 50 km synthetic ice sheet on a 50 x 50 grid, draining into the ocean along one edge.
Nx, Ny = 50, 50
grid = ArrayHydroGrid(Nx, Ny, (0.0, 50_000.0), (0.0, 50_000.0))

mask    = fill(GROUNDED, Nx, Ny)                      # grounded ice, solved
mask[end, :] .= OCEAN                                 # outlet: water leaves here (the other edges are closed)
h       = [2000.0 - 15.0 * i for i in 1:Nx, j in 1:Ny] # ice thickness [m]
b       = [-200.0 - 2.0 * j for i in 1:Nx, j in 1:Ny]  # bedrock elevation [m]

kappa   = zeros(Nx, Ny)                               # bed type: 0 = hard bed
abs_v_b = fill(100.0 / (60^2 * 24 * 365.25), Nx, Ny)  # basal sliding speed [m/s] (100 m/a)
A_visc  = fill(1e-24, Nx, Ny)                         # ice flow-law rate factor
G       = fill(1e-6 * 3.34e5, Nx, Ny)                 # geothermal heat [W/m^2]
q_T     = zeros(Nx, Ny)                               # conductive heat into the ice [W/m^2]

# Build the model and the state, then run the steady-state solve.
model = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, G, q_T; coupling_length_kamb86 = 10.0)
state = HydroState(grid, HydroMask(mask), h, b)
run!(SteadyStateSimulation(model, grid, state))

# The effective pressure N [Pa] and water layer thickness W [m] are now in `state`,
# and the water flux q in `model.q`.
state.N
```

For more runnable examples see the [online documentation](https://TakisAngelides.github.io/FastHydrology.jl/dev/).

## License

GNU General Public License v3.0 -- see [LICENSE](LICENSE).
