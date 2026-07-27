# FastHydrology.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://TakisAngelides.github.io/FastHydrology.jl/dev/)
[![](https://img.shields.io/badge/license-GNU_GPL_3.0-green.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)
[![][ci-img]][ci-url]

[ci-img]: https://github.com/TakisAngelides/FastHydrology.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/TakisAngelides/FastHydrology.jl/actions

A general Julia framework for simulating subglacial hydrology beneath ice sheets. FastHydrology
provides shared grid, state, and simulation infrastructure that different subglacial hydrology
models plug into, computing the effective pressure at the base of an ice sheet -- the quantity
that couples subglacial water to basal sliding. It can be run standalone or coupled into a larger
ice-sheet model such as yelmox.

## Models

Two models are implemented so far, sharing the same grid/state/simulation infrastructure:

- **`KazmierczakHydroModel`** -- the fast, simplified subglacial hydrology model of
  [Kazmierczak et al. 2024](https://doi.org/10.5194/tc-18-5887-2024). It routes the distributed
  subglacial water flux over the geometric potential using a recursive algorithm (following
  Le Brocq et al. 2009, *A subglacial water-flow model for West Antarctica*, Journal of
  Glaciology 55(193):879-888), matches that flux to a local conduit scale, and parameterizes the
  effective pressure for both hard and soft beds with an automatic switch between efficient
  (channelized) and inefficient (distributed) drainage.
- **`HABHydroModel`** -- the simpler "height above buoyancy" parameterization from
  Sec. 2.1.1 of [Kazmierczak et al. 2022](https://doi.org/10.5194/tc-16-4537-2022), which assumes
  a direct hydraulic connection to the ocean at the grounding line.

Both are steady-state models: subglacial hydrology is assumed to equilibrate on a timescale much
shorter than ice-sheet evolution, so effective pressure is recomputed from the current ice
geometry and melt rate at each call rather than time-stepped.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/TakisAngelides/FastHydrology.jl")
```

## Quick start

```julia
using FastHydrology

# 1. Build a rectilinear grid.
grid = OGRectHydroGrid(Nx, Ny, xlims, ylims)

# 2. Build a model, providing the model-specific input fields.
model = KazmierczakHydroModel(grid, kappa, abs_v_b, A_visc, mdot)
# or: model = HABHydroModel(grid)

# 3. Build the state, holding fields common to every model (grounded-ice mask, ice thickness,
#    bedrock elevation, and the outputs: effective pressure N and water layer thickness W).
state = HydroState(grid, mask, h, b)

# 4. Run the steady-state solve.
sim = SteadyStateSimulation(model, grid, state)
run!(sim)

# state.N and state.W (and, for KazmierczakHydroModel, model.q) now hold the solution.
```

See [`docs/src/examples`](docs/src/examples) for complete, runnable examples against real
Thwaites Glacier and whole-Antarctica input data, or the
[online documentation](https://TakisAngelides.github.io/FastHydrology.jl/dev/) for the rendered
versions.

## Package structure

The package is organized around four abstractions:

| Abstraction | Purpose | Concrete implementation(s) provided |
|---|---|---|
| `AbstractHydroGrid` | Grid geometry and the backend-specific glue (field allocation, halo filling, and a few array operations physics code needs without knowing the backend) | `OGRectHydroGrid` (wraps an `Oceananigans.RectilinearGrid`) -- any grid backend can implement this interface |
| `AbstractHydroModel` | Model-specific constants and fields | `KazmierczakHydroModel`, `HABHydroModel` |
| `AbstractHydroState` | Fields common to every model: mask, ice thickness, bedrock elevation (inputs), effective pressure and water thickness (outputs) | `HydroState` |
| `AbstractSimulation` | How to run a model | `SteadyStateSimulation` |

Physics code (`water_flux.jl`, `effective_pressure.jl`) is written against the grid interface --
`grid.Nx`, `grid.dx`, `alloc_field`, `fill_halo!`, `convolve!`, `masked_mean`, `overwrite_where!`
-- rather than reaching into Oceananigans-specific internals directly, so a different grid backend
would only need to implement that interface.

### Source layout

- `grid.jl` -- the grid abstraction and its `OGRectHydroGrid` implementation.
- `operations.jl` -- registers `min`, `max`, `erf` as broadcastable Oceananigans field operations.
- `fft_convolution.jl` -- a cached, plan-reusing FFT convolution (used by the stress-gradient
  coupling smoothing in `water_flux.jl`) that avoids `ImageFiltering.jl`'s per-call allocation.
- `model.jl` -- the two model structs and their constructors.
- `state.jl` -- `HydroState`.
- `simulation.jl`, `run.jl` -- the simulation abstraction and `run!`/`update_steady_state!`.
- `water_flux.jl` -- geometric potential, flux routing, and water layer thickness
  (`KazmierczakHydroModel` only).
- `effective_pressure.jl` -- effective pressure for both models.
- `data_loaders.jl` -- `load_Kazmierczak` and `load_yelmox`, for reading `.mat`/`.nc` input data
  into the arrays the constructors above expect.
- `utilities.jl`, `plotting.jl` -- unit conversions and visualization helpers.

## Testing

```julia
using Pkg
Pkg.test("FastHydrology")
```

The test suite exercises both models end-to-end, both data loaders across all bed-rheology
options, and regression-tests specific bugs found during development.

## License

GNU General Public License v3.0 -- see [LICENSE](LICENSE).
