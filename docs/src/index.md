# FastHydrology.jl

A general Julia framework for simulating subglacial hydrology beneath ice sheets. FastHydrology.jl
provides shared grid, state, and simulation infrastructure that different subglacial hydrology
models plug into, computing the effective pressure at the base of an ice sheet -- the quantity
that couples subglacial water to basal sliding. It can be run standalone or coupled into a larger
ice-sheet model such as [Yelmo](https://github.com/palma-ice/yelmo)/yelmox.

## Models

Two models are implemented so far, sharing the same grid/state/simulation infrastructure:

- [`KazmierczakHydroModel`](@ref) -- the fast, simplified subglacial hydrology model of
  [Kazmierczak et al. 2024](https://doi.org/10.5194/tc-18-5887-2024). It routes the distributed
  subglacial water flux over the geometric potential using a recursive algorithm (following
  Le Brocq et al. 2009, *A subglacial water-flow model for West Antarctica*, Journal of
  Glaciology 55(193):879-888), matches that flux to a local conduit scale, and parameterizes the
  effective pressure for both hard and soft beds with an automatic switch between efficient
  (channelized) and inefficient (distributed) drainage.
- [`HABHydroModel`](@ref) -- the simpler "height above buoyancy" parameterization from
  Sec. 2.1.1 of [Kazmierczak et al. 2022](https://doi.org/10.5194/tc-16-4537-2022), which assumes
  a direct hydraulic connection to the ocean at the grounding line.

Both are steady-state models: subglacial hydrology is assumed to equilibrate on a timescale much
shorter than ice-sheet evolution, so effective pressure is recomputed from the current ice
geometry and melt rate at each call rather than time-stepped.

A third, genuinely time-evolving model is available as an optional extension:

- [`ShaktiHydroModel`](@ref) -- wraps the separate
  [Shakti.jl](https://github.com/TakisAngelides/Shakti.jl) package, a solver for the SHAKTI
  subglacial hydrology model (Sommers et al. 2018, https://gmd.copernicus.org/articles/11/2955/2018/), which
  transitions smoothly between distributed and channelized drainage rather than treating them as
  separate regimes. FastHydrology just wraps a `Shakti.Simulation` and delegates to it; see
  [Coupling to Shakti.jl](@ref ShaktiCoupling) for a runnable example. Since Shakti pulls in a much
  heavier set of dependencies (CUDA, Metal, AlgebraicMultigrid, ...), it's loaded via a Julia
  package extension (`FastHydrologyShaktiExt`) rather than as a hard dependency --
  [`ShaktiHydroModel`](@ref) only becomes usable once you `using Shakti` alongside
  `using FastHydrology`.

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

See the full [API Reference](@ref) for every exported type and function, including their
arguments, units, and governing equations.

## Examples

Two complete, runnable examples against real Thwaites Glacier and whole-Antarctica input data are
included, plus a synthetic example of the `ShaktiHydroModel` extension:

- [Kazmierczak et al 2024](@ref Kazmierczak2024)
- [Height above buoyancy (HAB)](@ref HAB)
- [Coupling to Shakti.jl](@ref ShaktiCoupling)

## Package structure

The package is organized around four abstractions:

| Abstraction | Purpose | Concrete implementation(s) provided |
|---|---|---|
| [`AbstractHydroGrid`](@ref) | Grid geometry and the backend-specific glue (field allocation, halo filling, and a few array operations physics code needs without knowing the backend) | [`OGRectHydroGrid`](@ref) (wraps an `Oceananigans.RectilinearGrid`) -- any grid backend can implement this interface |
| [`AbstractHydroModel`](@ref) | Model-specific constants and fields | [`KazmierczakHydroModel`](@ref), [`HABHydroModel`](@ref), [`ShaktiHydroModel`](@ref) |
| [`AbstractHydroState`](@ref) | Fields common to every model: mask, ice thickness, bedrock elevation (inputs), effective pressure and water thickness (outputs) | [`HydroState`](@ref) (not used by [`ShaktiHydroModel`](@ref), which brings its own state) |
| [`AbstractSimulation`](@ref) | How to run a model | [`SteadyStateSimulation`](@ref), [`TimeSimulation`](@ref) |

Physics code (`water_flux.jl`, `effective_pressure.jl`) is written against the grid interface --
`grid.Nx`, `grid.dx`, `alloc_field`, [`fill_halo!`](@ref), [`convolve!`](@ref),
[`masked_mean`](@ref), [`overwrite_where!`](@ref) -- rather than reaching into
Oceananigans-specific internals directly, so a different grid backend would only need to implement
that interface.

### Source layout

- `grid.jl` -- the grid abstraction and its `OGRectHydroGrid` implementation.
- `operations.jl` -- registers `min`, `max`, `erf` as broadcastable Oceananigans field operations.
- `fft_convolution.jl` -- a cached, plan-reusing FFT convolution (used by the stress-gradient
  coupling smoothing in `water_flux.jl`) that avoids ImageFiltering.jl's per-call allocation.
- `model.jl` -- the model structs and their constructors, including [`ShaktiHydroModel`](@ref).
- `state.jl` -- `HydroState`.
- `simulation.jl`, `run.jl` -- the simulation abstraction and `run!`/`step!`/`update_steady_state!`
  ([`step!`](@ref) is a stub with no methods in core -- see below).
- `water_flux.jl` -- geometric potential, flux routing, and water layer thickness
  (`KazmierczakHydroModel` only).
- `effective_pressure.jl` -- effective pressure for both models.
- `data_loaders.jl` -- `load_Kazmierczak` and `load_yelmox`, for reading `.mat`/`.nc` input data
  into the arrays the constructors above expect.
- `utilities.jl`, `plotting.jl` -- unit conversions and visualization helpers.
- `../ext/FastHydrologyShaktiExt.jl` -- package extension adding the `run!`/`step!` methods that
  make [`ShaktiHydroModel`](@ref) actually runnable, active only once `Shakti` is loaded alongside
  `FastHydrology`.

## Testing

```julia
using Pkg
Pkg.test("FastHydrology")
```

The test suite exercises both steady-state models end-to-end, both data loaders across all
bed-rheology options, regression-tests specific bugs found during development, and (in its own
`module`, in `test/shakti_ext_test.jl`) [`ShaktiHydroModel`](@ref)'s `run!`/`step!` extension
methods against a real `Shakti.Simulation`.

## License

GNU General Public License v3.0 -- see
[LICENSE](https://github.com/TakisAngelides/FastHydrology.jl/blob/main/LICENSE).
