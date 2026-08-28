module FastHydrology

using Statistics: mean
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using ImageFiltering
using FFTW
using OffsetArrays
using DocStringExtensions
using MAT
using NCDatasets

include("grid.jl")
include("operations.jl")
include("fft_convolution.jl")
include("model.jl")
include("sliding_law.jl")
include("state.jl")
include("simulation.jl")
include("run.jl")
include("water_flux.jl")
include("effective_pressure.jl")
include("data_loaders.jl")
include("utilities.jl")
include("plotting.jl")

# grid.jl
export AbstractHydroGrid, OGRectHydroGrid, ArrayHydroGrid
export fill_halo!, alloc_field
export convolve!, masked_mean, masked_max_abs, masked_max_abs_diff, overwrite_where!
export minus_gradient_x!, minus_gradient_y!

# model.jl
export AbstractHydroModel, KazmierczakHydroModel, HABHydroModel, ShaktiHydroModel
export AbstractSlidingLaw, NoSlidingLaw, WeertmanSlidingLaw, PowerPlasticSlidingLaw, RegularizedCoulombSlidingLaw
export AbstractPsiOutAlgorithm, RecursivePsiOut, IterativePsiOut, TopologicalPsiOut
export AbstractWaterThicknessAlgorithm, ArealConduitThickness, DarcyWeisbachThickness, LaminarThickness
export AbstractGradientConvention, LocalGradient, MeanGradient

# sliding_law.jl
export calc_tau_b, update_tau_b!

# state.jl
export AbstractHydroState, HydroState

# simulations.jl
export AbstractSimulation, TimeSimulation, SteadyStateSimulation

# run.jl
export run!, step!, update_steady_state!

# water_flux.jl
export update_q!, update_W!
export update_phi0!, potential_filling!
export update_potential_gradients!, update_smoothed_potential_gradients!
export accumulate_psi_out!, update_psi_out!, update_psi_out_iterative!, route_psi_out!

# effective_pressure.jl
export update_N!, update_Po!, update_p_w!
export update_H!, update_S_inf!, update_N_inf!, update_Q!

# data_loaders
export load_Kazmierczak, load_yelmox

# utilities.jl
export compute_lims, perYear2perSecond, perSecond2perYear, Km2m

# plotting.jl
export visualize_grid, visualize_field, mask_field

end