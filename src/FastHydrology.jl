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

include("common/grid.jl")
include("common/operations.jl")
include("common/fft_convolution.jl")
include("common/model.jl")
include("common/state.jl")
include("common/simulation.jl")
include("common/effective_pressure.jl")
include("common/run.jl")
include("common/checkpoint.jl")
include("common/output.jl")
include("common/utilities.jl")
include("common/plotting.jl")

include("models/kazmierczak2024/model.jl")
include("models/kazmierczak2024/sliding_law.jl")
include("models/kazmierczak2024/water_flux.jl")
include("models/kazmierczak2024/effective_pressure.jl")
include("models/kazmierczak2024/run.jl")
include("models/kazmierczak2024/data_loaders.jl")

include("model.jl")
include("effective_pressure.jl")
include("run.jl")

# grid.jl
export AbstractHydroGrid, OGRectHydroGrid, ArrayHydroGrid
export fill_halo!, alloc_field
export convolve!, masked_mean, masked_max_abs, masked_max_abs_diff, overwrite_where!
export minus_gradient_x!, minus_gradient_y!

# model.jl
export AbstractHydroModel, KazmierczakHydroModel, HABHydroModel, ShaktiHydroModel
export AbstractDrainageMode, BothDrainage, EfficientOnly, InefficientOnly
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

# checkpoint.jl
export save_checkpoint, load_checkpoint!

# output.jl
export AbstractOutputWriter, NetCDFOutputWriter, write_output!, close_output!

end
