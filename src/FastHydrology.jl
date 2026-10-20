module FastHydrology

using Statistics: mean
using ImageFiltering
using SpecialFunctions: erf
using FFTW
using OffsetArrays
using DocStringExtensions
using MAT
using NCDatasets

include("common/grid.jl")
include("common/fft_convolution.jl")
include("common/model.jl")
include("common/mask.jl")
include("common/state.jl")
include("common/simulation.jl")
include("common/effective_pressure.jl")
include("common/run.jl")
include("common/checkpoint.jl")
include("common/output.jl")
include("common/utilities.jl")
include("common/plotting.jl")

# The option files (drainage.jl to water_thickness.jl) come before model.jl, whose KazmierczakParams
# stores their types; their methods take an AbstractKazmierczakHydroModel (abstract_model.jl).
include("models/kazmierczak2024/abstract_model.jl")
include("models/kazmierczak2024/drainage.jl")
include("models/kazmierczak2024/sliding_law.jl")
include("models/kazmierczak2024/melt_sources.jl")
include("models/kazmierczak2024/potential.jl")
include("models/kazmierczak2024/psi_out.jl")
include("models/kazmierczak2024/routing.jl")
include("models/kazmierczak2024/water_thickness.jl")
include("models/kazmierczak2024/model.jl")
include("models/kazmierczak2024/water_flux.jl")
include("models/kazmierczak2024/effective_pressure.jl")
include("models/kazmierczak2024/run.jl")
include("models/kazmierczak2024/data_loaders.jl")

include("models/hab/model.jl")
include("models/hab/effective_pressure.jl")
include("models/hab/run.jl")

include("models/shakti/model.jl")

# common/grid.jl
export AbstractHydroGrid, ArrayHydroGrid
export alloc_field
export convolve!, masked_mean, masked_max_abs, masked_max_abs_diff, overwrite_where!
export minus_gradient_x!, minus_gradient_y!, face_gradient_magnitude!

# common/mask.jl
export GROUNDED, OCEAN, LAND, OTHER_BASIN, HydroMask, classify_mask

# common/model.jl, models/kazmierczak2024/model.jl, models/hab/model.jl, models/shakti/model.jl
export AbstractHydroModel, KazmierczakHydroModel, HABHydroModel, ShaktiHydroModel

# models/kazmierczak2024/drainage.jl
export AbstractDrainageMode, BothDrainage, EfficientOnly, InefficientOnly

# models/kazmierczak2024/sliding_law.jl
export AbstractSlidingLaw, NoFrictionSlidingLaw, PrescribedFieldSlidingLaw, WeertmanSlidingLaw, PowerPlasticSlidingLaw, RegularizedCoulombSlidingLaw, RegularizedCoulombFieldSlidingLaw, ShaktiRegularizedCoulombSlidingLaw
export calc_tau_b, update_tau_b!

# models/kazmierczak2024/melt_sources.jl
export set_basal_terms!
export AbstractDissipationDiscretization, CellCentredDissipation, FaceDissipation
export AbstractFrictionDiscretization, CellCentredFriction, StaggeredFriction

# models/kazmierczak2024/potential.jl
export update_phi0!, potential_filling!
export AbstractFillAlgorithm, JacobiFill, LowestNeighbourFill, PriorityFloodFill
export AbstractGradientDiscretization, CentredGradient, FaceGradient
export AbstractPotentialWeight, GroundedIce, GroundedMask
export update_potential_gradients!, update_smoothed_potential_gradients!

# models/kazmierczak2024/psi_out.jl
export AbstractPsiOutAlgorithm, RecursivePsiOut, IterativePsiOut, TopologicalPsiOut, TapedPsiOut
export accumulate_psi_out!, update_psi_out!, update_psi_out_iterative!, update_psi_out_taped!, route_psi_out!, invalidate_routing_tape!

# models/kazmierczak2024/routing.jl
export AbstractRoutingScheme, GDSWarner, Warner, Quinn, Tarboton, ModifiedTarboton, GDSTarboton
export AbstractQConversion, QFromOutflow, QFromFaceAverage
export freeze_on_capacity!

# models/kazmierczak2024/water_thickness.jl
export AbstractWaterThicknessAlgorithm, ArealConduitThickness, DarcyWeisbachThickness, LaminarThickness
export AbstractGradientConvention, LocalGradient, MeanGradient
export update_W!

# common/state.jl
export AbstractHydroState, HydroState

# common/simulation.jl
export AbstractSimulation, TimeSimulation, SteadyStateSimulation

# common/run.jl, models/kazmierczak2024/run.jl, models/hab/run.jl
export run!, step!, update_steady_state!

# models/kazmierczak2024/water_flux.jl
export update_q!

# common/effective_pressure.jl, models/kazmierczak2024/effective_pressure.jl, models/hab/effective_pressure.jl
export update_N!, update_Po!, update_p_w!
export N_responds_to_ub, N_from_ub!
export update_H!, update_S_inf!, update_N_inf!, update_Q!

# models/kazmierczak2024/data_loaders.jl
export load_Kazmierczak, load_yelmox, load_Kazmierczak_melt

# common/utilities.jl
export compute_lims, perYear2perSecond, perSecond2perYear, Km2m

# common/plotting.jl
export visualize_grid, visualize_field, mask_field

# common/checkpoint.jl
export save_checkpoint, load_checkpoint!

# common/output.jl
export AbstractOutputWriter, NetCDFOutputWriter, write_output!, close_output!

end
