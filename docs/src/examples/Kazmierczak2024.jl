#=
# [Kazmierczak et al 2024](@id Kazmierczak2024)
This is an example of how to run the FastHydrology.jl package for the steady state problem of Kazmierczak et al 2024.
=#

using FastHydrology
using Statistics
using FileIO
using CairoMakie

# How to load data from a file to get the necessary inputs to initialize the grid, model and state for the Kazmierczak et al 2024 paper.

T = Float64 # Type of the physical fields
path = "$(@__DIR__)/input/Kazmierczak2024/THWAITES2km_m3_HAB_toto.mat" # Path to data for Fig. S3 of Kazmierczak et al 2024 which focuses on Thwaites glacier.
Nx, Ny, xlims, ylims, mask, h, b, abs_v_b, A_visc, ṁ, κ = load_Kazmierczak(path; bed_rheology = :hard)

# Here we prepare a grid using the Oceananigans rectilinear grid.

grid = OGRectHydroGrid(Nx, Ny, xlims, ylims; T = T)

# There is a function to visualize the Oceananigans rectilinear grid.

fig = visualize_grid(grid) 

# We then build the model using the data from the input file above. The model will hold its model-specific fields, with the rest of the fields common to all models stored in the HydroState below.

model = KazmierczakHydroModel(grid, κ, abs_v_b, A_visc, ṁ);

# Further, we build the hydrology state using some of the data from the input file. This state will store the fields common to all hydrology models.

state = HydroState(grid, mask, h, b);

# Finally we can create a simulation struct. Since the Kazmierczak et al 2024 model is a steady state problem we create a SteadyStateSimulation rather than a TimeSimulation.

sim = SteadyStateSimulation(model, grid, state);

# We can now run the simulation which will update the hydrology state, as well as the model-dependent fields defined in the model.

run!(sim)

# Applying some rescaling for plotting.

model.q .*= perSecond2perYear(1.0) / 1e4 # makes q [10⁴ m² a⁻¹]
state.N .*= 1e-6 # makes N [MPa]
model.q .= mask_field(model.q, state.mask, NaN)
state.N .= mask_field(state.N, state.mask, NaN)
state.W .= mask_field(state.W, state.mask, NaN)
display_flag = false

# Now we can visualize the resulting water flux q [10⁴ m² a⁻¹].

fig_q = visualize_field(model.q; plot_title = "Distributed water flux q [10⁴ m² a⁻¹]", transpose_data = true, display_flag = display_flag, colorrange = (0, 10))

# The water layer thickness W [m].

fig_W = visualize_field(state.W; plot_title = "Water thickness W [m]", transpose_data = true, display_flag = display_flag, colorrange = extrema(filter(!isnan, state.W.data)))

# The effective pressure N [MPa].

fig_N = visualize_field(state.N; plot_title = "Effective pressure N [MPa]", transpose_data = true, display_flag = display_flag, colorrange = (0, 10))