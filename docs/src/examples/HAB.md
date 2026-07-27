```@meta
EditURL = "HAB.jl"
```

# [Height above buoyancy (HAB)](@id HAB)
This is an example of how to run the FastHydrology.jl package for the steady state problem of the HAB model which is described in Sec. 2.1.1 of Kazmierczak et al 2022 (https://doi.org/10.5194/tc-16-4537-2022).

````@example HAB
using FastHydrology
using FileIO
using CairoMakie
````

How to load data from a file to get the necessary inputs to initialize the grid, model and state for the Kazmierczak et al 2024 paper.

````@example HAB
T = Float64 # Type of the physical fields
path = "$(@__DIR__)/input/Kazmierczak2024/THWAITES2km_m3_HAB_toto.mat" # Path to data for Fig. S3 of Kazmierczak et al 2024 which focuses on Thwaites glacier.
Nx, Ny, xlims, ylims, mask, h, b, abs_v_b, A_visc, ṁ, κ = load_Kazmierczak(path)
````

Here we prepare a grid using the Oceananigans rectilinear grid.

````@example HAB
grid = OGRectHydroGrid(Nx, Ny, xlims, ylims; T = T)
````

There is a function to visualize the Oceananigans rectilinear grid.

````@example HAB
fig = visualize_grid(grid)
````

We then build the model using the data from the input file above. The model will hold its model-specific constants and fields, with the rest of the fields common to all models stored in the HydroState below.

````@example HAB
model = HABHydroModel(grid);
nothing #hide
````

Further, we build the hydrology state using some of the data from the input file. This state will store the fields common to all hydrology models.

````@example HAB
state = HydroState(grid, mask, h, b);
nothing #hide
````

Finally we can create a simulation struct. Since the Kazmierczak et al 2024 model is a steady state problem we create a SteadyStateSimulation rather than a TimeSimulation.

````@example HAB
sim = SteadyStateSimulation(model, grid, state);
nothing #hide
````

We can now run the simulation which will update the hydrology state, as well as the model-dependent fields defined in the model.

````@example HAB
run!(sim)
````

Preprocessing the data before plotting

````@example HAB
state.N .*= 1e-6 # makes N [MPa]
state.N .= mask_field(state.N, state.mask, NaN);
nothing #hide
````

Now we can visualize the resulting effective pressure N [MPa].

````@example HAB
fig_N = visualize_field(state.N; plot_title = "N", transpose_data = true, colorrange = (0, 10))
````

