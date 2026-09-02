####################################
# Model: Height above buoyancy (HAB) #
####################################


"""
$(TYPEDSIGNATURES)

The height above buoyancy (HAB) hydrology model described in Sec. 2.1.1 of Kazmierczak et al 2022 (https://doi.org/10.5194/tc-16-4537-2022).
"""
mutable struct HABHydroModel{T <: AbstractFloat, A} <: AbstractHydroModel

    # Model constants
    rho_sw ::T  # Density of sea water [kg/m3]
    rho_i  ::T  # Density of ice [kg/m3]
    g      ::T  # Gravitational acceleration [m/s2]
    P_w    ::T  # Water pressure coefficient; see below Eq. (3) of Kazmierczak et al 2022
    sigmat ::T  # Effective pressure lower bound as fraction of overburden pressure

    # Effective pressure
    Po  ::A  # Ice overburden pressure (rho_i * g * ice_thickness) [Pa]
    p_w ::A  # Water pressure [Pa]

end


"""
$(TYPEDSIGNATURES)

The constructor to the HAB hydrology model.

Works with any concrete subtype of AbstractHydroGrid -- changing the grid does not require changing this constructor.

# Arguments

- `grid::AbstractHydroGrid`: grid of the simulation
"""
function HABHydroModel(grid::AbstractHydroGrid)

    T = typeof(grid.dx)

    # Physical constants
    rho_sw = T(1027.0)
    rho_i  = T(917.0)
    g      = T(9.81)
    P_w    = T(0.96)
    sigmat = T(0.02)

    # Effective pressure fields
    Po  = alloc_field(grid)
    p_w = alloc_field(grid)

    return HABHydroModel(rho_sw, rho_i, g, P_w, sigmat, Po, p_w)

end
