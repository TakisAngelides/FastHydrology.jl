"""
$(TYPEDSIGNATURES)

An abstract type for the hydrology model to be simulated. The model can hold revelant constants and model-specific fields.
"""
abstract type AbstractHydroModel end


#################################
# Model: Kazmierczak et al 2024 #
#################################


"""
$(TYPEDSIGNATURES)

The hydrology model described in Kazmierczak et al 2024 (https://doi.org/10.5194/tc-18-5887-2024). In our implementation here for
the calculations of water flux, we make the assumption that on the grid we have Delta_x = Delta_y.
"""
mutable struct KazmierczakHydroModel{T <: AbstractFloat, A} <: AbstractHydroModel

    # Model constants
    rho_w           ::T    # Density of fresh water [kg/m3]
    rho_i           ::T    # Density of ice [kg/m3]
    g               ::T    # Gravitational acceleration [m/s2]
    L_w             ::T    # Latent heat of fusion for ice [J/kg]
    n               ::T    # Glen's flow law exponent (typically 3)
    h_b             ::T    # Typical bed obstacle height [m]
    alpha           ::T    # Power law exponent for hydraulic transmissivity (m-scale)
    beta            ::T    # Power law exponent for hydraulic transmissivity (opening/closing)
    f               ::T    # Darcy-Weisbach friction factor
    F_till          ::T    # Till compressibility/yield factor for soft-bed transition
    Q_c             ::T    # Threshold discharge for laminar-to-turbulent transition [m3/s]
    H_0             ::T    # Thickness of canals for soft bed deformation [m]
    l_c             ::T    # Distance between conduits [m]
    K               ::T    # Conductivity coefficient in Darcy-Weisbach relation
    eta_w           ::T    # Dynamic viscosity of water [Pa s]
    Wmin            ::T    # Minimum subglacial water layer thickness [m]
    Wmax            ::T    # Maximum subglacial water layer thickness [m]
    longcoupwater   ::T    # Longitudinal coupling factor for the stress-gradient coupling smoothing of the geometric potential gradients
    sigmat          ::T    # Effective pressure lower bound as fraction of overburden pressure
    fill_iters      ::Int  # How many iterations to perform for the filling of local minima of the geometric potential phi0

    # Geometric potential
    phi0                   ::A  # Geometric potential [Pa]
    phi0_tmp               ::A  # Temporary storage for potential filling of phi0 to smoothen local minima and avoid stuck water
    minus_grad_phi0_x      ::A  # Geometric potential gradient x-component [Pa/m]
    minus_grad_phi0_y      ::A  # Geometric potential gradient y-component [Pa/m]
    abs_grad_phi0          ::A  # Magnitude of the geometric potential gradient [Pa/m]
    minus_grad_phi0_sx     ::A  # Smoothed gradient x-component of the geometric potential [Pa/m]
    minus_grad_phi0_sy     ::A  # Smoothed gradient y-component of the geometric potential [Pa/m]
    abs_grad_phi0_s        ::A  # Magnitude of the smoothed gradient of the geometric potential [Pa/m]

    # Water flux
    visited    ::A  # visited cells during the recursive algorithm to calculate psi_out
    h          ::A  # ice thickness after geometric potential filling serves as a temporary storage [m]
    mdot       ::A  # mass basal melt rate per unit area [Kg / m^2 / s]
    mdot_total ::A  # mdot plus the dissipation melt rate |q * grad(phi0)| / L_w [Kg / m^2 / s]
    psi_out    ::A  # Integrated scalar water flux [m3/s]
    corfac     ::A  # Correction factor to go from psi_out to q
    q          ::A  # Distributed water flux [m2/s]
    q_prev     ::A  # q from the previous Picard sweep, for the dissipation-melt convergence check

    # Effective pressure and Bed state
    Q       ::A  # Volumetric water flux within a conduit [m3/s]
    kappa   ::A  # Bed type indicator (0: hard, 1: soft)
    abs_v_b ::A  # Magnitude of basal sliding velocity [m/s]
    A_visc  ::A  # Ice flow law rate factor (Glen's A) [Pa^-n s^-1]
    S_inf   ::A  # Far-field (away from grounding line) conduit cross-sectional area [m2]
    H_hard  ::A  # Thickness of conduits over a hard bed [m]
    H_soft  ::A  # Thickness of conduits over a soft bed [m]
    H       ::A  # Thickness of conduits [m]
    N_inf   ::A  # Far-field (away from grounding line) effective pressure [Pa]
    Po      ::A  # Ice overburden pressure (rho_i * g * ice_thickness) [Pa]

end


"""
$(TYPEDSIGNATURES)

The constructor to the Kazmierczak et al 2024 hydrology model. All fields are initialized here to zero, except from
the viscosity parameter A_visc from Glen's flow, the kappa field describing the hardness of the bed, and the absolute value
of the basal velocity of the ice. These three fields are given values from an input file. The user must provide these fields
for the simulation to be able to start.

Works with any concrete subtype of AbstractHydroGrid -- changing the grid does not require changing this constructor.

# Arguments

- `grid::AbstractHydroGrid`: grid of the simulation
- `kappa_in::AbstractArray{<:AbstractFloat}`: Bed type indicator (0: hard, 1: soft)
- `abs_v_b_in::AbstractArray{<:AbstractFloat}`: Magnitude of basal sliding velocity [m/s]
- `A_visc_in::AbstractArray{<:AbstractFloat}`: Ice flow law rate factor (Glen's A) [Pa^-n s^-1]
- `mdot_in::AbstractArray{<:AbstractFloat}`: mass basal melt rate per unit area [Kg / m^2 / s]
"""
function KazmierczakHydroModel(
    grid::AbstractHydroGrid,
    kappa_in::AbstractArray{<:AbstractFloat},
    abs_v_b_in::AbstractArray{<:AbstractFloat},
    A_visc_in::AbstractArray{<:AbstractFloat},
    mdot_in::AbstractArray{<:AbstractFloat};
    rho_w         = 1000.0,
    rho_i         = 917.0,
    g             = 9.81,
    L_w           = 3.34e5,
    n             = 3.0,
    h_b           = 0.1,
    alpha         = 5/4,
    beta          = 3/2,
    f             = 0.1,
    F_till        = 1.1,
    Q_c           = 1.0,
    H_0           = 0.1,
    l_c           = 10000.0,
    eta_w         = perYear2perSecond(1.8e-3),
    Wmin          = 1e-8,
    Wmax          = 0.015,
    longcoupwater = 5.0,
    sigmat        = 0.02,
    fill_iters    = 10
)

    expected_size = (grid.Nx, grid.Ny)
    for (name, arr) in [("kappa", kappa_in), ("abs_v_b", abs_v_b_in), ("A_visc", A_visc_in), ("mdot_in", mdot_in)]
        size(arr)[1:2] == expected_size || throw(ArgumentError("$name size $(size(arr)) != grid size $expected_size"))
    end

    T = typeof(grid.dx)

    # Physical constants
    rho_w         = T(rho_w)
    rho_i         = T(rho_i)
    g             = T(g)
    L_w           = T(L_w)
    n             = T(n)
    h_b           = T(h_b)
    alpha         = T(alpha)
    beta          = T(beta)
    f             = T(f)
    F_till        = T(F_till)
    Q_c           = T(Q_c)
    H_0           = T(H_0)
    l_c           = T(l_c)
    K             = (T(2)/T(pi))^(T(0.25)) * sqrt((T(pi) + T(2)) / (rho_w * f))
    eta_w         = T(eta_w)
    Wmin          = T(Wmin)
    Wmax          = T(Wmax)
    longcoupwater = T(longcoupwater)
    sigmat        = T(sigmat)
    fill_iters    = Int(fill_iters)

    # Geometric potential
    phi0          = alloc_field(grid)
    phi0_tmp      = alloc_field(grid)
    minus_grad_phi0_x = alloc_field(grid)
    minus_grad_phi0_y = alloc_field(grid)
    abs_grad_phi0     = alloc_field(grid)
    minus_grad_phi0_sx = alloc_field(grid)
    minus_grad_phi0_sy = alloc_field(grid)
    abs_grad_phi0_s    = alloc_field(grid)

    # Water flux
    visited    = alloc_field(grid)
    h          = alloc_field(grid)
    mdot       = alloc_field(grid, mdot_in)
    mdot_total = alloc_field(grid)
    psi_out    = alloc_field(grid)
    corfac     = alloc_field(grid)
    q          = alloc_field(grid)
    q_prev     = alloc_field(grid)

    # Effective pressure
    Q       = alloc_field(grid)
    kappa   = alloc_field(grid, kappa_in)
    abs_v_b = alloc_field(grid, abs_v_b_in)
    A_visc  = alloc_field(grid, A_visc_in)
    S_inf   = alloc_field(grid)
    H_hard  = alloc_field(grid)
    H_soft  = alloc_field(grid)
    H       = alloc_field(grid)
    N_inf   = alloc_field(grid)
    Po      = alloc_field(grid)

    return KazmierczakHydroModel(
        rho_w, rho_i, g, L_w, n, h_b, alpha, beta, f, F_till, Q_c, H_0, l_c, K, eta_w, Wmin, Wmax, longcoupwater, sigmat, fill_iters,
        phi0, phi0_tmp, minus_grad_phi0_x, minus_grad_phi0_y,
        abs_grad_phi0, minus_grad_phi0_sx, minus_grad_phi0_sy, abs_grad_phi0_s,
        visited, h, mdot, mdot_total, psi_out, corfac, q, q_prev,
        Q, kappa, abs_v_b, A_visc, S_inf, H_hard, H_soft, H, N_inf, Po
    )

end


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


#################################
# Model: Shakti                 #
#################################


"""
$(TYPEDSIGNATURES)

Wraps a `Shakti.Simulation` (see the separate `Shakti.jl` package) so it can be run as an
`AbstractHydroModel`'s time evolution via `TimeSimulation`. Shakti's `Simulation` already owns its
own grid, state, and model parameters, so this wrapper holds nothing else -- FastHydrology's
`run!(::TimeSimulation)` for this model just delegates straight to `Shakti.run!`.

Defined unconditionally (no dependency on `Shakti` itself, since the field is generic over `S`), but
only usable once `Shakti` is loaded alongside `FastHydrology`: the `run!` method that dispatches on
this type is added by the `FastHydrologyShaktiExt` package extension (see `ext/FastHydrologyShaktiExt.jl`),
which Julia loads automatically once both packages are in scope (`using FastHydrology, Shakti`).

# Arguments

- `sim`: a `Shakti.Simulation` built the usual way (`Shakti.Simulation(grid, state, tsteps, dt, p, ...)`).
"""
struct ShaktiHydroModel{S} <: AbstractHydroModel
    sim::S
end
