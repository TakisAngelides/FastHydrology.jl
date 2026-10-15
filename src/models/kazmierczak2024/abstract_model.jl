#################################
# Model: Kazmierczak et al 2024 #
#################################


"""
$(TYPEDSIGNATURES)

Supertype of [`KazmierczakHydroModel`](@ref), its only subtype. Declared first so that the option files
(`drainage.jl` to `water_thickness.jl`), which `model.jl` needs before it can define `KazmierczakParams`, can
keep their model methods next to their types. Dispatch and performance are unchanged: Julia still
compiles every method for the concrete model type.
"""
abstract type AbstractKazmierczakHydroModel <: AbstractHydroModel end
