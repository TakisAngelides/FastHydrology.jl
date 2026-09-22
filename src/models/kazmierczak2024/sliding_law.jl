#################################
# Model: Kazmierczak et al 2024 #
#################################


"""
$(TYPEDSIGNATURES)

Basal shear stress tau_b [Pa] from `law`, given the current effective pressure `N` [Pa] and the
magnitude of the basal sliding velocity `abs_v_b` [m/s]. Feeds the frictional-heating term
tau_b*v_b/L_w of the melt rate (Eq. 3, Sec. 2.2.1 of Kazmierczak et al 2024). See the
`AbstractSlidingLaw` docstring in model.jl for the physics/provenance of each law. A plain scalar
function (used for tests/diagnostics on ordinary numbers) -- `update_tau_b!` below is the version
actually used inside the model's field broadcasts.
"""
calc_tau_b(::PrescribedFrictionSlidingLaw, N, abs_v_b) = zero(abs_v_b)

calc_tau_b(law::WeertmanSlidingLaw, N, abs_v_b) = law.C * abs_v_b^law.q

calc_tau_b(law::PowerPlasticSlidingLaw, N, abs_v_b) = law.c_till * N * (abs_v_b / law.u0)^law.q

calc_tau_b(law::RegularizedCoulombSlidingLaw, N, abs_v_b) = law.c_till * N * (abs_v_b / (abs_v_b + law.u0))^law.q


"""
$(TYPEDSIGNATURES)

Update `model.tau_b` in place from `sliding_law` and the current `state.N`/`model.abs_v_b`.
Duplicates the `calc_tau_b` formulas above with the law's parameters pulled out as plain scalar
locals first, rather than calling `calc_tau_b` from inside the `@.` broadcast, because
`OGRectHydroGrid`'s `Field` broadcasting goes through Oceananigans' `AbstractOperations` machinery,
which only recognises registered arithmetic/operators inside a broadcast -- not arbitrary
multi-argument user functions taking a struct argument. Writing the formula with plain arithmetic
(`+`, `*`, `/`, `^`) on fields/arrays and scalar locals, exactly as the rest of this model already
does (e.g. `update_N_inf!` in effective_pressure.jl), works uniformly for both grid backends.
"""
function update_tau_b!(model::KazmierczakHydroModel, state::HydroState, ::PrescribedFrictionSlidingLaw)
    model.tau_b .= 0.0
    return nothing
end

function update_tau_b!(model::KazmierczakHydroModel, state::HydroState, law::PrescribedFieldSlidingLaw)
    model.tau_b .= law.tau_b
    return nothing
end

function update_tau_b!(model::KazmierczakHydroModel, state::HydroState, law::WeertmanSlidingLaw)
    C, q = law.C, law.q
    @. model.tau_b = C * model.abs_v_b^q
    return nothing
end

function update_tau_b!(model::KazmierczakHydroModel, state::HydroState, law::PowerPlasticSlidingLaw)
    c_till, q, u0 = law.c_till, law.q, law.u0
    @. model.tau_b = c_till * state.N * (model.abs_v_b / u0)^q
    return nothing
end

# RegularizedCoulombSlidingLaw (scalar c_till) and RegularizedCoulombFieldSlidingLaw (per-cell
# c_till) share this one method: the formula is identical, and broadcasting already handles a
# scalar or a Field for c_till transparently, so there is nothing that actually differs per type.
function update_tau_b!(model::KazmierczakHydroModel, state::HydroState, law::Union{RegularizedCoulombSlidingLaw, RegularizedCoulombFieldSlidingLaw})
    c_till, q, u0 = law.c_till, law.q, law.u0
    @. model.tau_b = c_till * state.N * (model.abs_v_b / (model.abs_v_b + u0))^q
    return nothing
end

function update_tau_b!(model::KazmierczakHydroModel, state::HydroState, law::ShaktiRegularizedCoulombSlidingLaw)
    C, n, inv_n, lambda = law.C, law.n, law.inv_n, law.lambda
    @. model.tau_b = C * state.N * (model.abs_v_b / (model.abs_v_b + abs(state.N)^n * lambda))^inv_n
    return nothing
end
