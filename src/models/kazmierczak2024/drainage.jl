#################################
# Model: Kazmierczak et al 2024 #
#################################

# Drainage modes: which opening terms update_N_inf! includes and which Q_c update_H! uses.


"""
$(TYPEDSIGNATURES)

Trait controlling which opening terms `update_N_inf!` includes in Eq. (5b)/(6a) of Kazmierczak et al
2024, and which `Q_c` `update_H!` uses in Eq. (9) -- i.e. whether the model is left free to switch
between efficient and inefficient drainage (the paper's default behaviour) or is forced to produce
one or the other throughout, as in the paper's Sect. 3.2. Stored as a type parameter and resolved by
multiple dispatch, the same pattern as `AbstractDissipationMelt` and `AbstractPsiOutAlgorithm`.

- `BothDrainage()` (default): both the sliding-over-obstacles opening term (||v_b||h_b, inefficient)
  and the melt-driven opening term (Q_w||grad(phi)||/(rho_i*L_w), efficient) are included in Eq.
  (5b), and `update_H!` uses the model's configured `Q_c` unchanged -- the paper's normal automatic
  switch between drainage types based on the local water flux.
- `EfficientOnly()`: the sliding opening term is dropped from Eq. (5b)/(6a), and `update_H!` is given
  `Q_c = 0` so that Eq. (9)'s exp(-Q/Q_c) -> 0 for any Q > 0, forcing the soft-bed conduit geometry to
  the canal/efficient case (H_soft -> H_0) everywhere.
- `InefficientOnly()`: the melt opening term is dropped from Eq. (5b), and `update_H!` is given
  `Q_c = Inf` so that exp(-Q/Q_c) -> 1, forcing the soft-bed conduit geometry to the inter-clastic-
  film/inefficient case (H_soft -> sqrt(S_inf)/F_till) everywhere.

Note the Q_c -> 0 / Q_c -> Inf assignment above is the *corrected* version of the paper's own Sect.
3.2 description (as published, it prescribes Q_c = Inf for the entirely-efficient case and Q_c = 0
for the entirely-inefficient case); the authors confirmed by email that this is a typo and the two
are swapped, which also matches Eq. (9) directly -- the exp(-Q/Q_c) limits above only work out this
way around.
"""
abstract type AbstractDrainageMode end
struct BothDrainage    <: AbstractDrainageMode end
struct EfficientOnly   <: AbstractDrainageMode end
struct InefficientOnly <: AbstractDrainageMode end


"""
$(TYPEDSIGNATURES)

The `Q_c` value `update_H!` uses in Eq. (9)'s exp(-Q/Q_c) soft-bed geometry blend, selected by
`model.drainage_mode` -- see `AbstractDrainageMode` for the physical justification of each case.
"""
effective_Q_c(::BothDrainage, Q_c) = Q_c
effective_Q_c(::EfficientOnly, Q_c::T) where {T} = zero(T)
effective_Q_c(::InefficientOnly, Q_c::T) where {T} = T(Inf)


"""
$(TYPEDSIGNATURES)

Scalar 0/1 coefficients for the sliding-over-obstacles (inefficient) and melt-driven (efficient)
opening terms of Eq. (5b)/(6a), selected by `model.drainage_mode` -- see `AbstractDrainageMode`.
Resolved once outside the `@.` broadcast in `update_N_inf!`, the same "precompute the scalar
sub-expression" pattern as `denom_const` there, so `EfficientOnly`/`InefficientOnly` cost nothing
beyond multiplying the dropped term by 0.0.
"""
opening_coefficients(::BothDrainage, ::Type{T}) where {T} = (one(T), one(T))
opening_coefficients(::EfficientOnly, ::Type{T}) where {T} = (zero(T), one(T))
opening_coefficients(::InefficientOnly, ::Type{T}) where {T} = (one(T), zero(T))
