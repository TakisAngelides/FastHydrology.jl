"""
Register `min`, `max`, and `erf` as Oceananigans `AbstractOperation`s so they can be broadcast
directly onto `Field`s with `@.` (e.g. `@. model.q = min(model.q, 1e5)`), the same way Oceananigans
itself registers `+`, `-`, `*`, `/`, `^`, `sqrt`, etc. Without this, broadcasting an unregistered
function over a `Field` throws a `MethodError`, forcing a drop down to raw arrays via `.data`.
"""

import Base: min, max
import SpecialFunctions: erf
using Oceananigans.AbstractOperations: @unary, @binary

@binary min
@binary max
@unary erf
