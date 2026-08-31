############################################################################
# Checkpoint/resume: save/restore an AbstractHydroState's fields (generic over
# any subtype via fieldnames) plus a step/time tag, so a coupled driver's own
# loop can resume from where a killed run left off. Mirrors Shakti.jl's
# checkpoint.jl (Shakti/src/checkpoint.jl), adapted to NCDatasets (already a
# hard dependency here, unlike Shakti's JLD2) and to AbstractHydroState
# instead of a Shakti Simulation's full physics state. See output.jl for the
# separate (but related) periodic-output-writer concern: checkpointing is
# about resuming a killed run, periodic output is about not losing partial
# results even if it's never resumed.
############################################################################

"""
$(TYPEDSIGNATURES)

Reads a state/model field back as a plain `Array{Float64}`, ready to write to disk:
`interior(...)` for an Oceananigans `Field` (`OGRectHydroGrid`), the array itself otherwise
(`ArrayHydroGrid`). Shared by [`save_checkpoint`](@ref) and [`write_output!`](@ref).
"""
_output_array(field::Oceananigans.Fields.Field) = Array(interior(field, :, :, 1))
_output_array(field::AbstractArray) = Array(field)

"""
$(TYPEDSIGNATURES)

Copies plain array `data` into `field` in place: broadcasts into `interior(...)` for an
Oceananigans `Field` (`OGRectHydroGrid`), into `field` directly otherwise (`ArrayHydroGrid`).
Inverse of [`_output_array`](@ref); used by [`load_checkpoint!`](@ref).
"""
_copy_into!(field::Oceananigans.Fields.Field, data) = (interior(field, :, :, 1) .= data; nothing)
_copy_into!(field::AbstractArray, data) = (field .= data; nothing)

"""
$(TYPEDSIGNATURES)

Saves every field of `state` (generic over any `AbstractHydroState` subtype via `fieldnames`, so
this works for `HydroState`'s `mask`/`h`/`b`/`N`/`W` without hardcoding them) to `path`, tagged
with `step`/`time`.

Written to a temp file and renamed into place (same filesystem, so `mv` is an atomic rename)
rather than written directly to `path`, so a crash mid checkpoint-write can't leave a
truncated/corrupt checkpoint behind -- the previous good checkpoint at `path` stays intact until
the new one has fully landed. Mirrors Shakti.jl's own `save_checkpoint` (`Shakti/src/checkpoint.jl`),
using NetCDF (via `NCDatasets`, already a hard dependency here) instead of JLD2.

# Notes

FastHydrology's own models ([`KazmierczakHydroModel`](@ref)/[`HABHydroModel`](@ref)) are
steady-state (see their docstrings): `state.N`/`state.W` are fully recomputed from `state.mask`/
`h`/`b` and the model's own input fields on every [`run!`](@ref) call, so there is no
hydrology-side integration state that genuinely needs resuming. This checkpoint exists for the
coupled-driver pattern instead (e.g. an ice-flow model calling `run!` once per iteration of its own
loop -- see Kryonomos.jl's MISMIP scripts): resuming a killed job needs the ice model's own state
(geometry, velocity, ...) restored too, which is outside FastHydrology's scope -- this only
restores the hydrology side (the exact `mask`/`h`/`b` inputs the checkpointed step used, plus the
`N`/`W` outputs, in case a caller wants them), so the driver's own loop can pick back up at
`step`/`time` knowing exactly which iteration was last saved.
"""
function save_checkpoint(path::String, state::AbstractHydroState, step::Integer, time::Real)
    names = fieldnames(typeof(state))
    isempty(names) && error("save_checkpoint: $(typeof(state)) has no fields to save")
    Nx, Ny = size(_output_array(getfield(state, first(names))))
    tmp_path = path * ".tmp"
    isfile(tmp_path) && rm(tmp_path)
    NCDataset(tmp_path, "c") do ds
        ds.attrib["step"] = Int(step)
        ds.attrib["time"] = Float64(time)
        defDim(ds, "xc", Nx)
        defDim(ds, "yc", Ny)
        for name in names
            v = defVar(ds, String(name), Float64, ("xc", "yc"))
            v[:, :] = _output_array(getfield(state, name))
        end
    end
    mv(tmp_path, path; force = true)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Restores `state`'s fields in place from a checkpoint written by [`save_checkpoint`](@ref), and
returns `(step, time)` so the caller's own loop knows where to resume. Generic over any
`AbstractHydroState` subtype via `fieldnames`, matching [`save_checkpoint`](@ref).
"""
function load_checkpoint!(state::AbstractHydroState, path::String)
    step, time = NCDataset(path, "r") do ds
        for name in fieldnames(typeof(state))
            _copy_into!(getfield(state, name), Array{Float64}(ds[String(name)][:, :]))
        end
        return ds.attrib["step"], ds.attrib["time"]
    end
    return (Int(step), Float64(time))
end
