############################################################################
# Periodic output: write named fields to disk at whatever interval a caller's
# own loop decides on, so a killed run leaves every slice written before the
# kill intact and readable -- see this file's motivating incident, a ~4000-
# step coupled Yelmo+FastHydrology run (Kryonomos.jl's MISMIP scripts) that
# only wrote its NetCDF result once, at the very end, and lost everything
# when it hit its wall-time limit. See checkpoint.jl for the separate (but
# related) full-state resume concern.
############################################################################

"""
$(TYPEDSIGNATURES)

How periodic simulation output is recorded to disk -- multiple dispatch on the concrete subtype
picks the file format. Currently only [`NetCDFOutputWriter`](@ref) is provided, since `NCDatasets`
is already a hard dependency of FastHydrology and every existing coupled-driver script already
writes NetCDF output by hand (see e.g. Kryonomos.jl's MISMIP scripts) -- this factors that pattern
into a reusable, resumable writer instead of it being hand-rolled anew in every driver. See
[`write_output!`](@ref)/[`close_output!`](@ref) below, and [`save_checkpoint`](@ref)/
[`load_checkpoint!`](@ref) (checkpoint.jl) for the related but separate full-state resume
mechanism.

# Notes

FastHydrology's own models ([`KazmierczakHydroModel`](@ref)/[`HABHydroModel`](@ref)) are
steady-state (see their docstrings): they have no internal time loop of their own for a writer to
hook into automatically -- [`SteadyStateSimulation`](@ref) re-solves fresh from the current
geometry on every [`run!`](@ref) call, and owns no loop across calls. An `AbstractOutputWriter` is
instead meant to be driven directly from whichever loop owns the repeated `run!` calls -- typically
a coupled driver's own timestep loop (e.g. an ice-flow model), which already decides when to call
`run!`; call [`write_output!`](@ref) there too, at whatever interval that loop chooses.

[`TimeSimulation`](@ref) currently only wraps [`ShaktiHydroModel`](@ref), whose `run!` already
delegates entirely to `Shakti.run!` -- which has its own, independent periodic-output/checkpoint
machinery (`checkpoint_every`/`checkpoint_path`/`restart_path`, forwarded straight through as
kwargs already) that already solves this same problem for that model. There is nothing for
`AbstractOutputWriter` to hook into there without duplicating what Shakti already does; it exists
for the models (and coupled-driver patterns) that have no such mechanism of their own.
"""
abstract type AbstractOutputWriter end

"""
$(TYPEDSIGNATURES)

Records named fields to a NetCDF file, one time-slice at a time, along an unlimited `time`
dimension -- so a killed run leaves every slice written before the kill intact and readable
(NetCDF's own on-disk file structure, not a buffer FastHydrology holds in memory), instead of
losing everything the way a single write-at-the-end script does.

Build with the [`NetCDFOutputWriter`](@ref) constructor; write a time-slice with
[`write_output!`](@ref); close with [`close_output!`](@ref) when done.
"""
struct NetCDFOutputWriter{G <: AbstractHydroGrid} <: AbstractOutputWriter
    path::String
    grid::G
    field_names::Vector{Symbol}
    ds::Any
end

"""
$(TYPEDSIGNATURES)

Drops every time-slice with `step > resume_step` from the NetCDF file at `path`, in place, before
[`NetCDFOutputWriter`](@ref) reopens it for further appends -- a killed run's file may hold slices
written after the last checkpoint actually being resumed from (see [`save_checkpoint`](@ref)), and
replaying those steps would otherwise duplicate them. A no-op if every existing slice already has
`step <= resume_step`.

NetCDF's unlimited dimension can grow but not shrink in place, so trimming means rewriting: copies
the kept slices into a temp file with the same schema, then renames it over `path` (atomic, same
filesystem) once fully written -- same crash-safety reasoning as [`save_checkpoint`](@ref).

# Notes

Writes with an explicit index range (`v[1:n] = ...`), not `v[:] = ...`: assigning to `[:]` on a
variable along an as-yet-empty unlimited dimension is a silent no-op in NCDatasets (it selects the
*current*, zero-length extent rather than growing to match the right-hand side) -- confirmed
directly against NCDatasets' own behaviour, not assumed.
"""
function _trim_output_after!(path::String, field_names::AbstractVector{Symbol}, resume_step::Int)
    NCDataset(path, "r") do src
        steps = Array(src["step"])
        keep = findall(<=(resume_step), steps)
        length(keep) == length(steps) && return nothing

        first_field = src[String(first(field_names))]
        Nx, Ny = size(first_field, 1), size(first_field, 2)
        n_keep = length(keep)

        tmp_path = path * ".trim.tmp"
        isfile(tmp_path) && rm(tmp_path)
        NCDataset(tmp_path, "c") do dst
            defDim(dst, "xc", Nx)
            defDim(dst, "yc", Ny)
            defDim(dst, "time", Inf)
            defVar(dst, "step", Int, ("time",))[1:n_keep] = steps[keep]
            defVar(dst, "time", Float64, ("time",))[1:n_keep] = Array(src["time"])[keep]
            for name in field_names
                v = defVar(dst, String(name), Float64, ("xc", "yc", "time"))
                v[:, :, 1:n_keep] = Array(src[String(name)])[:, :, keep]
            end
        end
        mv(tmp_path, path; force = true)
        return nothing
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Builds a [`NetCDFOutputWriter`](@ref) that will record `field_names` (names given to
[`write_output!`](@ref)'s `fields` `NamedTuple`, e.g. `[:N, :W]`) on `grid`, writing to a NetCDF
file at `path`.

# Keywords

- `resume_step`: if given, instead reopens an existing file at `path` (previously written by a
  `NetCDFOutputWriter` with the same `field_names`) for further appends, picking up exactly where
  a killed run left off. Any time-slices with `step > resume_step` are dropped first (see
  [`_trim_output_after!`](@ref)) -- they were written by the crashed run *after* the last
  checkpoint actually being resumed from, so replaying those steps would otherwise duplicate them.
    (**Default**: `nothing`)
- `overwrite`: only consulted when `resume_step` is not given. If `path` already exists,
  `overwrite = true` deletes it and starts a fresh file; `overwrite = false` errors instead, to
  avoid silently discarding an existing file.
    (**Default**: `true`)
"""
function NetCDFOutputWriter(path::String, grid::AbstractHydroGrid, field_names::AbstractVector{Symbol};
                             resume_step::Union{Nothing, Integer} = nothing, overwrite::Bool = true)
    field_names = collect(Symbol, field_names)
    if resume_step !== nothing
        isfile(path) || error("NetCDFOutputWriter: resume_step given but no file found at \"$path\"")
        _trim_output_after!(path, field_names, Int(resume_step))
        ds = NCDataset(path, "a")
        for name in field_names
            haskey(ds, String(name)) || error("NetCDFOutputWriter: \"$path\" has no variable \"$name\" to resume")
        end
    else
        if isfile(path)
            overwrite || error("NetCDFOutputWriter: \"$path\" already exists (pass overwrite = true, or resume_step to continue it)")
            rm(path)
        end
        ds = NCDataset(path, "c")
        defDim(ds, "xc", grid.Nx)
        defDim(ds, "yc", grid.Ny)
        defDim(ds, "time", Inf)
        defVar(ds, "step", Int, ("time",))
        defVar(ds, "time", Float64, ("time",))
        for name in field_names
            defVar(ds, String(name), Float64, ("xc", "yc", "time"))
        end
    end
    return NetCDFOutputWriter(path, grid, field_names, ds)
end

"""
$(TYPEDSIGNATURES)

Appends one time-slice to `writer`'s file: `fields[name]` for each `name` in `writer.field_names`
(a `NamedTuple`, e.g. `(N = state.N, W = state.W)` -- values can be either an Oceananigans `Field`
or a plain `Array`, matching whichever grid backend is in use), tagged with `step`/`time`. Flushes
to disk immediately (`NCDatasets.sync`) so the write actually lands even if the process is killed
right after this call returns -- the entire point of this writer.
"""
function write_output!(writer::NetCDFOutputWriter, step::Integer, time::Real, fields::NamedTuple)
    ds = writer.ds
    idx = size(ds["step"], 1) + 1
    ds["step"][idx] = Int(step)
    ds["time"][idx] = Float64(time)
    for name in writer.field_names
        haskey(fields, name) || error("write_output!: field \"$name\" not provided (writer was built with field_names = $(writer.field_names))")
        ds[String(name)][:, :, idx] = _output_array(fields[name])
    end
    NCDatasets.sync(ds)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Closes `writer`'s underlying NetCDF file handle.
"""
function close_output!(writer::NetCDFOutputWriter)
    close(writer.ds)
    return nothing
end
