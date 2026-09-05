using Oceananigans.Advection: adapt_advection_order, materialize_advection
using Oceananigans.BoundaryConditions: fill_halo_regions!, regularize_field_boundary_conditions
using Oceananigans.Fields: TracerFields, ZeroField, tracernames
using Oceananigans.Forcings: model_forcing
using Oceananigans.Grids: Periodic, RectilinearGrid, topology
using Oceananigans.TimeSteppers: _ab2_step_field!, _rk3_substep_field!
using Oceananigans.TurbulenceClosures: with_tracers
using Oceananigans.Utils: launch!

#####
##### Validation and materialization
#####

function validate_extended_tracers(et::ExtendedTracers, grid, tracers)
    grid isa RectilinearGrid ||
        throw(ArgumentError("ExtendedTracers require a RectilinearGrid; got a $(typeof(grid).name.wrapper)"))

    TX, TY, _ = topology(grid)

    if et.east + et.west > 0 && TX !== Periodic
        throw(ArgumentError("The tracer domain can only be extended in x if x is Periodic; got $TX. " *
                            "ENDLESS relies on the periodic extension of the velocity field."))
    end

    if et.north + et.south > 0 && TY !== Periodic
        throw(ArgumentError("The tracer domain can only be extended in y if y is Periodic; got $TY. " *
                            "ENDLESS relies on the periodic extension of the velocity field."))
    end

    for name in et.names
        name ∈ tracernames(tracers) &&
            throw(ArgumentError("The extended tracer :$name collides with a model tracer of the same name. " *
                                "Extended tracers must be named distinctly from model tracers."))
    end

    return nothing
end

"""
$(TYPEDSIGNATURES)

Build the extended grid, tracer fields, tendency fields and periodically-tiled views of the
model fields that an [`ExtendedTracers`](@ref) specification needs in order to be stepped.
"""
function materialize_extended_tracers(et::ExtendedTracers, grid, clock, advection, closure,
                                      velocities, tracers, auxiliary_fields, closure_fields)

    validate_extended_tracers(et, grid, tracers)

    names = et.names
    extended_tracer_grid = extended_grid(grid; et.east, et.west, et.north, et.south)

    bcs = regularize_field_boundary_conditions(et.boundary_conditions, extended_tracer_grid, names)
    extended_tracer_fields = TracerFields(names, extended_tracer_grid, bcs)

    Gⁿ = map(similar, extended_tracer_fields)
    G⁻ = map(similar, extended_tracer_fields)

    extended_advection = isnothing(et.advection) ? advection : et.advection
    extended_advection = materialize_advection(adapt_advection_order(extended_advection, extended_tracer_grid), extended_tracer_grid)
    extended_closure = with_tracers(names, isnothing(et.closure) ? closure : et.closure)

    # Zero-memory periodic views of the base-grid fields. These are built once: the fields
    # they wrap are updated in place, so the views never go stale.
    tiled_velocities = tile(velocities, grid)
    tiled_closure_fields = tile(closure_fields, grid)
    tiled_auxiliary_fields = merge(tile(tracers, grid), tile(auxiliary_fields, grid))

    # No background fields: `velocities = nothing` selects the allocation-free branch of
    # `sum_of_velocities`, and `ZeroField` tracers compile the background flux divergence away.
    background_fields = BackgroundFields{false}(nothing, NamedTuple{names}(map(_ -> ZeroField(), names)))

    model_fields = merge(tiled_velocities, extended_tracer_fields, tiled_auxiliary_fields)
    forcing = model_forcing(et.forcing, model_fields, extended_tracer_fields)

    immersed_bcs = NamedTuple(name => extended_tracer_fields[name].boundary_conditions.immersed for name in names)

    return ExtendedTracers(names, et.east, et.west, et.north, et.south, bcs,
                           extended_tracer_grid, extended_tracer_fields, Gⁿ, G⁻,
                           extended_advection, extended_closure, forcing, background_fields,
                           tiled_velocities, tiled_auxiliary_fields, tiled_closure_fields, immersed_bcs)
end

#####
##### Time-stepping hooks. Each is a no-op when `model.extended_tracers === nothing`.
#####

compute_extended_tracer_tendencies!(model::NonhydrostaticModel) =
    compute_extended_tracer_tendencies!(model.extended_tracers, model)

function compute_extended_tracer_tendencies!(et::ExtendedTracers, model::NonhydrostaticModel)
    grid = et.grid
    arch = architecture(grid)
    model_fields = merge(et.velocities, et.tracers, et.auxiliary_fields)

    fill_halo_regions!(et.tracers, model.clock, model_fields)

    for (n, name) in enumerate(et.names)
        launch!(arch, grid, :xyz, compute_Gc!,
                et.Gⁿ[name], grid,
                Val(n), Val(name), et.advection, et.closure, et.immersed_bcs[name],
                model.buoyancy, nothing, et.background_fields,
                et.velocities, et.tracers, et.auxiliary_fields, et.closure_fields,
                model.clock, et.forcing[name])
    end

    return nothing
end

ab2_step_extended_tracers!(model::NonhydrostaticModel, Δt, χ) =
    ab2_step_extended_tracers!(model.extended_tracers, model, Δt, χ)

function ab2_step_extended_tracers!(et::ExtendedTracers, model::NonhydrostaticModel, Δt, χ)
    grid = et.grid
    kernel_Δt = convert(eltype(grid), Δt)

    for name in et.names
        launch!(architecture(grid), grid, :xyz, _ab2_step_field!,
                et.tracers[name], kernel_Δt, χ, et.Gⁿ[name], et.G⁻[name])
    end

    return nothing
end

rk3_substep_extended_tracers!(model::NonhydrostaticModel, Δt, γⁿ, ζⁿ) =
    rk3_substep_extended_tracers!(model.extended_tracers, model, Δt, γⁿ, ζⁿ)

function rk3_substep_extended_tracers!(et::ExtendedTracers, model::NonhydrostaticModel, Δt, γⁿ, ζⁿ)
    grid = et.grid
    kernel_Δt = convert(eltype(grid), Δt)

    for name in et.names
        launch!(architecture(grid), grid, :xyz, _rk3_substep_field!,
                et.tracers[name], kernel_Δt, γⁿ, ζⁿ, et.Gⁿ[name], et.G⁻[name])
    end

    return nothing
end

cache_previous_extended_tracer_tendencies!(model::NonhydrostaticModel) =
    cache_previous_extended_tracer_tendencies!(model.extended_tracers, model)

function cache_previous_extended_tracer_tendencies!(et::ExtendedTracers, model::NonhydrostaticModel)
    grid = et.grid

    for name in et.names
        launch!(architecture(grid), grid, :xyz, _cache_field_tendencies!,
                et.G⁻[name], et.Gⁿ[name])
    end

    return nothing
end

#####
##### Checkpointing
#####

function Oceananigans.prognostic_state(et::ExtendedTracers)
    return (tracers = Oceananigans.prognostic_state(et.tracers),
            Gⁿ = Oceananigans.prognostic_state(et.Gⁿ),
            G⁻ = Oceananigans.prognostic_state(et.G⁻))
end

function Oceananigans.restore_prognostic_state!(restored::ExtendedTracers, from)
    Oceananigans.restore_prognostic_state!(restored.tracers, from.tracers)
    Oceananigans.restore_prognostic_state!(restored.Gⁿ, from.Gⁿ)
    Oceananigans.restore_prognostic_state!(restored.G⁻, from.G⁻)
    return restored
end

Oceananigans.restore_prognostic_state!(::ExtendedTracers, ::Nothing) = nothing
