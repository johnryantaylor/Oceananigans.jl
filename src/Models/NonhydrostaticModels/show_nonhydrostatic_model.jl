using Oceananigans.Utils: prettytime, prettykeys
using Oceananigans.TurbulenceClosures: closure_summary

function Base.summary(model::NonhydrostaticModel)
    A = Base.summary(architecture(model.grid))
    G = nameof(typeof(model.grid))
    return string("NonhydrostaticModel{$A, $G}",
                  "(time = ", prettytime(model.clock.time), ", iteration = ", model.clock.iteration, ")")
end

function Base.show(io::IO, model::NonhydrostaticModel)
    TS = nameof(typeof(model.timestepper))
    tracernames = prettykeys(model.tracers)

    print(io, summary(model), "\n",
        "├── grid: ", summary(model.grid), "\n",
        "├── timestepper: ", TS, "\n",
        "├── advection scheme: ", summary(model.advection), "\n",
        "├── tracers: ", tracernames, "\n",
        "├── closure: ", closure_summary(model.closure), "\n",
        "├── buoyancy: ", summary(model.buoyancy), "\n")

    entries = ["coriolis: " * summary(model.coriolis)]
    isnothing(model.particles)        || push!(entries, "particles: " * summary(model.particles))
    isnothing(model.extended_tracers) || push!(entries, "extended tracers: " * extension_summary(model.extended_tracers))

    for (n, entry) in enumerate(entries)
        is_last = n == length(entries)
        print(io, is_last ? "└── " : "├── ", entry, is_last ? "" : "\n")
    end
end
