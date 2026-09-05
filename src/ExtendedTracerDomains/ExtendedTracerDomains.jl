"""
    ExtendedTracers

Passive tracers evolved on a horizontally-extended domain, following the ENDLESS approach of
Chen, Yang, Meneveau and Chamecki (*Ocean Modelling* 101, 121-132, 2016).

Because a horizontally-periodic LES velocity and pressure field can be replicated in `x` and
`y` without violating mass or momentum conservation, a passive tracer may be evolved on a
domain much larger than the one carrying the velocity, density and pressure. This lets a
plume disperse far beyond the size of the LES box while the momentum solve stays small.

The tracer is stored as a single contiguous `Field` on an extended `RectilinearGrid`, and is
advected by [`TiledArray`](@ref) views of the model velocities, which wrap base-grid indices
periodically at zero memory cost.
"""
module ExtendedTracerDomains

export ExtendedTracers, extended_grid

using DocStringExtensions

using Oceananigans.Utils: tupleit

include("tiled_fields.jl")
include("extended_grid.jl")

"""
    ExtendedTracers(names...; east = 0, west = 0, north = 0, south = 0,
                              advection = nothing,
                              closure = nothing,
                              forcing = NamedTuple(),
                              boundary_conditions = NamedTuple())

Return a specification for passive tracers `names` evolved on a domain built by replicating
the model grid `west` times to the west, `east` times to the east, `south` times to the
south and `north` times to the north, to be passed to `NonhydrostaticModel` via the
`extended_tracers` keyword argument.

Replication is only possible in `Periodic` directions, since it relies on the periodic
extension of the velocity field: `east + west` must be zero unless `x` is `Periodic`, and
likewise `north + south` for `y`.

The extended tracers are advected by the model velocities and mixed by the model closure,
both wrapped periodically onto the extended domain, matching the SGS scalar flux
`πᵪ = -(νₜ / Scₜ) ∇χ` of the reference. Passing `advection` or `closure` overrides the
model's; leaving them `nothing` inherits the model's.

Keyword arguments
=================

- `east`, `west`, `north`, `south`: number of replications of the model domain in each
  direction. Default: `0`.

- `advection`: tracer advection scheme. Default: `nothing`, meaning the model's.

- `closure`: turbulence closure. Default: `nothing`, meaning the model's.

- `forcing`: `NamedTuple` of forcing functions, used for example to represent a localized
  release of tracer. Default: `NamedTuple()`.

- `boundary_conditions`: `NamedTuple` of `FieldBoundaryConditions` for the extended tracers.
  Default: `NamedTuple()`.

Example
=======

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(16, 16, 8), extent=(500, 500, 300));

julia> model = NonhydrostaticModel(grid; extended_tracers = ExtendedTracers(:c, east=1, west=1))
NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 16×16×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── timestepper: RungeKutta3TimeStepper
├── advection scheme: Centered(order=2)
├── tracers: ()
├── closure: Nothing
├── buoyancy: Nothing
├── coriolis: Nothing
└── extended tracers: c on a 48×16×8 grid (east=1, west=1, north=0, south=0)
```
"""
struct ExtendedTracers{N, G, C, T, A, K, F, BF, V, AF, KF, I, B}
                  names :: N
                   east :: Int
                   west :: Int
                  north :: Int
                  south :: Int
    boundary_conditions :: B
                   grid :: G
                tracers :: C
                     Gⁿ :: T
                     G⁻ :: T
              advection :: A
                closure :: K
                forcing :: F
      background_fields :: BF
             velocities :: V
       auxiliary_fields :: AF
         closure_fields :: KF
           immersed_bcs :: I
end

function ExtendedTracers(names...; east = 0, west = 0, north = 0, south = 0,
                                   advection = nothing,
                                   closure = nothing,
                                   forcing = NamedTuple(),
                                   boundary_conditions = NamedTuple())

    names = tupleit(length(names) == 1 ? first(names) : names)

    all(name isa Symbol for name in names) ||
        throw(ArgumentError("ExtendedTracers names must be Symbols; got $names"))

    for (dir, n) in pairs((; east, west, north, south))
        (n isa Integer && n >= 0) ||
            throw(ArgumentError("The number of extended tracer domains to the $dir must be a non-negative Integer; got $n"))
    end

    return ExtendedTracers(names, east, west, north, south, boundary_conditions,
                           nothing, nothing, nothing, nothing,
                           advection, closure, forcing,
                           nothing, nothing, nothing, nothing, nothing)
end

# Allows `model.extended_tracers.c` to reach the tracer named `c`.
function Base.getproperty(et::ExtendedTracers, name::Symbol)
    hasfield(typeof(et), name) && return getfield(et, name)
    return getproperty(getfield(et, :tracers), name)
end

Base.propertynames(et::ExtendedTracers) = (fieldnames(typeof(et))..., getfield(et, :names)...)

#####
##### Interface extended by model implementations, with no-ops for models without
##### extended tracers. See src/Models/NonhydrostaticModels/nonhydrostatic_extended_tracers.jl
#####

function materialize_extended_tracers end

materialize_extended_tracers(::Nothing, args...; kw...) = nothing

compute_extended_tracer_tendencies!(::Nothing, model) = nothing
ab2_step_extended_tracers!(::Nothing, model, Δt, χ) = nothing
rk3_substep_extended_tracers!(::Nothing, model, Δt, γⁿ, ζⁿ) = nothing
cache_previous_extended_tracer_tendencies!(::Nothing, model) = nothing

function extension_summary(et::ExtendedTracers)
    names = getfield(et, :names)
    grid = getfield(et, :grid)
    domain = isnothing(grid) ? "an unmaterialized grid" : string(size(grid, 1), "×", size(grid, 2), "×", size(grid, 3), " grid")
    return string(join(names, ", "), " on a ", domain,
                  " (east=", et.east, ", west=", et.west, ", north=", et.north, ", south=", et.south, ")")
end

Base.summary(et::ExtendedTracers) = string("ExtendedTracers: ", extension_summary(et))
Base.show(io::IO, et::ExtendedTracers) = print(io, summary(et))

end # module
