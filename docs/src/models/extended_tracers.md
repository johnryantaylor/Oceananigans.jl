# Extended tracers (ENDLESS)

`NonhydrostaticModel` can evolve passive tracers on a horizontal domain much larger than the one
carrying the velocity, density and pressure, following the ENDLESS approach of
[Chen2016ENDLESS](@citet).

The idea rests on a property of horizontally-periodic large eddy simulation: if the velocity and
pressure fields are replicated in ``x`` and ``y``, the replicated fields still satisfy mass and
momentum conservation. A passive tracer may therefore be advected over an "extended domain" tiled
with copies of the LES box, which lets a plume disperse to scales far beyond the box while the
momentum solve stays small. This is the regime of, for example, a subsea oil release, where the
plume grows to tens of kilometers but Langmuir turbulence must be resolved at ``\mathcal{O}(10)``
meter spacing.

The extended tracer obeys

```math
\partial_t \tilde{\chi} + \boldsymbol{\nabla} \boldsymbol{\cdot} \left ( \tilde{\boldsymbol{u}} \tilde{\chi} \right ) = - \boldsymbol{\nabla} \boldsymbol{\cdot} \boldsymbol{\pi}_\chi + Q_\chi ,
```

where ``\tilde{\boldsymbol{u}}`` is the periodic extension of the resolved velocity,
``\boldsymbol{\pi}_\chi`` is the subgrid-scale flux supplied by the model's closure, and
``Q_\chi`` is an optional source term supplied as a forcing.

## Usage

Pass an `ExtendedTracers` to the `extended_tracers` keyword argument, saying how many times to
replicate the model domain in each direction:

```@meta
DocTestSetup = quote
    using Oceananigans
end
```

```jldoctest endless
grid = RectilinearGrid(size=(16, 16, 8), extent=(500, 500, 300))

model = NonhydrostaticModel(grid; closure = SmagorinskyLilly(),
                            extended_tracers = ExtendedTracers(:c, east=2, west=2, north=1, south=1))

# output
NonhydrostaticModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 16×16×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── timestepper: RungeKutta3TimeStepper
├── advection scheme: Centered(order=2)
├── tracers: ()
├── closure: Smagorinsky with coefficient = LillyCoefficient(smagorinsky = 0.16, reduction_factor = 1.0), Pr=NamedTuple()
├── buoyancy: Nothing
├── coriolis: Nothing
└── extended tracers: c on a 80×48×8 grid (east=2, west=2, north=1, south=1)
```

Each extended tracer is an ordinary `Field` on the extended grid, so `set!`, output writers and
`AbstractOperations` all work as usual:

```jldoctest endless
model.extended_tracers.c

# output
80×48×8 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 80×48×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
└── data: 86×54×14 OffsetArray(::Array{Float64, 3}, -2:83, -2:51, -2:11) with eltype Float64 with indices -2:83×-2:51×-2:11
    └── max=0.0, min=0.0, mean=0.0
```

The original domain sits in the tile it started in, so a point at ``(x, y)`` in the model grid is
the same physical point in the extended grid.

## Restrictions

Replication relies on the periodic extension of the velocity field, so a direction may only be
extended if it is `Periodic`. A channel (`Periodic`, `Bounded`, `Bounded`) can be extended in
``x`` but not in ``y``. The model grid must be a `RectilinearGrid`, and extended tracer names must
differ from the model's tracer names.

The outer edges of the extended domain are `Periodic`, which is the treatment consistent with the
replicated velocity field: it conserves tracer exactly, and the tracer wraps only after spreading
across the entire extended domain. Unlike [Chen2016ENDLESS](@citet), the tracer domain is not extended
adaptively; the number of replications is fixed when the model is built.

## Advection, diffusion and forcing

By default the extended tracers inherit the model's advection scheme and closure. The closure's
diffusivity fields are periodically extended along with the velocities, which reproduces the
subgrid-scale scalar flux ``\boldsymbol{\pi}_\chi = -(\nu_t / \mathrm{Sc}_t) \boldsymbol{\nabla} \chi``
of the reference. Both may be overridden:

```julia
ExtendedTracers(:c, east=1, west=1, advection=WENO(order=5), closure=ScalarDiffusivity(κ=1e-5))
```

A localized release is expressed as a forcing, exactly as for ordinary tracers:

```julia
plume_source(x, y, z, t) = ifelse((x - x₀)^2 + (y - y₀)^2 < r², Q, zero(Q))

ExtendedTracers(:c, east=2, west=2, forcing=(; c=plume_source))
```

## Cost

Extended tracers cost memory but almost no extra velocity storage. With ``R_x = \mathrm{west} + 1 + \mathrm{east}``
and ``R_y = \mathrm{south} + 1 + \mathrm{north}``, each extended tracer allocates ``R_x R_y`` times
the size of an ordinary tracer field, three times over: the tracer itself and its two tendency
fields. The velocities and diffusivity fields are *not* copied — they are read through a
zero-memory view that wraps base-grid indices periodically — so the replication factors are the
only memory knob to watch on a GPU.

```@meta
DocTestSetup = nothing
```
