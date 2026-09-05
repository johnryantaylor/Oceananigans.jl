# # Plume dispersion on a large domain (baseline for ENDLESS)
#
# Counterpart to [`endless_plume_dispersion`](@ref endless_plume_dispersion): the same
# plume release and mean eastward flow, but with the full nonhydrostatic model on a grid
# matching the extended tracer domain (``3L × 3L``, ``384²``). Momentum, pressure and
# tracer are all evolved on that large grid.

using Oceananigans
using Random
using Statistics

Random.seed!(43)

L = 2π

## Match ExtendedTracers(:c, east=2, west=0, north=1, south=1) on a 128² base grid:
## Rx = west + 1 + east = 3, Ry = south + 1 + north = 3
Nx = 128 * 3
Ny = 128 * 3
grid = RectilinearGrid(size=(Nx, Ny), x=(0, 3L), y=(-L, 2L),
                       topology=(Periodic, Periodic, Flat))

source_radius = L / 20

@inline plume_source(x, y, t) =
    ifelse((x - L/2)^2 + (y - L/2)^2 < source_radius^2, 1.0, 0.0)

model = NonhydrostaticModel(grid;
                            advection = WENO(order=5),
                            tracers = :c,
                            forcing = (; c = plume_source),
                            closure = ScalarDiffusivity(ν=1e-5, κ=1e-5))

U = 0.25

u, v, w = model.velocities

uᵢ = rand(size(u)...)
vᵢ = rand(size(v)...)

uᵢ .+= U .- mean(uᵢ)
vᵢ .-= mean(vᵢ)

set!(model, u=uᵢ, v=vᵢ)

simulation = Simulation(model, Δt=0.2, stop_time=48)

conjure_time_step_wizard!(simulation, IterationInterval(10), cfl=0.7, max_Δt=0.5)

progress(sim) = @info string("Iteration: ", iteration(sim), ", time: ", time(sim))
add_callback!(simulation, progress, IterationInterval(100))

c = model.tracers.c
ω = ∂x(v) - ∂y(u)

filename = "large_domain_plume_dispersion"

simulation.output_writers[:fields] = JLD2Writer(model, (; ω, c);
                                                filename,
                                                schedule = TimeInterval(0.5),
                                                overwrite_existing = true)

run!(simulation)

@info "Large-domain run wall time: $(run_wall_time(simulation))"
