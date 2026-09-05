# # [Plume dispersion on an extended domain (ENDLESS)](@id endless_plume_dispersion)
#
# In this example we release a passive tracer from a localized source into two-dimensional
# turbulence with a mean eastward flow, and let it stream out of the domain that carries
# the flow. This is the ENDLESS approach of [Chen2016ENDLESS](@citet): because the velocity
# field is horizontally periodic, replicating it in ``x`` and ``y`` produces a larger
# velocity field that still satisfies mass and momentum conservation, so a passive tracer
# may be advected across the replicated domain.
#
# In a plain periodic simulation the plume would wrap around and collide with itself after
# travelling one domain length. Here it does not: it keeps going into the replicated tiles.
#
# This example demonstrates:
#
#   * How to evolve a passive tracer on a domain larger than the model grid.
#   * How to extend the tracer domain asymmetrically, in the direction a plume travels.
#   * How to release tracer from a localized source with a forcing function.
#   * How to visualize a tracer that lives on a different grid than the velocities.

# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, CairoMakie"
# ```

# ## Model setup
#
# We use a doubly-periodic two-dimensional grid, which is the setting ENDLESS requires:
# replication is only possible in `Periodic` directions.

using Oceananigans
using Random

Random.seed!(43)

L = 2π
grid = RectilinearGrid(size=(128, 128), extent=(L, L), topology=(Periodic, Periodic, Flat))

# The tracer is released from a small patch at the center of the velocity domain. The
# source is the term ``Q_\chi`` of the tracer equation, expressed as an ordinary forcing.

source_radius = L / 20

@inline plume_source(x, y, t) =
    ifelse((x - L/2)^2 + (y - L/2)^2 < source_radius^2, 1.0, 0.0)

# `ExtendedTracers` says how many times to replicate the model domain in each direction.
# The plume travels east, so we extend twice to the east and once to either side, giving a
# tracer domain nine times the area of the flow domain while the momentum solve stays on
# the original 128² grid.

extended_tracers = ExtendedTracers(:c, east=2, west=0, north=1, south=1,
                                   advection = WENO(order=5),
                                   forcing = (; c = plume_source))

model = NonhydrostaticModel(grid;
                            advection = UpwindBiased(order=5),
                            closure = ScalarDiffusivity(ν=1e-5, κ=1e-5),
                            extended_tracers)

# Note that the extended tracer lives on its own, larger grid:

model.extended_tracers.grid

# ## Random initial conditions
#
# We seed the flow with random velocities as in the
# [two-dimensional turbulence example](@ref two_dimensional_turbulence), except that `u`
# is given a mean eastward component rather than zero mean. A uniform mean flow is
# preserved exactly by a periodic model, and it is what carries the plume out of the
# velocity domain and into the replicated tiles.

using Statistics

U = 0.25 # mean eastward flow

u, v, w = model.velocities

uᵢ = rand(size(u)...)
vᵢ = rand(size(v)...)

uᵢ .+= U .- mean(uᵢ)
vᵢ .-= mean(vᵢ)

set!(model, u=uᵢ, v=vᵢ)

# ## Running the simulation

simulation = Simulation(model, Δt=0.2, stop_time=48)

conjure_time_step_wizard!(simulation, IterationInterval(10), cfl=0.7, max_Δt=0.5)

progress(sim) = @info string("Iteration: ", iteration(sim), ", time: ", time(sim))
add_callback!(simulation, progress, IterationInterval(100))

# We output the extended tracer, which is an ordinary `Field` and so needs no special
# treatment, alongside the vorticity of the flow that stirs it.

c = model.extended_tracers.c
ω = ∂x(v) - ∂y(u)

filename = "endless_plume_dispersion"

simulation.output_writers[:fields] = JLD2Writer(model, (; ω, c);
                                                filename,
                                                schedule = TimeInterval(0.5),
                                                overwrite_existing = true)

run!(simulation)

# ## Visualizing the results
#
# The tracer covers ``[0, 3L] \times [-L, 2L]`` while the vorticity covers ``[0, L]^2``. We
# outline the velocity domain on the tracer plot to show how far the plume has travelled
# beyond it.

ω_timeseries = FieldTimeSeries(filename * ".jld2", "ω")
c_timeseries = FieldTimeSeries(filename * ".jld2", "c")

times = ω_timeseries.times
nothing #hide

using CairoMakie
set_theme!(Theme(fontsize = 20))

fig = Figure(size = (1000, 520))

ax_ω = Axis(fig[2, 1]; xlabel="x", ylabel="y", title="Vorticity (velocity domain)",
            limits=((0, L), (0, L)), aspect=AxisAspect(1))

ax_c = Axis(fig[2, 2]; xlabel="x", ylabel="y", title="Tracer (extended domain)",
            limits=((0, 3L), (-L, 2L)), aspect=AxisAspect(1))

n = Observable(1)

ω = @lift ω_timeseries[$n]
c = @lift c_timeseries[$n]

heatmap!(ax_ω, ω; colormap=:balance, colorrange=(-2, 2))
heatmap!(ax_c, c; colormap=:matter, colorrange=(0, 2))

## Outline the velocity domain within the extended domain
lines!(ax_c, [0, L, L, 0, 0], [0, 0, L, L, 0]; color=:black, linewidth=2, linestyle=:dash)

title = @lift "t = " * string(round(times[$n], digits=2))
Label(fig[1, 1:2], title, fontsize=24, tellwidth=false)

current_figure() #hide
fig

# And now we make a movie of the plume escaping the domain that stirs it.

frames = 1:length(times)

@info "Making an animation of a plume dispersing beyond its velocity domain..."

record(fig, filename * ".mp4", frames, framerate=24) do i
    n[] = i
end
nothing #hide

# ![](endless_plume_dispersion.mp4)

# By the end of the simulation roughly half the tracer has left the box that carries the
# flow. Note also how much of the extended domain the plume never reaches: that unused
# memory is what motivates the adaptive domain of [Chen2016ENDLESS](@citet), which
# activates and deactivates tiles as the plume moves. Here the number of tiles is fixed, so
# the extended domain should be chosen just large enough for the plume you expect.
