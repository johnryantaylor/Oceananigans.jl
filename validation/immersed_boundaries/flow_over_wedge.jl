pushfirst!(LOAD_PATH, joinpath("..", ".."))

using Oceananigans
using Oceananigans: fields
using Oceananigans.Grids: on_architecture
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, GridFittedBoundary, mask_immersed_field!
using Oceananigans.Architectures: device

using KernelAbstractions: MultiEvent
using Printf
using GLMakie

arch = CPU()
Nz = 64 # Resolution
κ = 1e-6 # Diffusivity and viscosity (Prandtl = 1)
U = 1

underlying_grid = RectilinearGrid(arch,
                                  size = (2Nz, Nz),
                                  x = (-4, 4),
                                  z = (0, 4),
                                  halo = (3, 3),
                                  topology = (Periodic, Flat, Bounded))

const slope = 1/2
@inline wedge(x, y) = slope * min(x, -x) + 1
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(wedge))

model = NonhydrostaticModel(grid = grid,
                            advection = UpwindBiasedFifthOrder(),
                            closure = IsotropicDiffusivity(ν=κ, κ=κ),
                            tracers = nothing,
                            coriolis = nothing,
                            buoyancy = nothing)

# Linear stratification
set!(model, u = U)

start_time = [time_ns()]

function x_momentum(u)
    mask_immersed_field!(u, NaN)
    u_cpu = Array(interior(u))
    fluid_u = filter(isfinite, u_cpu[:])
    mask_immersed_field!(u)
    return sum(fluid_u)
end

Δt = 5e-2 * (grid.Lx / grid.Nz) / U
simulation = Simulation(model, Δt = Δt, stop_time = 10)

u, v, w = model.velocities
ΣUi = x_momentum(u)

function progress(s)
    # Calculate total momentum, and change in total momentum
    ΣU = x_momentum(u)
    ΔU = (ΣU - ΣUi) / ΣU

    @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, x-momentum: %.2e, ΔU: %.1f %%",
                   100 * s.model.clock.time / s.stop_time,
                   s.model.clock.iteration,
                   s.model.clock.time,
                   ΣU,
                   ΔU * 100)

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

prefix = "flow_over_wedge_Nz$(Nz)_high_Re"
simulation.output_writers[:fields] = JLD2OutputWriter(model, model.velocities,
                                                      schedule = TimeInterval(0.1),
                                                      prefix = prefix,
                                                      field_slicer = nothing,
                                                      force = true)
                        
run!(simulation)

@info """
    Simulation complete.
    Runtime: $(prettytime(simulation.run_wall_time))
"""

filepath = prefix * ".jld2"
ut = FieldTimeSeries(filepath, "u", grid=grid)
wt = FieldTimeSeries(filepath, "w", grid=grid)

times = ut.times
Nt = length(times)

ut = [ut[n] for n = 1:Nt]
wt = [wt[n] for n = 1:Nt]

# Preprocess
events = []
for n = 1:Nt
    for f in (ut[n], wt[n])
        push!(events, mask_immersed_field!(f, NaN))
    end
end

wait(device(CPU()), MultiEvent(Tuple(events)))

max_u = 2.0
min_u = 0.5
max_w = 1.0

n = Node(1)

ui(n) = interior(ut[n])[:, 1, :]
wi(n) = interior(wt[n])[:, 1, :]

fluid_u(n) = filter(isfinite, ui(n)[:])
fluid_w(n) = filter(isfinite, wi(n)[:])

ΣUt = [sum(fluid_u(n)) for n = 1:Nt]

up = @lift ui($n)
wp = @lift wi($n)

fig = Figure(resolution=(1800, 1200))

ax = Axis(fig[1, 1], title="x-velocity")
hm = heatmap!(ax, up, colorrange=(min_u, max_u), colormap=:thermal)
cb = Colorbar(fig[1, 2], hm)

ax = Axis(fig[2, 1], title="z-velocity")
hm = heatmap!(ax, wp, colorrange=(-max_w, max_w), colormap=:balance)
cb = Colorbar(fig[2, 2], hm)

title_gen(n) = @sprintf("Flow over wedge (Δ∫u = %.1f %%) at t = %.2f", 100 * (ΣUt[n] - ΣUi) / ΣUi, times[n])
title_str = @lift title_gen($n)
ax_t = fig[0, :] = Label(fig, title_str)

record(fig, prefix * ".mp4", 1:Nt, framerate=8) do nt
    n[] = nt
end

display(fig)

