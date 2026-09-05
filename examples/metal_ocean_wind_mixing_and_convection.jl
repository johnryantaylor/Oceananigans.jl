# # [Wind- and convection-driven mixing on an Apple Metal GPU](@id metal_gpu_example)
#
# This example reproduces the [ocean wind mixing and convection example](@ref gpu_example)
# but runs on an Apple silicon GPU through [Metal.jl](https://github.com/JuliaGPU/Metal.jl).
# It demonstrates:
#
#   * How to select the Metal GPU backend with `GPU(Metal.MetalBackend())`.
#   * How to run a simulation in single precision, which Metal GPUs require.
#
# Apart from the architecture and the float type, the physical setup is identical to the
# CUDA version: a 128²×64 large eddy simulation of an ocean surface boundary layer driven
# by surface cooling, an evaporative salt flux, and a wind stress.

# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add Oceananigans, CairoMakie, SeawaterPolynomials, Metal"
# ```

using Oceananigans
using Oceananigans.Units

using CairoMakie
using Metal
using Printf
using Random
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

Random.seed!(1969) # for reproducible results

# ## Single precision
#
# Metal GPUs only support 32-bit floating point arithmetic, so we set the default float type
# to `Float32`. Every grid, field, and model built afterwards then uses single precision.

Oceananigans.defaults.FloatType = Float32

# ## The grid
#
# We build the grid on the Metal GPU by passing `GPU(Metal.MetalBackend())` as the
# architecture. The vertical spacing is stretched to keep relatively constant resolution in
# the mixed layer, exactly as in the CUDA example.

architecture = GPU(Metal.MetalBackend())

Nx = Ny = 128    # number of points in each of horizontal directions
Nz = 64          # number of points in the vertical direction

Lx = Ly = 128    # (m) domain horizontal extents
Lz = 64          # (m) domain depth

refinement = 1.2 # controls spacing near surface (higher means finer spaced)
stretching = 12  # controls rate of stretching at bottom

## Normalized height ranging from 0 to 1
h(k) = (k - 1) / Nz

## Linear near-surface generator
ζ₀(k) = 1 + (h(k) - 1) / refinement

## Bottom-intensified stretching function
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

## Generating function
z_interfaces(k) = Lz * (ζ₀(k) * Σ(k) - 1)

grid = RectilinearGrid(architecture,
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = z_interfaces)

# We plot vertical spacing versus depth to inspect the prescribed grid stretching. We
# evaluate the generating function directly on the host rather than reading spacings back
# from the GPU grid:

Δz_column = [z_interfaces(k + 1) - z_interfaces(k) for k in 1:Nz]
z_column = [(z_interfaces(k + 1) + z_interfaces(k)) / 2 for k in 1:Nz]

fig = Figure(size=(1200, 800))
ax = Axis(fig[1, 1], xlabel = "Vertical spacing (m)", ylabel = "z (m)")

lines!(ax, Δz_column, z_column)
scatter!(ax, Δz_column, z_column)

current_figure() #hide
fig

# ## Buoyancy that depends on temperature and salinity
#
# We use the `SeawaterBuoyancy` model with the TEOS10 equation of state,

ρₒ = 1026 # kg m⁻³, average density at the surface of the world ocean
equation_of_state = TEOS10EquationOfState(reference_density=ρₒ)
buoyancy = SeawaterBuoyancy(; equation_of_state)

# ## Boundary conditions
#
# We calculate the surface temperature flux associated with surface cooling of
# 200 W m⁻², reference density `ρₒ`, and heat capacity `cᴾ`,

Q = 200   # W m⁻², surface _heat_ flux
cᴾ = 3991 # J K⁻¹ kg⁻¹, typical heat capacity for seawater

Jᵀ = Q / (ρₒ * cᴾ) # K m s⁻¹, surface _temperature_ flux

# We impose a temperature gradient `dTdz` both initially and at the bottom of the domain:

dTdz = 0.01 # K m⁻¹

T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᵀ),
                                bottom = GradientBoundaryCondition(dTdz))

# For the velocity field, a wind blowing over the ocean surface with average velocity `u₁₀`
# at 10 meters exerts a kinematic stress that we estimate with a drag coefficient `cᴰ`:

u₁₀ = 10  # m s⁻¹, average wind velocity 10 meters above the ocean
cᴰ = 2e-3 # dimensionless drag coefficient
ρₐ = 1.2  # kg m⁻³, approximate average density of air at sea-level
τx = - ρₐ / ρₒ * cᴰ * u₁₀ * abs(u₁₀) # m² s⁻²

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx))

# For salinity, `S`, we impose an evaporative flux of the form

@inline Jˢ(x, y, t, S, evaporation_rate) = - evaporation_rate * S # [salinity unit] m s⁻¹
nothing #hide

# with an evaporation rate of 1 millimeter per hour. Parameters passed to a boundary
# condition function are used verbatim inside GPU kernels, so on Metal they must already be
# single precision:

evaporation_rate = convert(eltype(grid), 1e-3 / hour) # m s⁻¹
evaporation_bc = FluxBoundaryCondition(Jˢ, field_dependencies=:S, parameters=evaporation_rate)
S_bcs = FieldBoundaryConditions(top=evaporation_bc)

# ## Model instantiation
#
# We use `WENO` advection together with the `AnisotropicMinimumDissipation` closure for the
# subfilter stress. The `DynamicSmagorinsky` closure used in the CUDA version of this example
# relies on a cube-root operation that Metal cannot yet compile in single precision;
# `AnisotropicMinimumDissipation` is a self-tuning large eddy simulation closure that runs on
# Metal and, like `DynamicSmagorinsky`, requires no hand-tuned coefficient.

model = NonhydrostaticModel(grid; buoyancy,
                            advection = CenteredSecondOrder(),
                            tracers = (:T, :S),
                            coriolis = FPlane(f=1e-4),
                            closure = AnisotropicMinimumDissipation(),
                            boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs))

# ## Initial conditions
#
# Temperature is initialized with a linear stratification plus random noise damped at the
# walls; velocity is initialized with noise scaled by the friction velocity.

## Random noise damped at top and bottom
Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise

## Temperature initial condition: a stable density gradient with random noise superposed.
Tᵢ(x, y, z) = 20 + dTdz * z + dTdz * model.grid.Lz * 2e-6 * Ξ(z)

## Velocity initial condition: random noise scaled by the friction velocity.
uᵢ(x, y, z) = sqrt(abs(τx)) * 1e-3 * Ξ(z)

set!(model, u=uᵢ, w=uᵢ, T=Tᵢ, S=35)

# ## Setting up a simulation
#
# We set up a simulation with an initial time-step of 10 seconds that stops after 2 hours,
# with adaptive time-stepping and progress printing.

simulation = Simulation(model, Δt=10, stop_time=2hours)

conjure_time_step_wizard!(simulation, cfl=0.7)

## Print a progress message
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|w|) = %.1e ms⁻¹, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.w), prettytime(sim.run_wall_time))

add_callback!(simulation, progress_message, IterationInterval(200))

# ## Output
#
# We use the `JLD2Writer` to save ``x, z`` slices of the velocity and tracer fields plus the
# eddy viscosity.

eddy_viscosity = (; νₑ = model.closure_fields.νₑ)

filename = "metal_ocean_wind_mixing_and_convection"

simulation.output_writers[:slices] =
    JLD2Writer(model, merge(model.velocities, model.tracers, eddy_viscosity),
               filename = filename * ".jld2",
               indices = (:, grid.Ny/2, :),
               schedule = TimeInterval(1minute),
               overwrite_existing = true)

## Fail the docs build if this simulation produces NaNs #hide
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation) #hide
run!(simulation)

# ## Turbulence visualization
#
# We animate the data saved in `metal_ocean_wind_mixing_and_convection.jld2`.

filepath = filename * ".jld2"

time_series = (w = FieldTimeSeries(filepath, "w"),
               T = FieldTimeSeries(filepath, "T"),
               S = FieldTimeSeries(filepath, "S"),
               νₑ = FieldTimeSeries(filepath, "νₑ"))

times = time_series.w.times
n = Observable(length(times))

 wₙ = @lift time_series.w[$n]
 Tₙ = @lift time_series.T[$n]
 Sₙ = @lift time_series.S[$n]
νₑₙ = @lift time_series.νₑ[$n]

fig = Figure(size = (1800, 900))

axis_kwargs = (xlabel="x (m)",
               ylabel="z (m)",
               aspect = AxisAspect(grid.Lx/grid.Lz),
               limits = ((0, grid.Lx), (-grid.Lz, 0)))

ax_w  = Axis(fig[2, 1]; title = "Vertical velocity", axis_kwargs...)
ax_T  = Axis(fig[2, 3]; title = "Temperature", axis_kwargs...)
ax_S  = Axis(fig[3, 1]; title = "Salinity", axis_kwargs...)
ax_νₑ = Axis(fig[3, 3]; title = "Eddy viscocity", axis_kwargs...)

title = @lift @sprintf("t = %s", prettytime(times[$n]))

 wlims = (-0.05, 0.05)
 Tlims = (19.7, 19.99)
 Slims = (35, 35.005)
νₑlims = (1e-6, 5e-3)

hm_w = heatmap!(ax_w, wₙ; colormap = :balance, colorrange = wlims)
Colorbar(fig[2, 2], hm_w; label = "m s⁻¹")

hm_T = heatmap!(ax_T, Tₙ; colormap = :thermal, colorrange = Tlims)
Colorbar(fig[2, 4], hm_T; label = "ᵒC")

hm_S = heatmap!(ax_S, Sₙ; colormap = :haline, colorrange = Slims)
Colorbar(fig[3, 2], hm_S; label = "g / kg")

hm_νₑ = heatmap!(ax_νₑ, νₑₙ; colormap = :thermal, colorrange = νₑlims)
Colorbar(fig[3, 4], hm_νₑ; label = "m s⁻²")

fig[1, 1:4] = Label(fig, title, fontsize=24, tellwidth=false)

current_figure() #hide
fig

# And now record a movie, starting at ``t = 10`` minutes since things are pretty boring till
# then:

intro = searchsortedfirst(times, 10minutes)
frames = intro:length(times)

@info "Making a motion picture of ocean wind mixing and convection on Metal..."

CairoMakie.record(fig, filename * ".mp4", frames, framerate=8) do i
    n[] = i
end
nothing #hide

# ![](metal_ocean_wind_mixing_and_convection.mp4)
