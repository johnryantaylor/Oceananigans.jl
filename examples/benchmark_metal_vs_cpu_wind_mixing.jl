# Benchmark: ocean wind mixing and convection on the CPU vs an Apple Metal GPU.
#
# The physical setup mirrors the wind- and convection-driven mixing examples: a wind stress,
# surface cooling, and an evaporative salt flux applied to a stratified ocean surface boundary
# layer, with `WENO` advection and the `AnisotropicMinimumDissipation` closure.
#
# The CPU run uses `Float64` (as in `ocean_wind_mixing_and_convection.jl`) and the Metal run
# uses `Float32` (required by Metal GPUs). Each configuration is run twice for a fixed number
# of iterations with no output writers or plotting; the second pass is timed after
# compilation.

using Oceananigans
using Oceananigans.Units
using Metal
using Printf
using Random
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

const Nx = 128
const Ny = 128
const Nz = 64

const Lx = 128
const Ly = 128
const Lz = 64

const benchmark_iterations = 100
const Δt = 2 # seconds, fixed (no time-step wizard) so both runs do identical work

const ρₒ = 1026
const dTdz = 0.01
const Jᵀ = 200 / (ρₒ * 3991)
const τx = - 1.2 / ρₒ * 2e-3 * 10 * abs(10)
const evaporation_rate = 1e-3 / hour

@inline Jˢ(x, y, t, S, evaporation_rate) = - evaporation_rate * S

refinement = 1.2
stretching = 12
h(k) = (k - 1) / Nz
ζ₀(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))
z_interfaces(k) = Lz * (ζ₀(k) * Σ(k) - 1)

function make_simulation(architecture)
    grid = RectilinearGrid(architecture,
                           size = (Nx, Ny, Nz),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = z_interfaces)

    equation_of_state = TEOS10EquationOfState(reference_density=ρₒ)
    buoyancy = SeawaterBuoyancy(; equation_of_state)

    T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᵀ),
                                    bottom = GradientBoundaryCondition(dTdz))
    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx))
    evap = convert(eltype(grid), evaporation_rate)
    S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Jˢ, field_dependencies=:S,
                                                               parameters=evap))

    model = NonhydrostaticModel(grid; buoyancy,
                                advection = WENO(order=7),
                                tracers = (:T, :S),
                                coriolis = FPlane(f=1e-4),
                                closure = AnisotropicMinimumDissipation(),
                                boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs))

    Random.seed!(1969)
    Ξ(z) = randn() * z / Lz * (1 + z / Lz)
    Tᵢ(x, y, z) = 20 + dTdz * z + dTdz * Lz * 2e-6 * Ξ(z)
    uᵢ(x, y, z) = sqrt(abs(τx)) * 1e-3 * Ξ(z)
    set!(model, u=uᵢ, w=uᵢ, T=Tᵢ, S=35)

    return Simulation(model, Δt=Δt, stop_iteration=benchmark_iterations)
end

function timed_run(label, architecture; nruns=2)
    times = Float64[]

    for run in 1:nruns
        @info "$label: building simulation (run $run/$nruns)..."
        simulation = make_simulation(architecture)
        @info "$label: running $benchmark_iterations iterations (run $run/$nruns)..."
        run!(simulation)
        push!(times, simulation.run_wall_time)
        @info "$label run $run: wall time = $(prettytime(simulation.run_wall_time)), " *
              "per step = $(prettytime(simulation.run_wall_time / benchmark_iterations))"
    end

    return times
end

cpu_times = timed_run("CPU (Float64)", CPU())

metal_times = if Metal.functional()
    Oceananigans.defaults.FloatType = Float32
    timed_run("Metal (Float32)", GPU(Metal.MetalBackend()))
else
    @warn "Metal is not functional on this machine; skipping the GPU run."
    nothing
end

cpu_fast = cpu_times[2]
cpu_per_step = cpu_fast / benchmark_iterations

println()
println("="^72)
println("Wind mixing and convection: CPU vs Metal ($(Nx)×$(Ny)×$(Nz), $benchmark_iterations steps)")
println("="^72)
println(@sprintf("CPU   (Float64)  run 1: %-10s  run 2: %-10s  per step: %s",
                 prettytime(cpu_times[1]), prettytime(cpu_fast), prettytime(cpu_per_step)))

if metal_times !== nothing
    metal_fast = metal_times[2]
    metal_per_step = metal_fast / benchmark_iterations
    speedup = cpu_fast / metal_fast
    println(@sprintf("Metal (Float32)  run 1: %-10s  run 2: %-10s  per step: %s",
                     prettytime(metal_times[1]), prettytime(metal_fast), prettytime(metal_per_step)))
    println()
    println(@sprintf("Speedup (CPU / Metal, second run): %.1f×", speedup))
end
println("="^72)
