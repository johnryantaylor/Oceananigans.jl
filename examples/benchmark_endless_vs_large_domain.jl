# Benchmark: ENDLESS extended tracers vs a matched large-domain NonhydrostaticModel.
#
# Times only `run!` wall time (`simulation.run_wall_time`), with no output writers or
# plotting. Each configuration is run twice; the second pass is the fair comparison
# after compilation.

using Oceananigans
using Random
using Statistics

const L = 2π
const U = 0.25
const stop_time = 48
const source_radius = L / 20

@inline plume_source(x, y, t) =
    ifelse((x - L/2)^2 + (y - L/2)^2 < source_radius^2, 1.0, 0.0)

function random_horizontal_velocities!(model)
    Random.seed!(43)
    u, v, w = model.velocities
    uᵢ = rand(size(u)...)
    vᵢ = rand(size(v)...)
    uᵢ .+= U .- mean(uᵢ)
    vᵢ .-= mean(vᵢ)
    set!(model, u=uᵢ, v=vᵢ)
    return nothing
end

function make_endless_simulation()
    grid = RectilinearGrid(size=(128, 128), extent=(L, L), topology=(Periodic, Periodic, Flat))

    extended_tracers = ExtendedTracers(:c, east=2, west=0, north=1, south=1,
                                       advection = WENO(order=5),
                                       forcing = (; c = plume_source))

    model = NonhydrostaticModel(grid;
                                advection = UpwindBiased(order=5),
                                closure = ScalarDiffusivity(ν=1e-5, κ=1e-5),
                                extended_tracers)

    random_horizontal_velocities!(model)

    simulation = Simulation(model; Δt=0.2, stop_time)
    conjure_time_step_wizard!(simulation, IterationInterval(10), cfl=0.7, max_Δt=0.5)
    return simulation
end

function make_large_domain_simulation()
    ## Match ExtendedTracers(:c, east=2, west=0, north=1, south=1) on a 128² base grid
    Nx = 128 * 3
    Ny = 128 * 3
    grid = RectilinearGrid(size=(Nx, Ny), x=(0, 3L), y=(-L, 2L),
                           topology=(Periodic, Periodic, Flat))

    model = NonhydrostaticModel(grid;
                                advection = WENO(order=5),
                                tracers = :c,
                                forcing = (; c = plume_source),
                                closure = ScalarDiffusivity(ν=1e-5, κ=1e-5))

    random_horizontal_velocities!(model)

    simulation = Simulation(model; Δt=0.2, stop_time)
    conjure_time_step_wizard!(simulation, IterationInterval(10), cfl=0.7, max_Δt=0.5)
    return simulation
end

function timed_run(label, make_simulation; nruns=2)
    times = Float64[]
    iterations = Int[]

    for run in 1:nruns
        @info "$label: building simulation (run $run/$nruns)..."
        simulation = make_simulation()
        @info "$label: running to t = $stop_time (run $run/$nruns)..."
        run!(simulation)
        push!(times, simulation.run_wall_time)
        push!(iterations, iteration(simulation))
        @info "$label run $run: wall time = $(prettytime(simulation.run_wall_time)), " *
              "iterations = $(iteration(simulation)), " *
              "per step = $(prettytime(simulation.run_wall_time / iteration(simulation)))"
    end

    return (; times, iterations)
end

@info "=== ENDLESS (128² momentum + 384² tracer) ==="
endless = timed_run("ENDLESS", make_endless_simulation)

@info "=== Large domain (384² momentum + tracer) ==="
large = timed_run("Large domain", make_large_domain_simulation)

endless_fast = endless.times[2]
large_fast = large.times[2]
speedup = large_fast / endless_fast

println()
println("="^72)
println("Simulation wall-time comparison (second runs; excludes plotting/I/O)")
println("="^72)
println("ENDLESS      run 1: $(prettytime(endless.times[1]))  |  run 2: $(prettytime(endless_fast))  |  $(endless.iterations[2]) steps")
println("Large domain run 1: $(prettytime(large.times[1]))  |  run 2: $(prettytime(large_fast))  |  $(large.iterations[2]) steps")
println()
println("Speedup (large / ENDLESS, second run): $(round(speedup; digits=2))×")
println("ENDLESS is $(round(speedup; digits=2))× faster than the matched large-domain model.")
println("="^72)
