include("dependencies_for_runtests.jl")

using Oceananigans.ExtendedTracerDomains: extended_grid, tile

@kernel function _copy_from_tiled!(dst, tiled)
    i, j, k = @index(Global, NTuple)
    @inbounds dst[i, j, k] = tiled[i, j, k]
end

@testset "Extended (ENDLESS) tracers" begin
    @info "Testing extended tracers..."

    @testset "Extended grids" begin
        for arch in archs
            grid = RectilinearGrid(arch, size=(8, 8, 4), extent=(1, 1, 1))

            eg = extended_grid(grid, east=1, west=1, north=2, south=0)
            @test size(eg) == (24, 24, 4)
            @test halo_size(eg) == halo_size(grid)
            @test topology(eg) == topology(grid)
            @test eg.Lx ≈ 3 * grid.Lx
            @test eg.Ly ≈ 3 * grid.Ly
            @test eg.Lz ≈ grid.Lz

            # The original domain occupies the tile it started in
            @test minimum(xnodes(eg, Center())) ≈ minimum(xnodes(grid, Center())) - grid.Lx
            @test minimum(ynodes(eg, Center())) ≈ minimum(ynodes(grid, Center()))

            # Spacings are reproduced exactly, including for stretched coordinates
            xᶠ = [0, 0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1]
            stretched = RectilinearGrid(arch, size=(8, 8, 4), x=xᶠ, y=(0, 1), z=(-1, 0),
                                        topology=(Periodic, Periodic, Bounded))
            seg = extended_grid(stretched, east=2, west=1)
            @test size(seg) == (32, 8, 4)
            base_Δx = Array(parent(stretched.Δxᶜᵃᵃ))[1+3:8+3]
            tiled_Δx = Array(parent(seg.Δxᶜᵃᵃ))[1+3+8:8+3+8]
            @test base_Δx == tiled_Δx

            # Flat directions pass through untouched
            flat = RectilinearGrid(arch, size=(8, 4), x=(0, 1), z=(-1, 0),
                                   topology=(Periodic, Flat, Bounded))
            @test size(extended_grid(flat, east=1, west=1)) == (24, 1, 4)
        end
    end

    @testset "Tiled fields" begin
        for arch in archs
            grid = RectilinearGrid(arch, size=(8, 8, 4), extent=(1, 1, 1))
            eg = extended_grid(grid, east=1, west=1, north=1, south=1)

            u = XFaceField(grid)
            set!(u, (x, y, z) -> sin(2π * x) + cos(4π * y) + z)
            fill_halo_regions!(u)

            # The wrap is exercised on the device: no scalar indexing here
            tiled = CenterField(eg)
            launch!(arch, eg, :xyz, _copy_from_tiled!, tiled, tile(u, grid))

            Nx, Ny, Nz = size(grid)
            uᵢ = Array(interior(u))
            expected = [uᵢ[mod1(i, Nx), mod1(j, Ny), k] for i in 1:3Nx, j in 1:3Ny, k in 1:Nz]
            @test Array(interior(tiled)) == expected

            # A Bounded direction must not wrap
            bounded = RectilinearGrid(arch, size=(8, 8, 4), extent=(1, 1, 1),
                                      topology=(Bounded, Periodic, Bounded))
            ub = XFaceField(bounded)
            set!(ub, (x, y, z) -> x)
            fill_halo_regions!(ub)
            tiled_bounded = CenterField(bounded)
            launch!(arch, bounded, :xyz, _copy_from_tiled!, tiled_bounded, tile(ub, bounded))
            @test Array(interior(tiled_bounded)) == Array(interior(ub, 1:8, :, :))
        end
    end

    @testset "Validation" begin
        grid = RectilinearGrid(size=(8, 8, 4), extent=(1, 1, 1))
        channel = RectilinearGrid(size=(8, 8, 4), extent=(1, 1, 1), topology=(Periodic, Bounded, Bounded))

        @test_throws ArgumentError ExtendedTracers(:c, east=-1)
        @test_throws ArgumentError NonhydrostaticModel(channel; extended_tracers=ExtendedTracers(:c, north=1))
        @test_throws ArgumentError NonhydrostaticModel(grid; tracers=:c, extended_tracers=ExtendedTracers(:c, east=1))

        # A channel may still be extended along its periodic direction
        model = NonhydrostaticModel(channel; extended_tracers=ExtendedTracers(:c, east=1, west=1))
        @test size(model.extended_tracers.grid) == (24, 8, 4)
        @test model.extended_tracers.c isa Field
    end

    @testset "Tiled views are zero-copy" begin
        for arch in archs
            grid = RectilinearGrid(arch, size=(8, 8, 4), extent=(1, 1, 1))
            model = NonhydrostaticModel(grid; closure=SmagorinskyLilly(), tracers=:b, buoyancy=BuoyancyTracer(),
                                        extended_tracers=ExtendedTracers(:c, east=1, west=1, north=1, south=1))
            et = model.extended_tracers

            # The velocities, model tracers and closure fields are aliased, never replicated
            for name in (:u, :v, :w)
                @test et.velocities[name].data === model.velocities[name].data
            end
            @test et.auxiliary_fields.b.data === model.tracers.b.data
            @test et.closure_fields.νₑ.data === model.closure_fields.νₑ.data
        end
    end

    @testset "Output writing" begin
        grid = RectilinearGrid(size=(8, 8, 4), extent=(1, 1, 1))
        model = NonhydrostaticModel(grid; extended_tracers=ExtendedTracers(:c, east=1, west=1))
        set!(model.extended_tracers.c, (x, y, z) -> x)

        simulation = Simulation(model, Δt=1e-3, stop_iteration=1)
        filename = "test_extended_tracers_output"
        simulation.output_writers[:c] = JLD2Writer(model, (; c = model.extended_tracers.c);
                                                   filename, schedule=IterationInterval(1), overwrite_existing=true)
        run!(simulation)

        written = FieldTimeSeries(filename * ".jld2", "c")
        @test size(written[1]) == size(model.extended_tracers.c)
        @test interior(written[end]) ≈ interior(model.extended_tracers.c)
        rm(filename * ".jld2", force=true)
    end

    @testset "Time stepping" begin
        for arch in archs
            FT = eltype(RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1)))
            atol = 100 * eps(FT)

            N, L, H = 8, 1000, 200
            uᵢ(x, y, z) =  0.05 * sin(2π * x / L) * cos(2π * y / L) * (1 + z / H)
            vᵢ(x, y, z) = -0.05 * cos(2π * x / L) * sin(2π * y / L) * (1 + z / H)

            for timestepper in (:RungeKutta3, :QuasiAdamsBashforth2)
                closure = ScalarDiffusivity(ν=1e-2, κ=1e-2)

                # ENDLESS on an L × L velocity domain replicated once in every direction ...
                small = RectilinearGrid(arch, size=(N, N, 8), extent=(L, L, H))
                endless = NonhydrostaticModel(small; timestepper, closure, advection=WENO(),
                                              extended_tracers=ExtendedTracers(:c, east=1, west=1, north=1, south=1,
                                                                               advection=WENO()))

                # ... against a reference tracer on the full 3L × 3L domain
                big = RectilinearGrid(arch, size=(3N, 3N, 8), extent=(3L, 3L, H))
                reference = NonhydrostaticModel(big; timestepper, closure, advection=WENO(), tracers=:c)

                set!(endless, u=uᵢ, v=vᵢ)
                set!(reference, u=uᵢ, v=vᵢ)

                # The extended domain spans x ∈ [-L, 2L] and the reference x ∈ [0, 3L]:
                # a shift of exactly one tile, so matching indices are physically equivalent.
                cᵢ = [exp(-(((i - 1.5N)^2 + (j - 1.5N)^2) * (L / N)^2) / 150^2)
                      for i in 1:3N, j in 1:3N, k in 1:8]
                set!(endless.extended_tracers.c, cᵢ)
                set!(reference.tracers.c, cᵢ)

                for n in 1:10
                    time_step!(endless, 5)
                    time_step!(reference, 5)
                end

                cₑ = Array(interior(endless.extended_tracers.c))
                cᵣ = Array(interior(reference.tracers.c))
                @test all(isfinite, cₑ)
                @test maximum(abs, cₑ .- cᵣ) < atol * maximum(abs, cᵣ)

                # Periodic outer edges conserve tracer
                @test sum(cₑ) ≈ sum(cᵢ) rtol=atol
            end
        end
    end

    @testset "Zero extension" begin
        for arch in archs
            FT = eltype(RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1)))
            N, L, H = 8, 1000, 200
            uᵢ(x, y, z) = 0.05 * sin(2π * x / L) * cos(2π * y / L) * (1 + z / H)
            cᵢ(x, y, z) = exp(-((x - L/2)^2 + (y - L/2)^2) / 150^2)

            grid = RectilinearGrid(arch, size=(N, N, 8), extent=(L, L, H))
            closure = ScalarDiffusivity(ν=1e-2, κ=1e-2)

            extended = NonhydrostaticModel(grid; closure, advection=WENO(),
                                           extended_tracers=ExtendedTracers(:c))
            plain = NonhydrostaticModel(grid; closure, advection=WENO(), tracers=:c)

            @test size(extended.extended_tracers.grid) == size(grid)

            for model in (extended, plain)
                set!(model, u=uᵢ)
            end
            set!(extended.extended_tracers.c, cᵢ)
            set!(plain.tracers.c, cᵢ)

            for n in 1:10
                time_step!(extended, 5)
                time_step!(plain, 5)
            end

            cₑ = Array(interior(extended.extended_tracers.c))
            cₚ = Array(interior(plain.tracers.c))
            @test maximum(abs, cₑ .- cₚ) < 100 * eps(FT) * maximum(abs, cₚ)
        end
    end

    @testset "Forcing and closure fields" begin
        for arch in archs
            N, L, H = 8, 1000, 200
            grid = RectilinearGrid(arch, size=(N, N, 8), extent=(L, L, H))
            uᵢ(x, y, z) = 0.05 * sin(2π * x / L) * cos(2π * y / L) * (1 + z / H)

            # A localized source, Qᵪ of Chen et al. (2016) Eq. (4)
            source(x, y, z, t) = ifelse(abs(x - L/2) < 100 && abs(y - L/2) < 100 && z > -50, 1e-3, 0)

            forced = NonhydrostaticModel(grid; advection=WENO(), closure=ScalarDiffusivity(ν=1e-2, κ=1e-2),
                                         extended_tracers=ExtendedTracers(:c, east=1, west=1, forcing=(; c=source)))
            set!(forced, u=uᵢ)
            for n in 1:5
                time_step!(forced, 5)
            end
            c = Array(interior(forced.extended_tracers.c))
            @test all(isfinite, c)
            @test sum(c) > 0

            # A closure with diffusivity fields: those fields are tiled onto the extended domain
            smagorinsky = NonhydrostaticModel(grid; advection=WENO(), closure=SmagorinskyLilly(),
                                              tracers=:b, buoyancy=BuoyancyTracer(),
                                              extended_tracers=ExtendedTracers(:c, east=1, west=1, north=1, south=1))
            set!(smagorinsky, u=uᵢ, b=(x, y, z) -> 1e-5 * z)
            set!(smagorinsky.extended_tracers.c, (x, y, z) -> exp(-((x - L/2)^2 + (y - L/2)^2) / 150^2))
            c₀ = sum(Array(interior(smagorinsky.extended_tracers.c)))
            for n in 1:5
                time_step!(smagorinsky, 2)
            end
            c = Array(interior(smagorinsky.extended_tracers.c))
            @test all(isfinite, c)
            @test sum(c) ≈ c₀
        end
    end
end
