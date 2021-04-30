function diagnose_velocities_from_streamfunction(ψ, grid)
    ψᶠᶠᶜ = Field(Face, Face,   Center, CPU(), grid)
    uᶠᶜᶜ = Field(Face, Center, Center, CPU(), grid)
    vᶜᶠᶜ = Field(Center, Face, Center, CPU(), grid)

    for (f, grid_face) in enumerate(grid.faces)
        Nx, Ny, Nz = size(grid_face)

        ψᶠᶠᶜ_face = ψᶠᶠᶜ.data.faces[f]
        uᶠᶜᶜ_face = uᶠᶜᶜ.data.faces[f]
        vᶜᶠᶜ_face = vᶜᶠᶜ.data.faces[f]

        for i in 1:Nx+1, j in 1:Ny+1
            ψᶠᶠᶜ_face[i, j, 1] = ψ(grid_face.λᶠᶠᵃ[i, j], grid_face.φᶠᶠᵃ[i, j])
        end

        for i in 1:Nx+1, j in 1:Ny
            uᶠᶜᶜ_face[i, j, 1] = (ψᶠᶠᶜ_face[i, j, 1] - ψᶠᶠᶜ_face[i, j+1, 1]) / grid.faces[f].Δyᶠᶜᵃ[i, j]
        end

        for i in 1:Nx, j in 1:Ny+1
            vᶜᶠᶜ_face[i, j, 1] = (ψᶠᶠᶜ_face[i+1, j, 1] - ψᶠᶠᶜ_face[i, j, 1]) / grid.faces[f].Δxᶜᶠᵃ[i, j]
        end
    end

    return uᶠᶜᶜ, vᶜᶠᶜ, ψᶠᶠᶜ
end

function set_velocities_from_streamfunction!(u, v, ψ, grid)
    uψ, vψ, ψ₀ = diagnose_velocities_from_streamfunction(ψ, grid)
    set!(u, uψ)
    set!(v, vψ)
    return nothing
end

@testset "Cubed sphere circulation" begin

    arch = CPU()
    grid = ConformalCubedSphereGrid(cs32_filepath, Nz=1, z=(-1, 0))

    FT = eltype(grid)
    Nx, Ny, Nz, Nf = size(grid)
    R = grid.faces[1].radius

    u_field = XFaceField(arch, grid)
    v_field = YFaceField(arch, grid)

    ψ(λ, φ) = R * sind(φ)
    set_velocities_from_streamfunction!(u_field, v_field, ψ, grid)

    fill_horizontal_velocity_halos!(u_field, v_field, arch)
    # fill_horizontal_velocity_halos!(u_field, v_field, arch)

    for grid_face in grid.faces
        grid_face.Δxᶠᶜᵃ .= 1
        grid_face.Δyᶜᶠᵃ .= 1
    end

    grid_faces = [grid.faces[f] for f in 1:Nf]
    u_faces = [get_face(u_field, f) for f in 1:Nf]
    v_faces = [get_face(v_field, f) for f in 1:Nf]

    circulation(i, j, f) = Γᶠᶠᵃ(i, j, 1, grid_faces[f], u_faces[f], v_faces[f])

    @testset "Face 1-2 boundary" begin
        # Quick test at a single point.
        Γ1 = circulation(33, 16, 1)
        Γ2 = circulation(1,  16, 2)
        @test Γ1 == Γ2

        # Check the entire boundary.
        Γ1 = [circulation(33, j, 1) for j in 1:Ny+1]
        Γ2 = [circulation(1,  j, 2) for j in 1:Ny+1]

        # There should be no truncation errors here.
        @test Γ1 == Γ2
    end

    @testset "Face 1-3 boundary" begin
        Γ1 = circulation(16, 33, 1)
        Γ3 = circulation(1,  16, 3)
        @test Γ1 == Γ3

        # ffc locations are shifted by one along this boundary (and go in reverse).
        Γ1 = [circulation(i, 33, 1) for i in 1:Nx]
        Γ3 = [circulation(1,  j, 3) for j in Ny+1:-1:2]

        # Possible truncation errors at this boundary.
        ϵ = eps(maximum(abs, [Γ1; Γ3]))
        @test all(isapprox(Γ1, Γ3, atol=10ϵ))
    end

    @testset "Face 1-5 boundary" begin
        Γ1 = [circulation(1,  j, 1) for j in 1:Ny]
        Γ5 = [circulation(i, 33, 5) for i in Nx+1:-1:2]

        ϵ = eps(maximum(abs, [Γ1; Γ5]))
        @test all(isapprox(Γ1, Γ5, atol=10ϵ))
    end

    @testset "Face 1-6 boundary" begin
        Γ1 = [circulation(i,  1, 1) for i in 1:Nx+1]
        Γ6 = [circulation(i, 33, 6) for i in 1:Nx+1]
        @test Γ1 == Γ6
    end

    @testset "Face 1-2-3 corner" begin
        Γ1 = circulation(33, 33, 1)
        Γ2 = circulation(1,  33, 2)
        Γ3 = circulation(1,  1,  3)
        @test Γ1 == Γ2 == Γ3
    end

    @testset "Face 1-3-5 corner" begin
        Γ1, Γ3, Γ5 = [circulation(1, 33, f) for f in (1, 3, 5)]
        @test Γ1 == Γ3 == Γ5
    end

    @testset "Face 1-5-6 corner" begin
        Γ1 = circulation(1,   1, 1)
        Γ5 = circulation(33, 33, 5)
        Γ6 = circulation(1,  33, 6)
        @test Γ1 == Γ5 == Γ6
    end

    @testset "Face 1-2-6 corner" begin
        Γ1 = circulation(33,  1, 1)
        Γ2 = circulation(1,   1, 2)
        Γ6 = circulation(33, 33, 6)
        @test Γ1 == Γ2 == Γ6
    end

    @testset "Face 2-3-4 corner" begin
        Γ2 = circulation(33, 33, 2)
        Γ3 = circulation(33, 1,  3)
        Γ4 = circulation(1,  1,  4)
        @test Γ2 == Γ3 == Γ4
    end

    @testset "Face 2-4-6 corner" begin
        Γ2, Γ4, Γ6 = [circulation(33, 1, f) for f in (2, 4, 6)]
        @test Γ2 == Γ4 == Γ6
    end

end
