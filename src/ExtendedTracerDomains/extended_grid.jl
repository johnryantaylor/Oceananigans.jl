using Oceananigans.Architectures: architecture
using Oceananigans.Grids: RectilinearGrid, Flat, Periodic, topology, halo_size, pop_flat_elements,
                          cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z

"""
    tiled_faces(faces, L, R, first_tile)

Replicate the coordinate `faces` of a periodic direction of extent `L` a total of `R` times,
placing the original domain in tile `first_tile` (zero-based). Handles both the `(x₁, x₂)` tuple
returned by `cpu_face_constructor_x` for regularly-spaced grids and the vector of `N + 1`
face positions returned for stretched grids.
"""
tiled_faces(::Nothing, L, R, first_tile) = nothing

function tiled_faces(faces::Tuple, L, R, first_tile)
    x₁, x₂ = faces
    return (x₁ - first_tile * L, x₂ + (R - 1 - first_tile) * L)
end

function tiled_faces(faces::AbstractVector, L, R, first_tile)
    N = length(faces) - 1
    tiled = similar(faces, R * N + 1)

    for m in 0:R-1
        shift = (m - first_tile) * L
        for i in 1:N
            tiled[m * N + i] = faces[i] + shift
        end
    end

    tiled[end] = faces[end] + (R - 1 - first_tile) * L

    return tiled
end

"""
$(TYPEDSIGNATURES)

Return a `RectilinearGrid` built by replicating `grid` `west` times to the west, `east`
times to the east, `south` times to the south and `north` times to the north, so that the
returned grid has `(west + 1 + east) * Nx` points in `x` and `(south + 1 + north) * Ny`
points in `y`. Halos, vertical coordinate, topology, architecture and element type are
inherited from `grid`, and the original domain occupies the tile it started in — so grid
spacings are reproduced exactly, including for stretched horizontal coordinates.

Replication is only meaningful in `Periodic` directions; see [`ExtendedTracers`](@ref).

```jldoctest
julia> using Oceananigans

julia> using Oceananigans.ExtendedTracerDomains: extended_grid

julia> grid = RectilinearGrid(size=(8, 8, 4), extent=(1, 1, 1));

julia> extended_grid(grid, east=1, west=1, north=1, south=1)
24×24×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── Periodic x ∈ [-1.0, 2.0) regularly spaced with Δx=0.125
├── Periodic y ∈ [-1.0, 2.0) regularly spaced with Δy=0.125
└── Bounded  z ∈ [-1.0, 0.0] regularly spaced with Δz=0.25
```
"""
function extended_grid(grid::RectilinearGrid; east=0, west=0, north=0, south=0)
    topo = topology(grid)
    Hx, Hy, Hz = halo_size(grid)

    Rx = west + 1 + east
    Ry = south + 1 + north

    sz   = pop_flat_elements((Rx * grid.Nx, Ry * grid.Ny, grid.Nz), topo)
    halo = pop_flat_elements((Hx, Hy, Hz), topo)

    x = tiled_faces(cpu_face_constructor_x(grid), grid.Lx, Rx, west)
    y = tiled_faces(cpu_face_constructor_y(grid), grid.Ly, Ry, south)
    z = cpu_face_constructor_z(grid)

    return RectilinearGrid(architecture(grid), eltype(grid);
                           size = sz, halo, x, y, z, topology = topo)
end
