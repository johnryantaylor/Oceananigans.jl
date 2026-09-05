using Adapt
using Base: @propagate_inbounds

using Oceananigans.Grids: Periodic, topology
using Oceananigans.Fields: Field

"""
    TiledArray{Nx, Ny, D}

A zero-memory view of `data`, which is defined on a horizontally-periodic "base" grid with
`Nx, Ny` interior points, that may be indexed with the indices of an extended grid built by
replicating the base grid in `x` and `y` (see [`extended_grid`](@ref)).

Because `mod1(i, Nx) ∈ 1:Nx` always lands in the *interior* of the base field, the base
field's `x, y` halos are never read: the periodic wrap is exact and requires no additional
halo filling. The third index is passed through untouched, so vertical halos are used as-is.

`Nx` and `Ny` are type parameters so that `mod1` compiles to a multiply-and-shift rather
than an integer division, which matters on GPUs. A type parameter of `nothing` means "do
not wrap in this direction", which is used for directions that are not `Periodic` and which
therefore cannot be replicated.
"""
struct TiledArray{Nx, Ny, D}
    data :: D
end

TiledArray{Nx, Ny}(data::D) where {Nx, Ny, D} = TiledArray{Nx, Ny, D}(data)

@inline tiled_index(i, ::Nothing) = i
@inline tiled_index(i, N) = mod1(i, N)

@inline @propagate_inbounds Base.getindex(t::TiledArray{Nx, Ny}, i, j, k) where {Nx, Ny} =
    getindex(t.data, tiled_index(i, Nx), tiled_index(j, Ny), k)

Adapt.adapt_structure(to, t::TiledArray{Nx, Ny}) where {Nx, Ny} =
    TiledArray{Nx, Ny}(Adapt.adapt(to, t.data))

Base.summary(t::TiledArray{Nx, Ny}) where {Nx, Ny} =
    string("TiledArray tiling ", summary(t.data), " with (Nx, Ny) = (", Nx, ", ", Ny, ")")

Base.show(io::IO, t::TiledArray) = print(io, summary(t))

"""
    tiled_sizes(grid)

Return the `(Nx, Ny)` wrap sizes for `grid`: the number of interior points in directions
that are `Periodic`, and `nothing` in directions that are not (and which therefore cannot
be replicated).
"""
@inline function tiled_sizes(grid)
    TX, TY, _ = topology(grid)
    Nx = TX === Periodic ? size(grid, 1) : nothing
    Ny = TY === Periodic ? size(grid, 2) : nothing
    return Nx, Ny
end

"""
    tile(field, grid)

Wrap `field`, defined on the base `grid`, in a [`TiledArray`](@ref) so that it may be
indexed with extended-grid indices. `NamedTuple`s and `Tuple`s are mapped over; anything
that is not a `Field` (numbers, `nothing`, `ZeroField`, ...) is returned unchanged, since
such objects are already independent of horizontal position.
"""
function tile(field::Field, grid)
    Nx, Ny = tiled_sizes(grid)
    return TiledArray{Nx, Ny}(field.data)
end

tile(nt::NamedTuple, grid) = map(f -> tile(f, grid), nt)
tile(t::Tuple, grid) = map(f -> tile(f, grid), t)
tile(x, grid) = x
