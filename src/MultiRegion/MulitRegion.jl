module MultiRegion

using Printf
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.BoundaryConditions: getbc

struct EqualXParition end

struct XPartition
    sizes
end

#=
Simone says...

1.
grid = MultiRegionGrid(GPU(), RectilinearGrid; partition, devices, kw...)

2.
multi_region_arch = MultiRegion(...)
grid =
=#