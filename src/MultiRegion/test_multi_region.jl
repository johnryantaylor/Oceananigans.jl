using Oceananigans
using Oceananigans.MultiRegion
using Oceananigans.MultiRegion: XPartition, assoc_device, assoc_field
using Oceananigans.Fields: set!

grid = RectilinearGrid(CPU(), size = (12, 1), topology = (Periodic, Periodic, Flat), x = collect(0:12), y = (0, 1))

mrg = MultiRegionGrid(grid, partition = XPartition(4), devices = (0, 1))

field = MultiRegionField((Face, Face, Center), mrg)
set!(field, (x, y ,z) -> x)