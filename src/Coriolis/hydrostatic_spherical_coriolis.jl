using Oceananigans.Grids: LatitudeLongitudeGrid, OrthogonalSphericalShellGrid, peripheral_node, φnode
using Oceananigans.Operators: Δx_qᶜᶠᶜ, Δy_qᶠᶜᶜ, Δxᶠᶜᶜ, Δyᶜᶠᶜ, hack_sind
using Oceananigans.Advection: EnergyConserving, EnstrophyConserving
using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.ImmersedBoundaries

"""
    struct ActiveCellEnstrophyConserving

A parameter object for an enstrophy-conserving Coriolis scheme that excludes inactive (dry/land) edges
(indices for which `peripheral_node == true`) from the velocity interpolation.
"""
struct ActiveCellEnstrophyConserving end

"""
    struct HydrostaticSphericalCoriolis{S, FT} <: AbstractRotation

A parameter object for constant rotation around a vertical axis on the sphere.
"""
struct HydrostaticSphericalCoriolis{S, FT} <: AbstractRotation
    rotation_rate :: FT
    scheme :: S
end

"""
    HydrostaticSphericalCoriolis([FT=Float64;]
                                 rotation_rate = Ω_Earth,
                                 scheme = ActiveCellEnstrophyConserving())

Return a parameter object for Coriolis forces on a sphere rotating at `rotation_rate`.

Keyword arguments
=================

- `rotation_rate`: Sphere's rotation rate; default: [`Ω_Earth`](@ref).
- `scheme`: Either `EnergyConserving()`, `EnstrophyConserving()`, or `ActiveCellEnstrophyConserving()` (default).
"""
function HydrostaticSphericalCoriolis(FT::DataType=Oceananigans.defaults.FloatType;
                                      rotation_rate = Ω_Earth,
                                      scheme :: S = ActiveCellEnstrophyConserving()) where S

    return HydrostaticSphericalCoriolis{S, FT}(rotation_rate, scheme)
end

Adapt.adapt_structure(to, coriolis::HydrostaticSphericalCoriolis) =
    HydrostaticSphericalCoriolis(Adapt.adapt(to, coriolis.rotation_rate), 
                                 Adapt.adapt(to, coriolis.scheme))

@inline φᶠᶠᵃ(i, j, k, grid::LatitudeLongitudeGrid)        = φnode(j, grid, Face())
@inline φᶠᶠᵃ(i, j, k, grid::OrthogonalSphericalShellGrid) = φnode(i, j, grid, Face(), Face())
@inline φᶠᶠᵃ(i, j, k, grid::ImmersedBoundaryGrid)         = φᶠᶠᵃ(i, j, k, grid.underlying_grid)

@inline fᶠᶠᵃ(i, j, k, grid, coriolis::HydrostaticSphericalCoriolis) =
    2 * coriolis.rotation_rate * hack_sind(φᶠᶠᵃ(i, j, k, grid))

@inline z_f_cross_U(i, j, k, grid, coriolis::HydrostaticSphericalCoriolis, U) = zero(grid)

#####
##### Active Point Enstrophy-conserving scheme
#####

# It might happen that a cell is active but all the neighbouring staggered nodes are inactive,
# (an example is a 1-cell large channel)
# In that case the Coriolis force is equal to zero

const CoriolisActiveCellEnstrophyConserving = HydrostaticSphericalCoriolis{<:ActiveCellEnstrophyConserving}

@inline not_peripheral_node(args...) = !peripheral_node(args...)

@inline function mask_inactive_points_ℑxyᶠᶜᵃ(i, j, k, grid, f::Function, args...) 
    neighboring_active_nodes = ℑxyᶠᶜᵃ(i, j, k, grid, not_peripheral_node, Center(), Face(), Center())
    return ifelse(neighboring_active_nodes == 0, zero(grid),
                  ℑxyᶠᶜᵃ(i, j, k, grid, f, args...) / neighboring_active_nodes)
end

@inline function mask_inactive_points_ℑxyᶜᶠᵃ(i, j, k, grid, f::Function, args...) 
    neighboring_active_nodes = @inbounds ℑxyᶜᶠᵃ(i, j, k, grid, not_peripheral_node, Face(), Center(), Center())
    return ifelse(neighboring_active_nodes == 0, zero(grid),
                  ℑxyᶜᶠᵃ(i, j, k, grid, f, args...) / neighboring_active_nodes)
end

@inline x_f_cross_U(i, j, k, grid, coriolis::CoriolisActiveCellEnstrophyConserving, U) =
    @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) *
                mask_inactive_points_ℑxyᶠᶜᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, U[2]) / Δxᶠᶜᶜ(i, j, k, grid)

@inline y_f_cross_U(i, j, k, grid, coriolis::CoriolisActiveCellEnstrophyConserving, U) =
    @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) *
                mask_inactive_points_ℑxyᶜᶠᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, U[1]) / Δyᶜᶠᶜ(i, j, k, grid)

#####
##### Enstrophy-conserving scheme
#####

const CoriolisEnstrophyConserving = HydrostaticSphericalCoriolis{<:EnstrophyConserving}

@inline x_f_cross_U(i, j, k, grid, coriolis::CoriolisEnstrophyConserving, U) =
    @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) *
                ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, U[2]) / Δxᶠᶜᶜ(i, j, k, grid)

@inline y_f_cross_U(i, j, k, grid, coriolis::CoriolisEnstrophyConserving, U) =
    @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) *
                ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, U[1]) / Δyᶜᶠᶜ(i, j, k, grid)

#####
##### Energy-conserving scheme
#####

const CoriolisEnergyConserving = HydrostaticSphericalCoriolis{<:EnergyConserving}

@inline f_ℑx_vᶠᶠᵃ(i, j, k, grid, coriolis, v) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, v)
@inline f_ℑy_uᶠᶠᵃ(i, j, k, grid, coriolis, u) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑyᵃᶠᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, u)

@inline x_f_cross_U(i, j, k, grid, coriolis::CoriolisEnergyConserving, U) =
    @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, f_ℑx_vᶠᶠᵃ, coriolis, U[2]) / Δxᶠᶜᶜ(i, j, k, grid)

@inline y_f_cross_U(i, j, k, grid, coriolis::CoriolisEnergyConserving, U) =
    @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, f_ℑy_uᶠᶠᵃ, coriolis, U[1]) / Δyᶜᶠᶜ(i, j, k, grid)

#####
##### Show
#####

function Base.show(io::IO, hydrostatic_spherical_coriolis::HydrostaticSphericalCoriolis) 
    coriolis_scheme = hydrostatic_spherical_coriolis.scheme
    rotation_rate   = hydrostatic_spherical_coriolis.rotation_rate
    rotation_rate_Earth = rotation_rate / Ω_Earth
    rotation_rate_str = @sprintf("%.2e s⁻¹ = %.2e Ω_Earth", rotation_rate, rotation_rate_Earth)

    return print(io, "HydrostaticSphericalCoriolis", '\n',
                 "├─ rotation rate: ", rotation_rate_str, '\n',
                 "└─ scheme: ", summary(coriolis_scheme))
end
