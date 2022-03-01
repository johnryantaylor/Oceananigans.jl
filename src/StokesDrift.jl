module StokesDrift

using SpecialFunctions

export
    UniformStokesDrift,
    ∂t_uˢ,
    ∂t_vˢ,
    ∂t_wˢ,
    x_curl_Uˢ_cross_U,
    y_curl_Uˢ_cross_U,
    z_curl_Uˢ_cross_U

using Oceananigans.Grids: AbstractGrid

using Oceananigans.Fields
using Oceananigans.Operators

"""
    abstract type AbstractStokesDrift end

Parent type for parameter structs for Stokes drift fields
associated with surface waves.
"""
abstract type AbstractStokesDrift end

#####
##### Functions for "no surface waves"
#####

@inline ∂t_uˢ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, time) where FT = zero(FT)
@inline ∂t_vˢ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, time) where FT = zero(FT)
@inline ∂t_wˢ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, time) where FT = zero(FT)

@inline x_curl_Uˢ_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, time) where FT = zero(FT)
@inline y_curl_Uˢ_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, time) where FT = zero(FT)
@inline z_curl_Uˢ_cross_U(i, j, k, grid::AbstractGrid{FT}, ::Nothing, U, time) where FT = zero(FT)

#####
##### Uniform surface waves
#####

"""
    UniformStokesDrift{UZ, VZ, UT, VT} <: AbstractStokesDrift

Parameter struct for Stokes drift fields associated with surface waves.
"""
struct UniformStokesDrift{UZ, VZ, UT, VT} <: AbstractStokesDrift
    ∂z_uˢ :: UZ
    ∂z_vˢ :: VZ
    ∂t_uˢ :: UT
    ∂t_vˢ :: VT
end

addzero(args...) = 0

"""
    UniformStokesDrift(; ∂z_uˢ=addzero, ∂z_vˢ=addzero, ∂t_uˢ=addzero, ∂t_vˢ=addzero)

Construct a set of functions that describes the Stokes drift field beneath
a uniform surface gravity wave field.
"""
UniformStokesDrift(; ∂z_uˢ=addzero, ∂z_vˢ=addzero, ∂t_uˢ=addzero, ∂t_vˢ=addzero) =
    UniformStokesDrift(∂z_uˢ, ∂z_vˢ, ∂t_uˢ, ∂t_vˢ)

const USD = UniformStokesDrift

@inline ∂t_uˢ(i, j, k, grid, sw::USD, time) = sw.∂t_uˢ(znode(Center(), k, grid), time)
@inline ∂t_vˢ(i, j, k, grid, sw::USD, time) = sw.∂t_vˢ(znode(Center(), k, grid), time)

@inline ∂t_wˢ(i, j, k, grid::AbstractGrid{FT}, sw::USD, time) where FT = zero(FT)

@inline x_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, time) =
    @inbounds ℑxzᶠᵃᶜ(i, j, k, grid, U.w) * sw.∂z_uˢ(znode(Center(), k, grid), time)

@inline y_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, time) =
    @inbounds ℑyzᵃᶠᶜ(i, j, k, grid, U.w) * sw.∂z_vˢ(znode(Center(), k, grid), time)

@inline z_curl_Uˢ_cross_U(i, j, k, grid, sw::USD, U, time) = @inbounds begin (
    - ℑxzᶜᵃᶠ(i, j, k, grid, U.u) * sw.∂z_uˢ(znode(Face(), k, grid), time)
    - ℑyzᵃᶜᶠ(i, j, k, grid, U.v) * sw.∂z_vˢ(znode(Face(), k, grid), time) )
end

Base.@kwdef struct EquilibriumStokesDrift{T}
    Ckᵖ :: T = 0.167 # Peak wavenumber scaling parameter
    Cuˢ :: T = 0.016 # Scaling between wind stress and surface stokes drift
    Aⱽ :: T = 1960.7 # "acceleration" parameter relating wind stress to Stokes transport
    τˣ :: T = 0.0 # Wind stress in x-direction
    τʸ :: T = 0.0 # Wind stress in y-direction
end

@inline Stokes_transport(eq::EquilibriumStokesDrift) = eq.Aⱽ * (eq.τˣ^2 + eq.τʸ^2)

# TODO: add a reference
@inline T₁(k, z) = exp(2k * z)
@inline T₂(k, z) = sqrt(2π*k*abs(z)) * erfc(sqrt(2k*abs(z)))

# @inline surface_stokes_drift(eq::EquilibriumWindWaveStokesDrift)

@inline function peak_wavenumber(eq::) =  = 
    uˢ₀ = surface_Stokes_drift(eq)
    Vˢ = Stokes_transport(eq)
    return eq.Ckᵖ * uˢ₀ / Vˢ
end

end # module
