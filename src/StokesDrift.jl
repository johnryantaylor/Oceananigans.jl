module StokesDrift

using Oceananigans.Fields: field
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

@inline ∂t_uˢ(i, j, k, grid, ::Nothing) = zero(eltype(grid))
@inline ∂t_vˢ(i, j, k, grid, ::Nothing) = zero(eltype(grid))
@inline ∂t_wˢ(i, j, k, grid, ::Nothing) = zero(eltype(grid))

@inline x_curl_Uˢ_cross_U(i, j, k, grid, ::Nothing, U) = zero(eltype(grid))
@inline y_curl_Uˢ_cross_U(i, j, k, grid, ::Nothing, U) = zero(eltype(grid))
@inline z_curl_Uˢ_cross_U(i, j, k, grid, ::Nothing, U) = zero(eltype(grid))

regularize_stokes_drift(::Nothing, grid, clock) = nothing

#####
##### Helper functions for Stokes drift that's uniform in x, y
#####

@inline x_curl_Uˢ_cross_U(i, j, k, grid, w, ∂z_uˢ) = ℑxzᶠᵃᶜ(i, j, k, grid, w) * ∂z_uˢ
@inline y_curl_Uˢ_cross_U(i, j, k, grid, w, ∂z_vˢ) = ℑyzᵃᶠᶜ(i, j, k, grid, w) * ∂z_vˢ
@inline z_curl_Uˢ_cross_U(i, j, k, grid, u, v, ∂z_uˢ, ∂z_vˢ) =
    - ℑxzᶜᵃᶠ(i, j, k, grid, u) * sw.∂z_uˢ - ℑyzᵃᶜᶠ(i, j, k, grid, v) * sw.∂z_vˢ

#####
##### Stokes drift uniform in x, y
#####

struct UniformStokesDrift{UZ, VZ, UT, VT} <: AbstractStokesDrift
    ∂z_uˢ :: UZ
    ∂z_vˢ :: VZ
    ∂t_uˢ :: UT
    ∂t_vˢ :: VT
end

const USD = UniformStokesDrift

@inline Stokes_shear_xᶠᶜᶜ(i, j, k, grid, sd::USD{<:Nothing}, time) = sd.∂z_uˢ[i, j, k]
@inline Stokes_shear_yᵃᵃᶜ(i, j, k, grid, sd::USD{<:Any, <:Nothing}, time) = sd.∂z_vˢ[i, j, k]

# Fallbacks use the Stokes drift
@inline Stokes_shear_xᶠᶜᶜ(i, j, k, grid, sd::USD, time) = ∂zᶠᶜᶜ(i, j, k, grid, sd.uˢ)
@inline Stokes_shear_yᶜᶠᶜ(i, j, k, grid, sd::USD, time) = ∂zᶜᶠᶜ(i, j, k, grid, sd.vˢ)

"""
    UniformStokesDrift(; uˢ=nothing, vˢ=nothing, ∂z_uˢ=nothing, ∂z_vˢ=nothing, ∂t_uˢ=nothing, ∂t_vˢ=nothing)

Construct a set of functions that describes the Stokes drift field beneath
a uniform surface gravity wave field.
"""
function UniformStokesDrift(∂z_uˢ = nothing,
                            ∂z_vˢ = nothing,
                            ∂t_uˢ = nothing,
                            ∂t_vˢ = nothing)

    return UniformStokesDrift(∂z_uˢ, ∂z_vˢ, ∂t_uˢ, ∂t_vˢ)
end

function regularize_stokes_drift(sd::USD, grid, clock)
    shear_location = (Nothing, Nothing, Face)
    drift_location = (Nothing, Nothing, Center)
    ∂z_uˢ = field(shear_location, sd.∂z_uˢ, grid; clock)
    ∂z_vˢ = field(shear_location, sd.∂z_vˢ, grid; clock)
    ∂t_uˢ = field(drift_location, sd.∂t_uˢ, grid; clock)
    ∂t_vˢ = field(drift_location, sd.∂t_vˢ, grid; clock)
    return UniformStokesDrift(∂z_uˢ, ∂z_vˢ, ∂t_uˢ, ∂t_vˢ)
end

@inline ∂t_wˢ(i, j, k, grid, sd::USD) = zero(eltype(grid))

@inline ∂t_uˢ(i, j, k, grid, sd::USD) = @inline sd.∂t_uˢ[i, j, k]
@inline ∂t_vˢ(i, j, k, grid, sd::USD) = @inline sd.∂t_vˢ[i, j, k]
@inline x_curl_Uˢ_cross_U(i, j, k, grid, sd::USD, U) = ℑxzᶠᵃᶜ(i, j, k, grid, U.w) * ℑzᵃᵃᶜ(i, j, k, grid, sd.∂z_uˢ)
@inline y_curl_Uˢ_cross_U(i, j, k, grid, sd::USD, U) = ℑyzᵃᶠᶜ(i, j, k, grid, U.w) * ℑzᵃᵃᶜ(i, j, k, grid, sd.∂z_vˢ)

@inline z_curl_Uˢ_cross_U(i, j, k, grid, sd::USD, U) = @inbounds begin
    - ℑxzᶜᵃᶠ(i, j, k, grid, U.u) * sd.∂z_uˢ[i, j, k] - ℑyzᵃᶜᶠ(i, j, k, grid, U.v) * sd.∂z_vˢ[i, j, k]
end

# Special cases
@inline ∂t_uˢ(i, j, k, grid, sd::USD{<:Any, <:Any, Nothing})        = zero(eltype(grid))
@inline ∂t_vˢ(i, j, k, grid, sd::USD{<:Any, <:Any, <:Any, Nothing}) = zero(eltype(grid))

@inline x_curl_Uˢ_cross_U(i, j, k, grid, sd::USD{Nothing}, U)        = zero(eltype(grid))
@inline y_curl_Uˢ_cross_U(i, j, k, grid, sd::USD{<:Any, Nothing}, U) = zero(eltype(grid))
@inline z_curl_Uˢ_cross_U(i, j, k, grid, sd::USD{Nothing}, U)        = @inbounds - ℑyzᵃᶜᶠ(i, j, k, grid, U.v) * sd.∂z_vˢ[i, j, k]
@inline z_curl_Uˢ_cross_U(i, j, k, grid, sd::USD{<:Any, Nothing}, U) = @inbounds - ℑxzᶜᵃᶠ(i, j, k, grid, U.u) * sd.∂z_uˢ[i, j, k]

#=
#####
##### Equilibrium Stokes drift utility
#####

struct EquilibriumStokesDrift{T}
    uˢ₀ :: T # Surface Stokes drift in x-direction
    vˢ₀ :: T # Surface Stokes drift in y-direction
    kᵖ :: T  # Peak wavenumber
end

const ESD = EquilibriumStokesDrift

@inline ∂t_uˢ(i, j, k, grid, ::ESD, time) = zero(eltype(grid))
@inline ∂t_vˢ(i, j, k, grid, ::ESD, time) = zero(eltype(grid))
@inline ∂t_wˢ(i, j, k, grid, ::ESD, time) = zero(eltype(grid))

function EquilibriumStokesDrift(; Ckᵖ= 0.167 # Peak wavenumber scaling parameter
                                  Cuˢ= 0.016 # Scaling between wind stress and surface stokes drift
                                  Aᵐʳ = 5.1e-4 # "inverse acceleration" parameter relating wind stress to Stokes transport
                                  ρʷ = 1035 # Water density for "toy" bulk formula
                                  ρᵃ = 1.225 # Air density for "toy" bulk formula
                                  Cᵈ = 1e-3 # Drag coefficient for toy bulk formula
                                  τˣ = 0.0 # Wind stress in x-direction
                                  τʸ = 0.0 # Wind stress in y-direction
    
    τ = sqrt(τˣ^2 + τʸ^2)
    U₁₀ = sqrt(τ * ρʷ / (Cᵈ * ρᵃ))
    u₁₀ = U₁₀ * τˣ / τ
    v₁₀ = U₁₀ * τʸ / τ

    uˢ₀ = * u₁₀
    vˢ₀ = Cuˢ * v₁₀
    Uˢ = Aᵐʳ * U₁₀^2 * u₁₀
    Vˢ = Aᵐʳ * U₁₀^2 * v₁₀

    # Peak wavenumber
    #
    # kᵖ = Ckᵖ * uˢ₀ / Uˢ
    #    = Ckᵖ * Cuˢ / (Aᵐʳ * U₁₀^2) 
    #
    # with uˢ₀ * Cuˢ * u₁₀ and Uˢ = Aᵐʳ * U₁₀^2 * u₁₀ ... ?
    kᵖ = Ckᵖ * Cuˢ / (Aᵐʳ * U₁₀^2)
    uˢ₀ = Cuˢ * U₁₀

    return EquilibriumStokesDrift(Ckᵖ, Cuˢ, AⱽEqilibriu
end

# TODO: add a reference
@inline T₁(k, z) = exp(2k * z)
@inline T₂(k, z) = sqrt(2π * k * abs(z)) * erfc(sqrt(2k * abs(z)))

@inline (eq::EquilibriumStokesDrift)(z, t) = eq.uˢ₀ * (T₁(eq.kᵖ, z)

const ESD = EquilibriumStokesDrift

@inline U₁₀(e::ESD) = sqrt(sqrt(e.τˣ^2 + e.τʸ^2) * e.ρʷ / (e.Cᵈ * e.ρᵃ))
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
=#

end # module
