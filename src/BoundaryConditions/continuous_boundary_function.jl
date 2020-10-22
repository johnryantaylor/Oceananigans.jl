using Oceananigans.Operators: index_and_interp_dependencies
using Oceananigans.Utils: tupleit, user_function_arguments

import Oceananigans: location

"""
    ContinuousBoundaryFunction{X, Y, Z, I, F, P, D, N, ℑ} <: Function

A wrapper for the user-defined boundary condition function `func` at location
`X, Y, Z`. `I` denotes the boundary-normal index (`I=1` at western boundaries,
`I=grid.Nx` at eastern boundaries, etc). `F, P, D, N, ℑ` are, respectively, the 
user-defined function, parameters, field dependencies, indices of the field dependencies
in `model_fields`, and interpolation operators for interpolating `model_fields` to the
location at which the boundary condition is applied.
"""
struct ContinuousBoundaryFunction{X, Y, Z, I, F, P, D, N, ℑ} <: Function
                          func :: F
                    parameters :: P
            field_dependencies :: D
    field_dependencies_indices :: N
     field_dependencies_interp :: ℑ

    """ Returns a location-less wrapper for `func`, `parameters`, and `field_dependencies`."""
    function ContinuousBoundaryFunction(func, parameters, field_dependencies)

        field_dependencies = tupleit(field_dependencies)

        return new{Nothing, Nothing, Nothing, Nothing,
                   typeof(func), typeof(parameters),
                   typeof(field_dependencies), Nothing, Nothing}(func, parameters, field_dependencies, nothing, nothing)
    end

    function ContinuousBoundaryFunction{X, Y, Z, I}(func, parameters, field_dependencies,
                                                    field_dependencies_indices, field_dependencies_interp) where {X, Y, Z, I}
        return new{X, Y, Z, I,
                   typeof(func),
                   typeof(parameters),
                   typeof(field_dependencies),
                   typeof(field_dependencies_indices),
                   typeof(field_dependencies_interp)}(func,
                                                      parameters,
                                                      field_dependencies,
                                                      field_dependencies_indices,
                                                      field_dependencies_interp)
    end
end

location(bc::ContinuousBoundaryFunction{X, Y, Z}) where {X, Y, Z} = X, Y, Z

#####
##### "Regularization" for IncompressibleModel setup
#####

regularize_boundary_condition(bc, X, Y, Z, I, model_field_names) = bc # fallback

"""
    regularize_boundary_condition(bc::BoundaryCondition{C, <:ContinuousBoundaryFunction},
                                  X, Y, Z, I, model_field_names) where C

Regularizes `bc.condition` for location `X, Y, Z`, boundary index `I`, and `model_field_names`,
returning `BoundaryCondition(C, regularized_condition)`.

The regularization of `bc.condition::ContinuousBoundaryFunction` requries

1. Setting the boundary location to `X, Y, Z`.
   The boundary-normal direction is tagged with `Nothing` location.

2. Setting the boundary-normal index `I` for indexing into `field_dependencies`.
   `I` is either `1` (for left boundaries) or
   `size(grid, n)` for a boundary in the `n`th direction where `n ∈ (1, 2, 3)` corresponds
   to `x, y, z`.

3. Determining the `indices` that map `model_fields` to `field_dependencies`.

4. Determining the `interps` functions that interpolate field_dependencies to the location
   of the boundary.
"""
function regularize_boundary_condition(bc::BoundaryCondition{C, <:ContinuousBoundaryFunction},
                                       X, Y, Z, I, model_field_names) where C
    boundary_func = bc.condition

    indices, interps = index_and_interp_dependencies(X, Y, Z,
                                                     boundary_func.field_dependencies,
                                                     model_field_names)

    regularized_boundary_func = ContinuousBoundaryFunction{X, Y, Z, I}(boundary_func.func,
                                                                       boundary_func.parameters,
                                                                       boundary_func.field_dependencies,
                                                                       indices, interps)

    return BoundaryCondition(C, regularized_boundary_func)
end

#####
##### Kernel functions
#####

@inline function (bc::ContinuousBoundaryFunction{Nothing, Y, Z, i})(j, k, grid, clock, model_fields) where {Y, Z, i}
    args = user_function_arguments(i, j, k, grid, model_fields, bc.parameters, bc)
    return bc.func(ynode(Y, j, grid), znode(Z, k, grid), clock.time, args...)
end

@inline function (bc::ContinuousBoundaryFunction{X, Nothing, Z, j})(i, k, grid, clock, model_fields) where {X, Z, j}
    args = user_function_arguments(i, j, k, grid, model_fields, bc.parameters, bc)
    return bc.func(xnode(X, i, grid), znode(Z, k, grid), clock.time, args...)
end

@inline function (bc::ContinuousBoundaryFunction{X, Y, Nothing, k})(i, j, grid, clock, model_fields) where {X, Y, k}
    args = user_function_arguments(i, j, k, grid, model_fields, bc.parameters, bc)
    return bc.func(xnode(X, i, grid), ynode(Y, j, grid), clock.time, args...)
end

# Don't re-convert ContinuousBoundaryFunctions passed to BoundaryCondition constructor
BoundaryCondition(TBC, condition::ContinuousBoundaryFunction) =
    BoundaryCondition{TBC, typeof(condition)}(condition)
