"""
`GroupedTransform`

The sturct `GroupedTransform` represents a grouped transform

# properties
 * `setting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int}, Module, Vector{Int}}}}` ... vector of the dimensions, mode, and bandwidths for each term/group
 * `X::Array{Float64}` ... nodes of the transform in d x M format
 * `transforms::Vector{LinearMap{ComplexF64}}` ... vector of the transformations for each term

# constructor
`GroupedTransform(setting, X)` constructs coefficients from `data`

## Input
 * `setting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int}, Module, Vector{Int}}}}` ... vector of the dimensions, mode, and bandwidths for each term/group
 * `X::Array{Float64}` ... nodes of the transform in d x M format
"""
struct GroupedTransform
    system::String
    setting::Vector{
        NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}},
    }
    X::Array{Float64}
    transforms::Vector{Tuple{Int64,Int64}}

    function GroupedTransform(
        system::String,
        setting::Vector{
            NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}},
        },
        X::Array{Float64},
    )
        if !haskey(systems, system)
            error("System not found.")
        end

        if system == "exp"
            if (minimum(X) < -0.5) || (maximum(X) >= 0.5)
                error("Nodes must be between -0.5 and 0.5.")
            end
        elseif system == "cos"
            if (minimum(X) < 0) || (maximum(X) > 0.5)
                error("Nodes must be between 0 and 0.5.")
            end
        end

        transforms = Vector{Tuple{Int64,Int64}}(undef, length(setting))
        f = Vector{Tuple{Int64,Future}}(undef, length(setting))
        w = (nworkers() == 1) ? 1 : 2

        for (idx, s) in enumerate(setting)
            try
                f[idx] =
                    (w, remotecall(s[:mode].get_transform, w, s[:bandwidths], X[s[:u], :]))
                if nworkers() != 1
                    w = (w == nworkers()) ? 2 : (w + 1)
                end
            catch
                error(
                    "The mode is not supported yet or does not have the function get_transform.",
                )
            end
        end

        for (idx, s) in enumerate(setting)
            try
                transforms[idx] = (f[idx][1], fetch(f[idx][2]))
            catch
                error(
                    "The mode is not supported yet or does not have the function get_transform.",
                )
            end
        end
        new(system, setting, X, transforms)
    end
end

function GroupedTransform(
    system::String,
    d::Int,
    ds::Int,
    N::Vector{Int},
    X::Array{Float64},
)
    s = get_setting(system, d, ds, N)
    return GroupedTransform(system, s, X)
end

function GroupedTransform(
    system::String,
    U::Vector{Vector{Int}},
    N::Vector{Int},
    X::Array{Float64},
)
    s = get_setting(system, U, N)
    return GroupedTransform(system, s, X)
end

"""
`f = F*fhat`

# Input
 * `F::GroupedTransform`
 * `fhat::GroupedCoeff`

# Output
 * `f::Vector{ComplexF64}`
"""
function Base.:*(F::GroupedTransform, fhat::GroupedCoefficients)::Vector{<:Number}
    if F.setting != fhat.setting
        error("The GroupedTransform and the GroupedCoefficients have different settings")
    end
    f = Vector{Future}(undef, length(F.transforms))
    for i = 1:length(F.transforms)
        f[i] =
            @spawnat F.transforms[i][1] (F.setting[i][:mode].trafos[F.transforms[i][2]]) *
                                        (fhat[F.setting[i][:u]])
    end

    return sum(i -> fetch(f[i]), 1:length(F.transforms))
end


"""
`fhat = F*f`

# Input
 * `F::GroupedTransform`
 * `f::Vector{ComplexF64}`

# Output
 * `fhat::GroupedCoefficients`
"""
function Base.:*(F::GroupedTransform, f::Vector{<:Number})::GroupedCoefficients
    fh = Vector{Future}(undef, length(F.transforms))
    for i = 1:length(F.transforms)
        fh[i] =
            @spawnat F.transforms[i][1] (F.setting[i][:mode].trafos[F.transforms[i][2]])' *
                                        f
    end
    fhat = GroupedCoefficients(F.setting)
    for i = 1:length(F.transforms)
        fhat[F.setting[i][:u]] = fetch(fh[i])
    end
    return fhat
end


"""
`adjoint(F)`

overloads the F' notation and gives back the same GroupdTransform. GroupedTransform decides by the input if it is the normal trafo or the adjoint so this is only for convinience.

# Input
 * `F::GroupedTransform`

# Output
 * `F::GroupedTransform`
"""
function Base.:adjoint(F::GroupedTransform)::GroupedTransform
    return F
end


"""
This function overloads getindex of GroupedTransform such that you can do
  F[[1,3]]
to obtain the transform of the corresponding ANOVA term
"""

function Base.:getindex(F::GroupedTransform, u::Vector)
    idx = findfirst(s -> s[:u] == u, F.setting)
    if isnothing(idx)
        error("This term is not contained")
    else
        if F.system == "cos"
            function trafo_cos(fhat::Vector{Float64})::Vector{Float64}
                return remotecall_fetch(
                    F.setting[idx][:mode].trafo,
                    F.transforms[idx][1],
                    F.transforms[idx][2],
                    fhat,
                )
            end

            function adjoint_cos(f::Vector{Float64})::Vector{Float64}
                return remotecall_fetch(
                    F.setting[idx][:mode].adjoint,
                    F.transforms[idx][1],
                    F.transforms[idx][2],
                    f,
                )
            end

            N = prod(F.setting[idx][:bandwidths] .- 1)
            M = size(F.X, 2)
            return LinearMap{Float64}(trafo_cos, adjoint_cos, M, N)
        elseif F.system == "exp"
            function trafo_exp(fhat::Vector{ComplexF64})::Vector{ComplexF64}
                return remotecall_fetch(
                    F.setting[idx][:mode].trafo,
                    F.transforms[idx][1],
                    F.transforms[idx][2],
                    fhat,
                )
            end

            function adjoint_exp(f::Vector{ComplexF64})::Vector{ComplexF64}
                return remotecall_fetch(
                    F.setting[idx][:mode].adjoint,
                    F.transforms[idx][1],
                    F.transforms[idx][2],
                    f,
                )
            end

            N = prod(F.setting[idx][:bandwidths] .- 1)
            M = size(F.X, 2)
            return LinearMap{ComplexF64}(trafo_exp, adjoint_exp, M, N)
        end
    end
end


"""
`getmatrix(F)`

# Input
 * `F::GroupedTransform`

# Output
 * `F_direct::Array{ComplexF64}` ... Matrix representation of the transform
"""
function get_matrix(F::GroupedTransform)
    s1 = F.setting[1]
    F_direct = s1[:mode].get_matrix(s1[:bandwidths], F.X[s1[:u], :])
    for (idx, s) in enumerate(F.setting)
        idx == 1 && continue
        F_direct = hcat(F_direct, s[:mode].get_matrix(s[:bandwidths], F.X[s[:u], :]))
    end
    return F_direct
end
