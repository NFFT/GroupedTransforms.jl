using SparseArrays
@doc raw"""
    GroupedTransform

A struct to describe a GroupedTransformation

# Fields
* `system::String` - choice of `"exp"` or `"cos"` or `"chui1"` or `"chui2"` or `"chui3"` or `"chui4"` or `"expcos"`
* `setting::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{Bool}}}}` - vector of the dimensions, mode, bandwidths and bases for each term/group, see also [`get_setting(system::String,d::Int,ds::Int,N::Vector{Int},dcos::Vector{Bool})::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{Bool}}}}`](@ref) and [`get_setting(system::String,U::Vector{Vector{Int}},N::Vector{Int},dcos::Vector{Bool})::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{Bool}}}}`](@ref)
* `X::Array{Float64}` - array of nodes
* `transforms::Vector{Tuple{Int64,Int64}}` - holds the low-dimensional sub transformations
* `dcos::Vector{Bool}` - holds for every dimension if a cosinus basis [true] or exponential basis [false] is used

# Constructor
    GroupedTransform( system, setting, X, dcos::Vector{Bool} = Vector{Bool}([]) )

# Additional Constructor
    GroupedTransform( system, d, ds, N::Vector{Int}, X, dcos::Vector{Bool} = Vector{Bool}([]) )
    GroupedTransform( system, U, N, X, dcos::Vector{Bool} = Vector{Bool}([]) )
"""
struct GroupedTransform
    system::String
    setting::Vector{
        NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{Bool}}}
    }
    X::Array{Float64}
    transforms::Vector{Int64}
    dcos::Vector{Bool}

    function GroupedTransform(
        system::String,
        setting::Vector{
            NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{Bool}}}
        },
        X::Array{Float64},
        dcos::Vector{Bool} = Vector{Bool}([]),
    )

        
        if !haskey(systems, system)
            error("System not found.")
        end

        if system == "expcos"
            if length(dcos) == 0
                error("please call GroupedTransform with dcos for a NFFCT transform.")
            end
            if length(dcos) != size(X)[1]
                error("dcos must have an entry for every dimension.")
            end
        end

        if (system == "exp"  || system =="chui1" || system =="chui2"||system =="chui3"||system =="chui4")
            if (minimum(X) < -0.5) || (maximum(X) >= 0.5)
                error("Nodes must be between -0.5 and 0.5.")
            end
        elseif system == "cos"
            if (minimum(X) < 0) || (maximum(X) > 0.5)
                error("Nodes must be between 0 and 0.5.")
            end
        elseif system == "expcos"
            if sum(dcos)>0 
                if (minimum(X[dcos,:]) < 0) || (maximum(X[dcos,:]) > 1)
                    error("Nodes must be between 0 and 0.5 for cosinus dimensions.")
                end
            end
            if sum(.!dcos)>0 
                if (minimum(X[(.!dcos),:]) < -0.5) || (maximum(X[(.!dcos),:]) > 0.5)
                    error("Nodes must be between -0.5 and 0.5 for exponentional dimensions.")
            
                end
            end
        end

        transforms = Vector{Int64}(undef,length(setting))

        for (idx, s) in enumerate(setting)
            if system =="chui1"
                transforms[idx] = s[:mode].get_transform(s[:bandwidths], X[s[:u], :], 1)
            elseif system =="chui2"
                transforms[idx] = s[:mode].get_transform(s[:bandwidths], X[s[:u], :], 2)
            elseif system =="chui3"
                transforms[idx] = s[:mode].get_transform(s[:bandwidths], X[s[:u], :], 3)
            elseif system =="chui4"
                transforms[idx] = s[:mode].get_transform(s[:bandwidths], X[s[:u], :], 4)
            elseif system == "expcos"
                transforms[idx] = s[:mode].get_transform(s[:bandwidths], X[s[:u], :], s[:bases])
            else
                transforms[idx] = s[:mode].get_transform(s[:bandwidths], X[s[:u], :])
            end
        end

        new(system, setting, X, transforms, dcos)
    end
end

function GroupedTransform(
    system::String,
    d::Int,
    ds::Int,
    N::Vector{Int},
    X::Array{Float64},
    dcos::Vector{Bool} = Vector{Bool}([]),
    #m::Int64 = 1,
)
    s = get_setting(system, d, ds, N, dcos)
    return GroupedTransform(system, s, X, dcos)
end

function GroupedTransform(
    system::String,
    U::Vector{Vector{Int}},
    N::Vector{Int},
    X::Array{Float64},
    dcos::Vector{Bool} = Vector{Bool}([]),
    #m::Int64 = 1,
)
    s = get_setting(system, U, N, dcos)
    return GroupedTransform(system, s, X, dcos)
end

@doc raw"""
    *( F::GroupedTransform, fhat::GroupedCoefficients )::Vector{<:Number}

Overloads the `*` notation in order to achieve `f = F*fhat`.
"""
function Base.:*(F::GroupedTransform, fhat::GroupedCoefficients)::Vector{<:Number}
    if F.setting != fhat.setting
        error("The GroupedTransform and the GroupedCoefficients have different settings")
    end

    return sum(i -> (F.setting[i][:mode].trafos[F.transforms[i]]) * (fhat[F.setting[i][:u]]), 1:length(F.transforms))
end

@doc raw"""
    *( F::GroupedTransform, f::Vector{<:Number} )::GroupedCoefficients

Overloads the * notation in order to achieve the adjoint transform `f = F*f`.
"""
function Base.:*(F::GroupedTransform, f::Vector{<:Number})::GroupedCoefficients
    fhat = GroupedCoefficients(F.setting)
    for i = 1:length(F.transforms)
        fhat[F.setting[i][:u]] = (F.setting[i][:mode].trafos[F.transforms[i]])' * f
    end
    return fhat
end

@doc raw"""
    adjoint( F::GroupedTransform )::GroupedTransform

Overloads the `F'` notation and gives back the same GroupdTransform. GroupedTransform decides by the input if it is the normal trafo or the adjoint so this is only for convinience.
"""
function Base.:adjoint(F::GroupedTransform)::GroupedTransform
    return F
end

@doc raw"""
    F::GroupedTransform[u::Vector{Int}]::LinearMap{<:Number} or SparseArray

This function overloads getindex of GroupedTransform such that you can do `F[[1,3]]` to obtain the transform of the corresponding ANOVA term defined by `u`.
"""
function Base.:getindex(F::GroupedTransform, u::Vector{Int})#::LinearMap{<:Number}
    idx = findfirst(s -> s[:u] == u, F.setting)
    if isnothing(idx)
        error("This term is not contained")
    else
        if F.system == "cos"
            function trafo(fhat::Vector{Float64})::Vector{Float64}
                return F.setting[idx][:mode].trafos[F.transforms[idx]](fhat)
            end

            function adjoint(f::Vector{Float64})::Vector{Float64}
                return F.setting[idx][:mode].trafos[F.transforms[idx]]'(f)
            end

            N = prod(F.setting[idx][:bandwidths] .- 1)
            M = size(F.X, 2)
            return LinearMap{Float64}(trafo, adjoint, M, N)
        elseif (F.system == "exp" || F.system == "expcos")
            function trafo(fhat::Vector{ComplexF64})::Vector{ComplexF64}
                return F.setting[idx][:mode].trafos[F.transforms[idx]](fhat)
            end

            function adjoint(f::Vector{ComplexF64})::Vector{ComplexF64}
                return F.setting[idx][:mode].trafos[F.transforms[idx]]'(f)
            end

            N = prod(F.setting[idx][:bandwidths] .- 1)
            M = size(F.X, 2)
            return LinearMap{ComplexF64}(trafo, adjoint, M, N)

        elseif F.system == "chui1" || F.system == "chui2"  || F.system == "chui3"||F.system == "chui4"
            #S = SparseMatrixCSC{Float64, Int}
            S = (F.setting[idx][:mode].trafos[F.transforms[idx]])
            return SparseMatrixCSC{Float64, Int}(S)
        end
    end
end


@doc raw"""
    get_matrix( F::GroupedTransform )::Matrix{<:Number}

This function returns the actual matrix of the transformation. This is not available for the wavelet basis
"""
function get_matrix(F::GroupedTransform)::Matrix{<:Number}
    if F.system == "expcos"
        s1 = F.setting[1]
        F_direct = s1[:mode].get_matrix(s1[:bandwidths], F.X[s1[:u], :], s1[:bases])
        for (idx, s) in enumerate(F.setting)
            idx == 1 && continue
            F_direct = hcat(F_direct, s[:mode].get_matrix(s[:bandwidths], F.X[s[:u], :], s[:bases]))
        end
    else
        s1 = F.setting[1]
        F_direct = s1[:mode].get_matrix(s1[:bandwidths], F.X[s1[:u], :])
        for (idx, s) in enumerate(F.setting)
            idx == 1 && continue
            F_direct = hcat(F_direct, s[:mode].get_matrix(s[:bandwidths], F.X[s[:u], :]))
        end
    end
    return F_direct
end