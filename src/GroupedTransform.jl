using SparseArrays
@doc raw"""
    GroupedTransform

A struct to describe a GroupedTransformation

# Fields
* `system::String` - choice of `"exp"` or `"cos"` or `"chui1"` or `"chui2"` or `"chui3"` or `"chui4"` or `"mixed"`
* `setting::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{String}}}}` - vector of the dimensions, mode, bandwidths and bases for each term/group, see also [`get_setting(system::String,d::Int,ds::Int,N::Vector{Int},basis_vect::Vector{String})::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{String}}}}`](@ref) and [`get_setting(system::String,U::Vector{Vector{Int}},N::Vector{Int},basis_vect::Vector{String})::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{String}}}}`](@ref)
* `X::Array{Float64}` - array of nodes
* `transforms::Vector{LinearMap}` - holds the low-dimensional sub transformations
* `basis_vect::Vector{String}` - holds for every dimension if a cosinus basis [true] or exponential basis [false] is used

# Constructor
    GroupedTransform( system, setting, X, basis_vect::Vector{String} = Vector{String}([]) )

# Additional Constructor
    GroupedTransform( system, d, ds, N::Vector{Int}, X, basis_vect::Vector{String} = Vector{String}([]) )
    GroupedTransform( system, U, N, X, basis_vect::Vector{String} = Vector{String}([]) )
"""
struct GroupedTransform
    system::String
    setting::Vector{
        NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{String}}}
    }
    X::Array{Float64}
    transforms::Vector{LinearMap{<:Number}}
    basis_vect::Vector{String}

    function GroupedTransform(
        system::String,
        setting::Vector{
            NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{String}}}
        },
        X::Array{Float64},
        basis_vect::Vector{String} = Vector{String}([]),
    )

        
        if !haskey(systems, system)
            error("System not found.")
        end

        if system == "mixed"
            if length(basis_vect) == 0
                error("please call GroupedTransform with basis_vect for a NFMT transform.")
            end
            if length(basis_vect) != size(X)[1]
                error("basis_vect must have an entry for every dimension.")
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
        
        elseif system == "mixed"
            if sum(getindex.([NFMTtools.BASES],basis_vect).>0)>0 
                if (minimum(X[getindex.([NFMTtools.BASES],basis_vect).>0,:]) < 0) || (maximum(X[getindex.([NFMTtools.BASES],basis_vect).>0,:]) > 1)
                    error("Nodes must be between 0 and 1 for cosine or Chebyshev dimensions.")
                end
            end
            if sum(.!(getindex.([NFMTtools.BASES],basis_vect).>0))>0 
                if (minimum(X[(.!(getindex.([NFMTtools.BASES],basis_vect).>0)),:]) < -0.5) || (maximum(X[(.!(getindex.([NFMTtools.BASES],basis_vect).>0)),:]) > 0.5)
                    error("Nodes must be between -0.5 and 0.5 for exponentional dimensions.")
                end
            end
        end

        transforms = Vector{LinearMap{<:Number}}(undef, length(setting))

        for (idx, s) in enumerate(setting)
            if system =="chui1"
                transforms[idx] = s[:mode].get_transform(s[:bandwidths], X[s[:u], :], 1)
            elseif system =="chui2"
                transforms[idx] = s[:mode].get_transform(s[:bandwidths], X[s[:u], :], 2)
            elseif system =="chui3"
                transforms[idx] = s[:mode].get_transform(s[:bandwidths], X[s[:u], :], 3)
            elseif system =="chui4"
                transforms[idx] = s[:mode].get_transform(s[:bandwidths], X[s[:u], :], 4)
            elseif system == "mixed"
                transforms[idx] = s[:mode].get_transform(s[:bandwidths], X[s[:u], :], s[:bases])
            else
                transforms[idx] = s[:mode].get_transform( s[:bandwidths], X[s[:u], :])
            end
        end
        
        for (idx, s) in enumerate(setting)
            transforms[idx] = (f[idx][1], fetch(f[idx][2]))
        end


        new(system, setting, X, transforms, basis_vect)
    end
end

function GroupedTransform(
    system::String,
    d::Int,
    ds::Int,
    N::Vector{Int},
    X::Array{Float64},
    basis_vect::Vector{String} = Vector{String}([]),
    #m::Int64 = 1,
)
    s = get_setting(system, d, ds, N, basis_vect)
    return GroupedTransform(system, s, X, basis_vect)
end

function GroupedTransform(
    system::String,
    U::Vector{Vector{Int}},
    N::Vector{Int},
    X::Array{Float64},
    basis_vect::Vector{String} = Vector{String}([]),
    #m::Int64 = 1,
)
    s = get_setting(system, U, N, basis_vect)
    return GroupedTransform(system, s, X, basis_vect)
end

function GroupedTransform(
    system::String,
    U::Vector{Vector{Int}},
    N::Vector{Vector{Int}},
    X::Array{Float64},
    basis_vect::Vector{String} = Vector{String}([]),
    #m::Int64 = 1,
)
    s = get_setting(system, U, N, basis_vect)
    return GroupedTransform(system, s, X, basis_vect)
end

@doc raw"""
    *( F::GroupedTransform, fhat::GroupedCoefficients )::Vector{<:Number}

Overloads the `*` notation in order to achieve `f = F*fhat`.
"""
function Base.:*(F::GroupedTransform, fhat::GroupedCoefficients)::Vector{<:Number}
    if F.setting != fhat.setting
        error("The GroupedTransform and the GroupedCoefficients have different settings")
    end
    f = Vector{Task}(undef, length(F.transforms))
    for i in eachindex(F.transforms)
        f[i] = Threads.@spawn (F.transforms[i]) * (fhat[F.setting[i][:u]]) 
    end  
    #println(length(F.transforms))
    #return Folds.mapreduce(i -> (F.transforms[i]) * (fhat[F.setting[i][:u]]), +, 1:length(F.transforms))
    #return ThreadsX.sum((F.transforms[i]) * (fhat[F.setting[i][:u]]) for i=1:length(F.transforms))
    return sum(i -> fetch(f[i]), eachindex(F.transforms))
end

@doc raw"""
    *( F::GroupedTransform, f::Vector{<:Number} )::GroupedCoefficients

Overloads the * notation in order to achieve the adjoint transform `f = F*f`.
"""
function Base.:*(F::GroupedTransform, f::Vector{<:Number})::GroupedCoefficients
    #fhat = GroupedCoefficients(F.setting)
    #Threads.@threads for i in eachindex(F.transforms)
    #    fhat[F.setting[i][:u]] = (F.transforms[i])' * f
    #end
    
    fh = Vector{Task}(undef, length(F.transforms))
    for i in eachindex(F.transforms)
        fh[i] = Threads.@spawn (F.transforms[i])' * f
    end
    fhat = GroupedCoefficients(F.setting)
    for i in eachindex(F.transforms)
        fhat[F.setting[i][:u]] = fetch(fh[i])
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
function Base.:getindex(F::GroupedTransform, u::Vector{Int})::LinearMap{<:Number}
    idx = findfirst(s -> s[:u] == u, F.setting)
    if isnothing(idx)
        error("This term is not contained")
    else
        return F.transforms[idx]
    end
end

@doc raw"""
    get_matrix( F::GroupedTransform )::Matrix{<:Number}

This function returns the actual matrix of the transformation. This is not available for the wavelet basis
"""
function get_matrix(F::GroupedTransform)::Matrix{<:Number}
    if F.system == "chui1" || F.system == "chui2"  || F.system == "chui3"||F.system == "chui4"

        error("Direct computation with full matrix not supported for wavelet basis.")
    elseif F.system == "mixed"
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