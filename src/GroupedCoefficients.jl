abstract type GroupedCoefficients end

@doc raw"""
    GroupedCoefficientsComplex

A struct to hold complex coefficients belonging to indices in a grouped index set 

```math
    \mathcal{I}_{\pmb{N}}(U) = \left\{ \pmb{k} \in \Z^d : \mathrm{supp} \pmb{k} \in U, \pmb{k}_{\mathrm{supp} \pmb{k}} \in [- \frac{N_{\mathrm{supp} \pmb{k} }}{2}, \frac{N_{ \mathrm{supp} \pmb{k} }}{2} - 1 ) \right\}.
```

# Fields
* `setting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}}` - uniquely describes the setting such as the bandlimits ``N_{\pmb u}``, see also [`get_setting(system::String,d::Int,ds::Int,N::Vector{Int})::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}}`](@ref) and [`get_setting(system::String,U::Vector{Vector{Int}},N::Vector{Int})::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}}`](@ref)
* `data::Union{Vector{ComplexF64},Nothing}` - the vector of coefficients 

# Constructor
    GroupedCoefficientsComplex( setting, data = nothing ) 

# Additional Constructor
    GroupedCoefficients( setting, data = nothing ) 
"""
struct GroupedCoefficientsComplex <: GroupedCoefficients
    setting::Vector{
        NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}},
    }
    data::Vector{ComplexF64}

    function GroupedCoefficientsComplex(
        setting,
        data::Union{Vector{ComplexF64},Nothing} = nothing,
    )
        try
            N = sum(s -> s[:mode].datalength(s[:bandwidths]), setting)
            if isnothing(data)
                data = zeros(ComplexF64, N)
            end
            if length(data) != N
                error("the supplied data vector has the wrong length.")
            end
            return new(setting, data)
        catch
            error("The mode is not supportet yet or does not have the function datalength.")
        end
    end
end

@doc raw"""
    GroupedCoefficientsReal

A struct to hold real valued coefficients belonging to indices in a grouped index set 

```math
    \mathcal{I}_{\pmb{N}}(U) = \left\{ \pmb{k} \in \Z^d : \mathrm{supp} \pmb{k} \in U, \pmb{k}_{\mathrm{supp} \pmb{k}} \in [0, N_{\mathrm{supp} \pmb{k} } - 1 ] \right\}.
```

# Fields
* `setting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}}` - uniquely describes the setting such as the bandlimits ``N_{\pmb u}``, see also [`get_setting(system::String,d::Int,ds::Int,N::Vector{Int})::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}}`](@ref) and [`get_setting(system::String,U::Vector{Vector{Int}},N::Vector{Int})::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}}`](@ref)
* `data::Union{Vector{Float64},Nothing}` - the vector of coefficients 

# Constructor
    GroupedCoefficientsReal( setting, data = nothing ) 

# Additional Constructor
    GroupedCoefficients( setting, data = nothing ) 
"""
struct GroupedCoefficientsReal <: GroupedCoefficients
    setting::Vector{
        NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}},
    }
    data::Vector{Float64}

    function GroupedCoefficientsReal(
        setting,
        data::Union{Vector{Float64},Nothing} = nothing,
    )
        try
            N = sum(s -> s[:mode].datalength(s[:bandwidths]), setting)
            if isnothing(data)
                data = zeros(Float64, N)
            end
            if length(data) != N
                error("the supplied data vector has the wrong length.")
            end
            return new(setting, data)
        catch
            error("The mode is not supportet yet or does not have the function datalength.")
        end
    end
end

function GroupedCoefficients(
    setting::Vector{
        NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}},
    },
    data::Union{Vector{ComplexF64},Vector{Float64},Nothing} = nothing,
)
    if setting[1][:mode] == NFFTtools
        return GroupedCoefficientsComplex(setting, data)
    elseif setting[1][:mode] == NFCTtools
        return GroupedCoefficientsReal(setting, data)
    end
end

@doc raw"""
    fhat::GroupedCoefficients[u::Vector{Int}]

This function overloads getindex of GroupedCoefficients such that you can do `fhat[[1,3]]` to obtain the basis coefficients of the corresponding ANOVA term defined by `u`.
"""
function Base.:getindex(fhat::GroupedCoefficients, u::Vector{Int})::Vector{<:Number}
    start = 1
    for s in fhat.setting
        if s[:u] == u
            stop = start + s[:mode].datalength(s[:bandwidths]) - 1
            return fhat.data[start:stop]
        else
            start += s[:mode].datalength(s[:bandwidths])
        end
    end
    error("This term is not contained")
end

@doc raw"""
    fhat::GroupedCoefficients[idx::Int]

This function overloads getindex of GroupedCoefficients such that you can do `fhat[1]` to obtain the basis coefficient determined by `idx`.
"""
function Base.:getindex(fhat::GroupedCoefficients, idx::Int64)::Number
    return fhat.data[idx]
end

@doc raw"""
    fhat::GroupedCoefficients[u::Vector{Int}] = fhatu::Union{Vector{ComplexF64},Vector{Float64}}

This function overloads setindex of GroupedCoefficients such that you can do `fhat[[1,3]] = [1 2 3]` to set the basis coefficients of the corresponding ANOVA term defined by `u`.
"""
function Base.:setindex!(
    fhat::GroupedCoefficients,
    fhatu::Union{Vector{ComplexF64},Vector{Float64}},
    u::Vector{<:Integer},
)
    if (isa(fhat, GroupedCoefficientsComplex) && isa(fhatu, Vector{Float64})) ||
       (isa(fhat, GroupedCoefficientsReal) && isa(fhatu, Vector{ComplexF64}))
        error("Type mismatch.")
    end
    start = 1
    for s in fhat.setting
        if s[:u] == u
            stop = start + s[:mode].datalength(s[:bandwidths]) - 1
            fhat.data[start:stop] = fhatu
            return
        else
            start += s[:mode].datalength(s[:bandwidths])
        end
    end
    error("This term is not contained")
end

@doc raw"""
    fhat::GroupedCoefficients[idx::Int] = z::Number

This function overloads setindex of GroupedCoefficients such that you can do `fhat[1] = 3` to set the basis coefficient determined by `idx`.
"""
function Base.:setindex!(fhat::GroupedCoefficients, z::Number, idx::Int64)
    fhat.data[idx] = z
end

@doc raw"""
    vec( fhat::GroupedCoefficients )::Vector{<:Number}

This function returns the vector of the basis coefficients of fhat. This is useful for working with `lsqr` or similar.
"""
Base.:vec(fhat::GroupedCoefficients)::Vector{<:Number} = fhat.data

@doc raw"""
    *( z::Number, fhat::GroupedCoefficients )::GroupedCoefficients

This function defines the multiplication of a number with a GroupedCoefficients object.
"""
Base.:*(α::Number, fhat::GroupedCoefficients) =
    GroupedCoefficients(fhat.setting, α * vec(fhat))

@doc raw"""
    +( z::Number, fhat::GroupedCoefficients )::GroupedCoefficients

This function defines the addition of two GroupedCoefficients objects.
"""
function Base.:+(fhat::GroupedCoefficients, ghat::GroupedCoefficients)::GroupedCoefficients
    if fhat.setting == ghat.setting
        return GroupedCoefficients(fhat.setting, vec(fhat) + vec(ghat))
    else
        error("Settings mismatch.")
    end
end

@doc raw"""
    -( z::Number, fhat::GroupedCoefficients )::GroupedCoefficients

This function defines the subtraction of two GroupedCoefficients objects.
"""
Base.:-(fhat::GroupedCoefficients, ghat::GroupedCoefficients) = fhat + (-1 * ghat)

function set_data!(
    fhat::GroupedCoefficients,
    data::Union{Vector{ComplexF64},Vector{Float64}},
)
    if (isa(fhat, GroupedCoefficientsComplex) && isa(data, Vector{Float64})) ||
       (isa(fhat, GroupedCoefficientsReal) && isa(data, Vector{ComplexF64}))
        error("Type mismatch.")
    end
    (fhat.data)[:] = data
end

function norms(fhat::GroupedCoefficients)::Vector{Float64}
    setting = fhat.setting
    return [norm(fhat[setting[i][:u]]) for i = 1:length(setting)]
end

function norms(fhat::GroupedCoefficients, what::GroupedCoefficients)::Vector{Float64}
    c = GroupedCoefficients(fhat.setting, (sqrt.(real.(what.data))) .* fhat.data)
    setting = c.setting
    return [norm(c[setting[i][:u]]) for i = 1:length(setting)]
end
