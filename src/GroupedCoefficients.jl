abstract type GroupedCoefficients end

@doc raw"""
    GroupedCoefficientsComplex

A struct to hold complex coefficients belonging to indices in a grouped index set 

```math
    \mathcal{I}_{\pmb{N}}(U) = \left\{ \pmb{k} \in \Z^d : \mathrm{supp} \pmb{k} \in U, \pmb{k}_{\mathrm{supp} \pmb{k}} \in [- \frac{N_{\mathrm{supp} \pmb{k} }}{2}, \frac{N_{ \mathrm{supp} \pmb{k} }}{2} - 1 ) \right\}.
```

# Fields
* `setting` - uniquely describes the setting such as the bandlimits ``N_{\pmb u}``, see also [`get_setting(system::String,d::Int,ds::Int,N::Vector{Int})::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}}`](@ref) and [`get_setting(system::String,U::Vector{Vector{Int}},N::Vector{Int})::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}}`](@ref)
* `data` - the vector of coefficients 

# Constructor
    GroupedCoefficientsComplex( setting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}}, data::Union{Vector{ComplexF64},Nothing} = nothing ) 

# Additional Constructor
    GroupedCoefficients( setting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}}, data::Union{Vector{ComplexF64},Nothing} = nothing ) 
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
* `setting` - uniquely describes the setting such as the bandlimits ``N_{\pmb u}``, see also [`get_setting(system::String,d::Int,ds::Int,N::Vector{Int})::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}}`](@ref) and [`get_setting(system::String,U::Vector{Vector{Int}},N::Vector{Int})::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}}`](@ref)
* `data` - the vector of coefficients 

# Constructor
    GroupedCoefficientsReal( setting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}}, data::Union{Vector{Float64},Nothing} = nothing ) 

# Additional Constructor
    GroupedCoefficients( setting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}}, data::Union{Vector{Float64},Nothing} = nothing ) 
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
    setting,
    data::Union{Vector{ComplexF64},Vector{Float64},Nothing} = nothing,
)
    if setting[1][:mode] == NFFTtools
        return GroupedCoefficientsComplex(setting, data)
    elseif setting[1][:mode] == NFCTtools
        return GroupedCoefficientsReal(setting, data)
    end
end

@doc raw"""
    Base.:getindex( fhat, u )

This function overloads getindex of GroupedCoefficients such that you can do `fhat[[1,3]]` to obtain the basis coefficients of the corresponding ANOVA term (or the corresponding support)

# Input
* `fhat` - a GroupedCoefficients object
* `u` - a vector of integers
"""
function Base.:getindex(fhat::GroupedCoefficients, u::Vector{<:Integer})::Vector{<:Number}
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
    Base.:getindex( fhat, idx )

This function overloads getindex of GroupedCoefficients such that you can do `fhat[1]` to obtain the basis coefficient

# Input
* `fhat` - a GroupedCoefficients object
* `idx` - an integer
"""
function Base.:getindex(fhat::GroupedCoefficients, idx::Int64)::Number
    return fhat.data[idx]
end

@doc raw"""
    Base.:setindex!( fhat, fhatu, idx )

This function overloads setindex of GroupedCoefficients such that you can do `fhat[[1,3]] = [1 2 3]` to set the basis coefficients of the corresponding ANOVA term

# Input
* `fhat` - a GroupedCoefficients object
* `fhatu` - the coefficients belonging to the term ``f_{\pmb u}``
* `u` - a vector of integers
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
    Base.:setindex!( fhat, z, idx )

This function overloads setindex of GroupedCoefficients such that you can do `fhat[1] = 3` to set the basis coefficients

# Input
* `fhat` - a GroupedCoefficients object
* `z` - the coefficient
* `idx` - an integer
"""
function Base.:setindex!(fhat::GroupedCoefficients, z::Number, idx::Int64)
    fhat.data[idx] = z
end

@doc raw"""
    vec( fhat )

This function returns the vector of basis coefficients of fhat. This is useful for working with `lsqr` or similar.

# Input
* `fhat` - a GroupedCoefficients object
"""
Base.:vec(fhat::GroupedCoefficients)::Vector{<:Number} = fhat.data

@doc raw"""
    *( z, fhat )

This function defines the multiplication of a number with the GroupedCoefficients

# Input
* `z` - a number
* `fhat` - a GroupedCoefficients object
"""
Base.:*(α::Number, fhat::GroupedCoefficients) =
    GroupedCoefficients(fhat.setting, α * vec(fhat))

@doc raw"""
    +( z, fhat )

This function defines the addition of two GroupedCoefficients objects

# Input
* `fhat` - a GroupedCoefficients object
* `ghat` - a GroupedCoefficients object
"""
function Base.:+(fhat::GroupedCoefficients, ghat::GroupedCoefficients)::GroupedCoefficients
    if fhat.setting == ghat.setting
        return GroupedCoefficients(fhat.setting, vec(fhat) + vec(ghat))
    else
        error("Settings mismatch.")
    end
end

@doc raw"""
    -( z, fhat )

This function defines the subtraction of two GroupedCoefficients objects

# Input
* `fhat` - a GroupedCoefficients object
* `ghat` - a GroupedCoefficients object
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
    return [sqrt(sum(abs.(fhat[s[:u]]) .^ 2)) for s in fhat.setting]
end


function norms(fhat::GroupedCoefficients, what::GroupedCoefficients)::Vector{Float64}
    return [sqrt(sum(real.(what[s[:u]]) .* abs.(fhat[s[:u]]) .^ 2)) for s in fhat.setting]
end
