abstract type GroupedCoefficients end

"""
`GroupedCoefficientsComplex`

The struct `GroupedCoefficientsComplex` represents grouped coefficients

# properties
 * `setting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int}, Module, Vector{Int}}}}` ... vector of the dimensions, mode, and bandwidths for each term/group
 * `data::Vector{ComplexF64}` ... all coefficients

# constructor
`GroupedCoefficientsComplex(setting)` constructs empty coefficients
`GroupedCoefficientsComplex(setting, data)` constructs coefficients from `data`

## Input
 * `setting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int}, Module, Vector{Int}}}}` ... vector of the dimensions, mode, and bandwidths for each term/group
 * `data{ComplexF64}` ... coefficients
"""
struct GroupedCoefficientsComplex <: GroupedCoefficients
  setting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int}, Module, Vector{Int}}}}
  data::Vector{ComplexF64}
  
  function GroupedCoefficientsComplex(setting, data::Union{Vector{ComplexF64}, Nothing} = nothing)
    try
      N = sum( s -> s[:mode].datalength(s[:bandwidths]), setting)
      if isnothing(data)
        data = zeros( ComplexF64, N )
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

"""
`GroupedCoefficientsReal`

The struct `GroupedCoefficientsReal` represents grouped coefficients

# properties
 * `setting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int}, Module, Vector{Int}}}}` ... vector of the dimensions, mode, and bandwidths for each term/group
 * `data::Vector{ComplexF64}` ... all coefficients

# constructor
`GroupedCoefficientsReal(setting)` constructs empty coefficients
`GroupedCoefficientsReal(setting, data)` constructs coefficients from `data`

## Input
 * `setting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int}, Module, Vector{Int}}}}` ... vector of the dimensions, mode, and bandwidths for each term/group
 * `data{ComplexF64}` ... coefficients
"""
struct GroupedCoefficientsReal <: GroupedCoefficients
  setting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int}, Module, Vector{Int}}}}
  data::Vector{Float64}
  
  function GroupedCoefficientsReal(setting, data::Union{Vector{Float64}, Nothing} = nothing)
    try
      N = sum( s -> s[:mode].datalength(s[:bandwidths]), setting)
      if isnothing(data)
        data = zeros( Float64, N )
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

function GroupedCoefficients( setting, data::Union{Vector{ComplexF64}, Vector{Float64}, Nothing} = nothing ) 
  if setting[1][:mode] == NFFTtools
    return GroupedCoefficientsComplex( setting, data )
  elseif setting[1][:mode] == NFCTtools
    return GroupedCoefficientsReal( setting, data )
  end
end


"""
This function overloads getindex of GroupedCoefficients such that you can do
  fhat[[1,3]]
to obtain the basis coefficients of the corresponding ANOVA term (or the corresponding support)
"""
function Base.:getindex(fhat::GroupedCoefficients, u::Vector{<:Integer})::Vector{<:Number}
  start = 1
  for s in fhat.setting
    if s[:u] == u
      stop = start+s[:mode].datalength(s[:bandwidths])-1
      return fhat.data[start:stop]
    else
      start += s[:mode].datalength(s[:bandwidths])
    end
  end
  error( "This term is not contained" )
end


"""
This function overloads getindex of GroupedCoeff such that you can do
  fhat[1]
to obtain the basis coefficient
"""
function Base.:getindex(fhat::GroupedCoefficients, idx::Int64)::Number
  return fhat.data[idx]
end


"""
This function overloads setindex of GroupedCoeff such that you can do
  fhat[[1,3]] = [1 2 3]
to set the basis coefficients of the corresponding ANOVA term
"""
function Base.:setindex!(fhat::GroupedCoefficients, fhatu::Union{Vector{ComplexF64}, Vector{Float64}}, u::Vector{<:Integer})
  if ( isa(fhat,GroupedCoefficientsComplex) && isa(fhatu, Vector{Float64}) ) || ( isa(fhat,GroupedCoefficientsReal) && isa(fhatu, Vector{ComplexF64}) )
    error( "Type mismatch." )
  end
  start = 1
  for s in fhat.setting
    if s[:u] == u
      stop = start+s[:mode].datalength(s[:bandwidths])-1
      fhat.data[start:stop] = fhatu
      return
    else
      start += s[:mode].datalength(s[:bandwidths])
    end
  end
  error( "This term is not contained" )
end


"""
This function overloads setindex of GroupedCoeff such that you can do
  fhat[1] = 3
to set the basis coefficients
"""
function Base.:setindex!(fhat::GroupedCoefficients, z::Number, idx::Int64)
  fhat.data[idx] = z
end


"""
`fhat = vec(fhat::GroupedCoeff)`

This function returns the vector of Fourier coefficients of `GroupedCoeff`. This is usefull for working with `lsqr` or similar.
"""
Base.:vec(fhat::GroupedCoefficients)::Vector{<:Number} = fhat.data


Base.:*(α::Number, fhat::GroupedCoefficients) = GroupedCoefficients(fhat.setting, α*vec(fhat))


function Base.:+(fhat::GroupedCoefficients, ghat::GroupedCoefficients)::GroupedCoefficients
  if fhat.setting == ghat.setting
    return GroupedCoefficients(fhat.setting, vec(fhat)+vec(ghat))
  else 
    error( "Settings mismatch.")
  end
end

Base.:-(fhat::GroupedCoefficients, ghat::GroupedCoefficients) = fhat+(-1*ghat)

function set_data!(fhat::GroupedCoefficients, data::Union{Vector{ComplexF64}, Vector{Float64}})
  if ( isa(fhat,GroupedCoefficientsComplex) && isa(data, Vector{Float64}) ) || ( isa(fhat,GroupedCoefficientsReal) && isa(data, Vector{ComplexF64}) )
    error( "Type mismatch." )
  end
  (fhat.data)[:] = data
end


function norms(fhat::GroupedCoefficients)::Vector{Float64}
  return [ sqrt(sum(abs.(fhat[s[:u]]).^2)) for s in fhat.setting ]
end


function norms(fhat::GroupedCoefficients, what::GroupedCoefficients)::Vector{Float64}
  return [ sqrt(sum(real.(what[s[:u]]).*abs.(fhat[s[:u]]).^2)) for s in fhat.setting ]
end
