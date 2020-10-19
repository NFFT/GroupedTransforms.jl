"""
`GroupedCoeff`

The sturct `GroupedCoeff` represents grouped coefficients

# properties
 * `setting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int}, Module, Vector{Int}}}}` ... vector of the dimensions, mode, and bandwidths for each term/group
 * `data::Vector{ComplexF64}` ... all coefficients

# constructor
`GroupedCoeff(setting)` constructs empty coefficients
`GroupedCoeff(setting, data)` constructs coefficients from `data`

## Input
 * `setting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int}, Module, Vector{Int}}}}` ... vector of the dimensions, mode, and bandwidths for each term/group
 * `data{ComplexF64}` ... coefficients
"""
struct GroupedCoeff
  setting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int}, Module, Vector{Int}}}}
  data::Vector{ComplexF64}
  
  function GroupedCoeff(setting, data::Union{Vector{ComplexF64}, Nothing} = nothing)
    try
      global N = sum( s -> s[:mode].datalength(s[:bandwidths]), setting)
    catch
      error("The mode is not supportet yet or does not have the function datalength.")
    end
    isnothing(data) && ( data = Vector{ComplexF64}(undef, N) ) # if data uninitialized then initialize it
    if length(data) != N
      error("the supplied data vector has the wrong length.")
    end
    this = new(setting, data)
  end
end



"""
This function overloads getindex of GroupedCoeff such that you can do
  fhat[[1,3]]
to obtain the Fourier coefficients of the corresponding ANOVA term
"""
function Base.:getindex(fhat::GroupedCoeff, u::Vector)::Vector{ComplexF64}
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
to obtain the Fourier coefficients of the corresponding ANOVA term
"""
function Base.:getindex(fhat::GroupedCoeff, idx::Int64)::ComplexF64
  return fhat.data[idx]
end


"""
This function overloads setindex of GroupedCoeff such that you can do
  fhat[[1,3]] = [1 2 3]
to set the Fourier coefficients of the corresponding ANOVA term
"""
function Base.:setindex!(fhat::GroupedCoeff, fhatu::Vector{ComplexF64}, u::Vector)
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
to set the Fourier coefficients
"""
function Base.:setindex!(fhat::GroupedCoeff, z::ComplexF64, idx::Int64)
  fhat.data[idx] = z
end


"""
`fhat = vec(fhat::GroupedCoeff)`

This function returns the vector of Fourier coefficients of `GroupedCoeff`. This is usefull for working with `lsqr` or similar.
"""
Base.:vec(fhat::GroupedCoeff)::Vector{ComplexF64} = fhat.data


Base.:*(α::Number, fhat::GroupedCoeff) = GroupedCoeff(fhat.setting, α*vec(fhat))


Base.:+(fhat::GroupedCoeff, ghat::GroupedCoeff) = GroupedCoeff(fhat.setting, vec(fhat)+vec(ghat))


Base.:-(fhat::GroupedCoeff, ghat::GroupedCoeff) = fhat+(-1*ghat)


function set_data!(fhat::GroupedCoeff, data::Vector{ComplexF64})
  (fhat.data)[:] = data
end


function norms(fhat::GroupedCoeff)::Vector{Float64}
  return [ sqrt(sum(abs.(fhat[s[:u]]).^2)) for s in fhat.setting ]
end


function norms(fhat::GroupedCoeff, what::GroupedCoeff)::Vector{Float64}
  return [ sqrt(sum(real.(what[s[:u]]).*abs.(fhat[s[:u]]).^2)) for s in fhat.setting ]
end
