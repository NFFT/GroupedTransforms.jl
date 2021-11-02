module NFCTtools

using NFFT3
using LinearMaps

"""
`N = datalength(bandwidths)`

# Input:
 * `bandwidths::Vector{Int}`

# Output:
 * `N::Int` ... length of a Fourier-coefficient with the given bandwidths
"""
function datalength(bandwidths::Vector{Int})::Int
  return prod(bandwidths.-1)
end


"""
`freq = nfct_index_set_without_zeros(bandwidths)`

# Input:
 * `bandwidths::Vector{Int}`

# Output:
 * `freq::Array{Int}` ... all frequencies of the full cube without any vector having a zero entry
"""
function nfct_index_set_without_zeros(bandwidths::Vector{Int})::Array{Int}
  d = length(bandwidths)
  d == 0 && return [0,]
  d == 1 && return collect(Int.([1:1;2:bandwidths[1]-1]))

  tmp = Tuple([ Int.([1:1;2:bw-1]) for bw in bandwidths ])
  tmp = Iterators.product(tmp...)
  freq = Matrix{Int}(undef, d, prod(bandwidths.-1))
  for (m, x) in  enumerate(tmp)
    freq[:, m] = [reverse(x)...]
  end
  return freq
end


"""
`freq = nfct_index_set(bandwidths)`

# Input:
 * `bandwidths::Vector{Int}`

# Output:
 * `freq::Array{Int}` ... all frequencies of the full cube
"""
function nfct_index_set(bandwidths::Vector{Int})::Array{Int}
  d = length(bandwidths)
  d == 0 && return [0,]
  d == 1 && return collect(Int.(0:bandwidths[1]-1))

  tmp = Tuple([ Int.(0:bw-1) for bw in bandwidths ])
  tmp = Iterators.product(tmp...)
  freq = Matrix{Int}(undef, d, prod(bandwidths))
  for (idx, x) in  enumerate(tmp)
    freq[:, idx] = [reverse(x)...]
  end
  return freq
end


"""
`mask = nfct_index_set(bandwidths)`

# Input:
 * `bandwidths::Vector{Int}`

# Output:
 * `mask::BitArray{1}` ... mask with size of the full cube having zeros whereever a frequency has at least one zero-element and vice-versa
"""
function nfct_mask(bandwidths::Vector{Int})::BitArray{1}
  freq = nfct_index_set(bandwidths)
  nfct_mask = BitArray{1}
  if length(size(freq)) == 1
    return ( freq .!= 0 )
  else
    return .![ ( 0 in n ) for n in eachcol(freq) ]
  end
end

"""
trafos::Vector{LinearMap{Float64}}
This vector is local to the module on every worker.  It stores the transformations in order to access them later.
"""
trafos = Vector{LinearMap{Float64}}(undef,1)

"""
`F = get_transform(bandwidths, X)

# Input:
 * `bandwidths::Vector{Int}`
 * `X::Array{Float64}` ... nodes in |u| x M format

# Output:
 * `F::LinearMap{Float64}` ... Linear map of the Fourier-transform implemented by the NFCT
"""
function get_transform(bandwidths::Vector{Int}, X::Array{Float64})::Int64
  if size(X, 1) == 1
    X = vec(X)
    d = 1
    M = length(X)
  else
    (d, M) = size(X)
  end

  if bandwidths == []
    idx = length(trafos)
    trafos[idx] = LinearMap{Float64}(fhat -> fill(fhat[1], M), f -> [sum(f)], M, 1)
    append!( trafos, Vector{LinearMap{Float64}}(undef,1) )
    return idx
  end

  mask = nfct_mask(bandwidths)

  plan = NFCT(Tuple(bandwidths), M)
  plan.x = X

  function trafo(fhat::Vector{Float64})::Vector{Float64}
    plan.fhat = zeros(Float64, length(mask))
    plan.fhat[mask] = fhat
    nfct_trafo(plan)
    return plan.f
  end

  function adjoint(f::Vector{Float64})::Vector{Float64}
    plan.f = f
    nfct_adjoint(plan)
    return plan.fhat[mask]
  end

  N = prod(bandwidths.-1)
  idx = length(trafos)
  trafos[idx] = LinearMap{Float64}(trafo, adjoint, M, N)
  append!( trafos, Vector{LinearMap{Float64}}(undef,1) )
  return idx
end



"""
`F = get_matrix(bandwidths, X)

# Input:
 * `bandwidths::Vector{Int}`
 * `X::Array{Float64}` ... nodes in |u| x M format

# Output:
 * `F::Array{ComplexF64}` ... Matrix of the Fourier-transform
"""
function get_matrix(bandwidths::Vector{Int}, X::Array{Float64})::Array{Float64}
  if size(X, 1) == 1
    X = vec(X)
    d = 1
    M = length(X)
  else
    (d, M) = size(X)
  end

  if bandwidths == []
    return ones(Float64, M, 1)
  end

  if d == 1
    freq = nfct_index_set_without_zeros(bandwidths)
    F_direct = [ cos.(2*pi*(x.*n)) for x in vec(X), n in freq ]
  else
    freq = nfct_index_set_without_zeros(bandwidths)
    F_direct = [ prod(cos.(2*pi*(x.*n))) for x in eachcol(X), n in eachcol(freq) ]
  end
  return F_direct
end
end
