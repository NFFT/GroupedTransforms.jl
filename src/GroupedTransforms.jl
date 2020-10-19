module GroupedTransforms

using LinearMaps
using Distributed
using Combinatorics

export GroupedTransform
export get_matrix
export GroupedCoeff
export set_data!
export norms
export get_superposition_set

"""
`U = function get_superposition_set(d, ds)`

# Input:
 * `d::Int` ... dimension
 * `ds::Int` ... superposition dimension

# Output:
 * `U::Vector{Vector{Int}}` ... all sets of dimensions with at most ds dimensions
"""
function get_superposition_set(d::Int, ds::Int)::Vector{Vector{Int}}
  return [[[]]; collect(powerset(1:d, 1, ds))]
end

include("GroupedCoeff.jl")
include("GroupedTransform.jl")
include("NFFTtools.jl")
include("NFCTtools.jl")

end # module
