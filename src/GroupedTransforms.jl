module GroupedTransforms

using LinearMaps
using Distributed
using Combinatorics

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

include("NFFTtools.jl")
include("NFCTtools.jl")

export NFCTtools
export NFFTtools

systems = Dict("exp" => NFFTtools, "cos" => NFCTtools)

function get_setting(
    system::String,
    d::Int,
    ds::Int,
    N::Vector{Int},
)::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}}
    if !haskey(systems, system)
        error("System not found.")
    end
    if length(N) != ds
        error("N must have ds entries.")
    end
    tmp = vcat([0], N)
    U = GroupedTransforms.get_superposition_set(d, ds)
    bandwidths = [fill(tmp[length(u)+1], length(u)) for u in U]
    return [
        (u = U[idx], mode = systems[system], bandwidths = bandwidths[idx]) for
        idx = 1:length(U)
    ]
end

function get_setting(
    system::String,
    U::Vector{Vector{Int}},
    N::Vector{Int},
)::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}}
    if !haskey(systems, system)
        error("System not found.")
    end
    if length(N) != length(U)
        error("N must have |U| entries.")
    end
    bws = Vector{Vector{Int}}(undef, length(U))
    for i = 1:length(U)
        u = U[i]
        if u == []
            bws[i] = fill(0, length(u))
        else
            bws[i] = fill(N[i], length(u))
        end
    end

    return [
        (u = U[idx], mode = systems[system], bandwidths = bws[idx]) for idx = 1:length(U)
    ]
end

include("GroupedCoefficients.jl")
include("GroupedTransform.jl")

export GroupedTransform
export get_matrix
export GroupedCoefficients
export set_data!
export norms
export get_superposition_set

end # module
