module GroupedTransforms

using LinearMaps
using Distributed
using Combinatorics
using Aqua
using LinearAlgebra
using Test
using ToeplitzMatrices

@doc raw"""
    get_superposition_set( d::Int, ds::Int )::Vector{Vector{Int}}

This function returns ``U^{(d,ds)} = \{  \pmb u \subset \{1,2,\dots,d\} : |\pmb u| \leq ds \}``.
"""
function get_superposition_set(d::Int, ds::Int)::Vector{Vector{Int}}
    return [[[]]; collect(powerset(1:d, 1, ds))]
end

include("NFFTtools.jl")
include("NFCTtools.jl")
include("CWWTtools.jl")

export NFCTtools
export NFFTtools
export CWWTtools

systems = Dict("exp" => NFFTtools, "cos" => NFCTtools, "chui1" => CWWTtools, "chui2" => CWWTtools, "chui3" => CWWTtools, "chui4" => CWWTtools)

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

function get_NumFreq(
    setting::Vector{
        NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}},
    },
)::Int
if setting[1].mode == CWWTtools
    function datalength(bandwidths::Vector{Int})::Int
        if bandwidths == []
            return 1
        elseif length(bandwidths) == 1
            return 2^(bandwidths[1]+1)-1
        elseif length(bandwidths) == 2
            return 2^(bandwidths[1]+1)*bandwidths[1]+1
        elseif length(bandwidths) == 3
            n = bandwidths[1]
            return 2^n*n^2+2^n*n+2^(n+1)-1
    else
        d = length(bandwidths)
        n = bandwidths[1]
        tmp = 0
        for i =0:n
            tmp += 2^i*binomial(i+d-1,d-1)
        end
        return s
    end
    end
    return sum(s -> datalength(s[:bandwidths]), setting)

else
    return sum(s -> prod(s[:bandwidths] .- 1), setting)
end
end

function get_IndexSet(
    setting::Vector{
        NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}},
    },
    d::Int,
)::Matrix{Int}
    nf = get_NumFreq(setting)
    index_set = zeros(Int64, d, nf)
    idx = 1

    for i = 1:length(setting)
        s = setting[i]
        if s[:u] == []
            idx += 1
            continue
        end
        nf_u = prod(s[:bandwidths] .- 1)
        if s[:mode] == NFFTtools
            index_set_u = s[:mode].nfft_index_set_without_zeros(s[:bandwidths])
        elseif s[:mode] == NFCTtools
            index_set_u = s[:mode].nfct_index_set_without_zeros(s[:bandwidths])
        end

        if length(s[:u]) == 1
            index_set[s[:u][1], idx:(idx+nf_u-1)] = index_set_u
        else
            index_set[s[:u], idx:(idx+nf_u-1)] = index_set_u
        end

        idx += nf_u
    end
    return index_set
end

include("GroupedCoefficients.jl")
include("GroupedTransform.jl")

export GroupedTransform
export get_matrix
export GroupedCoefficients
export set_data!
export norms
export get_superposition_set
export get_NumFreq
export get_IndexSet

end # module
