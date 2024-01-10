module GroupedTransforms

using LinearMaps
using Distributed
using Combinatorics
using Aqua
using LinearAlgebra
using Test
using ToeplitzMatrices
using Base.Threads
#using Folds
#using ThreadsX

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
include("NFMTtools.jl")

export NFCTtools
export NFFTtools
export CWWTtools
export NFMTtools

systems = Dict("exp" => NFFTtools, "cos" => NFCTtools, "chui1" => CWWTtools, "chui2" => CWWTtools, "chui3" => CWWTtools, "chui4" => CWWTtools, "mixed" => NFMTtools)

function get_setting(
    system::String,
    d::Int,
    ds::Int,
    N::Vector{Int},
    basis_vect::Vector{String} = Vector{String}([]),
)::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{String}}}}
    if !haskey(systems, system)
        error("System not found.")
    end
    if length(N) != ds
        error("N must have ds entries.")
    end
    tmp = vcat([0], N)
    U = GroupedTransforms.get_superposition_set(d, ds)
    bandwidths = [fill(tmp[length(u)+1], length(u)) for u in U]
    if systems[system] == NFMTtools
        if length(basis_vect) == 0
            error("please call get_setting with basis_vect for a NFMT transform.")
        end
        if length(basis_vect) != d
            error("basis_vect must have an entry for every dimension.")
        end
        return [
            (u = U[idx], mode = systems[system], bandwidths = bandwidths[idx], bases = basis_vect[U[idx]]) for idx = 1:length(U)
        ]       
    else
        return [
            (u = U[idx], mode = systems[system], bandwidths = bandwidths[idx], bases = []) for idx = 1:length(U)
        ]
    end
end

function get_setting(
    system::String,
    U::Vector{Vector{Int}},
    N::Vector{Int},
    basis_vect::Vector{String} = Vector{String}([]),
)::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{String}}}}
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

    if systems[system] == NFMTtools
        if length(basis_vect) == 0
            error("please call get_setting with basis_vect for a NFMT transform.")
        end
        if length(basis_vect) < maximum(U)[1]
            error("basis_vect must have an entry for every dimension.")
        end
        return [
            (u = U[idx], mode = systems[system], bandwidths = bws[idx], bases = basis_vect[U[idx]]) for idx = 1:length(U)
        ]       
    else
        return [
            (u = U[idx], mode = systems[system], bandwidths = bws[idx], bases = []) for idx = 1:length(U)
        ]
    end
end

function get_setting(
    system::String,
    U::Vector{Vector{Int}},
    N::Vector{Vector{Int}},
    basis_vect::Vector{String} = Vector{String}([]),
)::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{String}}}}
    if !haskey(systems, system)
        error("System not found.")
    end
    bws = Vector{Vector{Int}}(undef, length(U))
    for i = 1:length(U)
        u = U[i]
        if u == []
            bws[i] = fill(0, length(u))
        else
            if length(N[i])!=length(u)
                error("Vector N has for the set", u, "not the right length")
            end
            bws[i] = N[i]
        end
    end

    if systems[system] == NFMTtools
        if length(basis_vect) == 0
            error("please call get_setting with basis_vect for a NFMT transform.")
        end
        if length(basis_vect) < maximum(U)[1]
            error("basis_vect must have an entry for every dimension.")
        end
        return [
            (u = U[idx], mode = systems[system], bandwidths = bws[idx], bases = basis_vect[U[idx]]) for idx = 1:length(U)
        ]       
    else
        return [
            (u = U[idx], mode = systems[system], bandwidths = bws[idx], bases = []) for idx = 1:length(U)
        ]
    end
end

function get_NumFreq(
    setting::Vector{
        NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{String}}}
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
        NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{String}}}
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
        elseif s[:mode] == NFMTtools
            index_set_u = s[:mode].nfmt_index_set_without_zeros(s[:bandwidths], s[:bases])
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
