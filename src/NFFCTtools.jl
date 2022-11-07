module NFFCTtools

#using NFFCT
include("/home/arch/git/NFFT3.jl/src/NFFCT.jl")
using LinearMaps

"""
`N = datalength(bandwidths)`

# Input:
 * `bandwidths::Vector{Int}`

# Output:
 * `N::Int` ... length of a Fourier-coefficient with the given bandwidths
"""
function datalength(bandwidths::Vector{Int})::Int
    return prod(bandwidths .- 1)
end

"""
`freq = nfct_index_set_without_zeros(bandwidths)`

# Input:
 * `bandwidths::Vector{Int}`
 * `dcos::Vector{Bool}`

# Output:
 * `freq::Array{Int}` ... all frequencies of the full cube without any vector having a zero entry
"""
function nffct_index_set_without_zeros(bandwidths::Vector{Int}, dcos::Vector{Bool})::Array{Int}
    d = length(bandwidths)
    d == 0 && return [0]
    d == 1 && dcos[1] && return collect([1:1; 2:bandwidths[1]-1])
    d == 1 && !dcos[1] && return collect([-bandwidths[1]÷2:-1; 1:bandwidths[1]÷2-1])

    bandwidths = reverse(bandwidths)
    dcos = reverse(dcos)
    tmp = Vector{Vector{Int64}}()
    for (idx, s) in enumerate(dcos)
        if s
            append!(tmp, [[1:1; 2:bandwidths[idx]-1]])
        else
            append!(tmp, [[-bandwidths[idx]÷2:-1; 1:bandwidths[idx]÷2-1]])
        end
    end
    tmp = Tuple(tmp)
    tmp = Iterators.product(tmp...)
    freq = Matrix{Int}(undef, d, prod(bandwidths .- 1))
    for (m, x) in enumerate(tmp)
        freq[:, m] = [reverse(x)...]
    end
    return freq
end

"""
`freq = nfct_index_set(bandwidths)`

# Input:
 * `bandwidths::Vector{Int}`
 * `dcos::Vector{Bool}`

# Output:
 * `freq::Array{Int}` ... all frequencies of the full cube
"""
function nffct_index_set(bandwidths::Vector{Int}, dcos::Vector{Bool})::Array{Int}
    d = length(bandwidths)
    d == 0 && return [0]
    d == 1 && dcos[1] && return collect([0:0; 1:bandwidths[1]-1])
    d == 1 && !dcos[1] && return collect([-bandwidths[1]÷2:0; 1:bandwidths[1]÷2-1])

    bandwidths = reverse(bandwidths)
    dcos = reverse(dcos)
    tmp = Vector{Vector{Int64}}()
    for i = range(1,d)
        if dcos[i]
            append!(tmp, [[0:0; 1:bandwidths[i]-1]])
        else
            append!(tmp, [[-bandwidths[i]÷2:0; 1:bandwidths[i]÷2-1]])
        end
    end
    tmp = Tuple(tmp)
    tmp = Iterators.product(tmp...)
    freq = Matrix{Int}(undef, d, prod(bandwidths))
    for (idx, x) in enumerate(tmp)
        freq[:, idx] = [reverse(x)...]
    end
    return freq
end

"""
`mask = nfct_index_set(bandwidths)`

# Input:
 * `bandwidths::Vector{Int}`
 * `dcos::Vector{Bool}`

# Output:
 * `mask::BitArray{1}` ... mask with size of the full cube having zeros whereever a frequency has at least one zero-element and vice-versa
"""
function nffct_mask(bandwidths::Vector{Int}, dcos::Vector{Bool})::BitArray{1}
    freq = nffct_index_set(bandwidths, dcos)
    nfft_mask = BitArray{1}
    if length(size(freq)) == 1
        return (freq .!= 0)
    else
        return .![(0 in n) for n in eachcol(freq)]
    end
end

"""
trafos::Vector{LinearMap{Float64}}
This vector is local to the module on every worker.  It stores the transformations in order to access them later.
"""
trafos = Vector{LinearMap{ComplexF64}}(undef, 1)

"""
`F = get_transform(bandwidths, X)

# Input:
 * `bandwidths::Vector{Int}`
 * `X::Array{Float64}` ... nodes in |u| x M format
 * `dcos::Vector{Bool}`

# Output:
 * `F::LinearMap{Float64}` ... Linear map of the Fourier-transform implemented by the NFCT
"""
function get_transform(bandwidths::Vector{Int}, X::Array{Float64}, dcos::Vector{Bool})::Int64
    if size(X, 1) == 1
        X = vec(X)
        d = 1
        M = length(X)
    else
        (d, M) = size(X)
    end

    if bandwidths == []
        idx = length(trafos)
        trafos[idx] = LinearMap{ComplexF64}(fhat -> fill(fhat[1], M), f -> [sum(f)], M, 1)
        append!(trafos, Vector{LinearMap{ComplexF64}}(undef, 1))
        return idx
    end

    mask = nffct_mask(bandwidths, dcos)
    N = Tuple(bandwidths)
    println(N)
    println(dcos)
    for (idx, s) in enumerate(dcos)
        if s
            N[idx] *= 2
        end
    end
    plan = NFFCT(Tuple(dcos), N, M, Tuple(2 * collect(N)), 12)
    plan.x = X

    function trafo(fhat::Vector{ComplexF64})::Vector{ComplexF64}
        plan.fhat = zeros(ComplexF64, length(mask))
        plan.fhat[mask] = fhat
        nffct_trafo(plan)
        return plan.f
    end

    function adjoint(f::Vector{ComplexF64})::Vector{ComplexF64}
        plan.f = f
        nffct_adjoint(plan)
        return plan.fhat[mask]
    end

    N = prod(bandwidths .- 1)
    idx = length(trafos)
    trafos[idx] = LinearMap{ComplexF64}(trafo, adjoint, M, N)
    append!(trafos, Vector{LinearMap{ComplexF64}}(undef, 1))
    return idx
end

function get_phi(x::Vector{Float64}, k::Vector{Int64}, dcos::Vector{Bool})::ComplexF64
    p = 1
    for (idx, s) in enumerate(dcos)
        if s
            if k[idx] ≠ 0
                p *= sqrt(2.0)*cos(2*pi*k[idx]*x[idx])
            end
        else
            p *= exp(-2.0*pi*im*k[idx]*x[idx])
        end
    end
    return p
end

"""
`F = get_matrix(bandwidths, X)

# Input:
 * `bandwidths::Vector{Int}`
 * `X::Array{Float64}` ... nodes in |u| x M format
 * `dcos::Vector{Bool}`

# Output:
 * `F::Array{ComplexF64}` ... Matrix of the Fourier-transform
"""
function get_matrix(bandwidths::Vector{Int}, X::Array{Float64}, dcos::Vector{Bool})::Array{ComplexF64}
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
        freq = nffct_index_set_without_zeros(bandwidths, dcos)
        F_direct = [get_phi(append!(Vector{Float64}(),x), append!(Vector{Int}(),n), dcos) for x in vec(X), n in freq]
    else
        freq = nffct_index_set_without_zeros(bandwidths, dcos)
        F_direct = [get_phi(Vector(x), Vector(n), dcos) for x in eachcol(X), n in eachcol(freq)]
    end

    return F_direct
end 

end