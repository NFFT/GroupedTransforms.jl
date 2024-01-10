module NFMTtools

using LinearMaps
using NFFT3

BASES = Dict("exp"=>0,"cos"=>1,"alg"=>2)

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
 * `basis_vect::Vector{String}`

# Output:
 * `freq::Array{Int}` ... all frequencies of the full cube without any vector having a zero entry
"""
function nfmt_index_set_without_zeros(bandwidths::Vector{Int}, basis_vect::Vector{String})::Array{Int}
    d = length(bandwidths)
    d == 0 && return [0]
    d == 1 && BASES[basis_vect[1]]>0 && return collect([1:1; 2:bandwidths[1]-1])
    d == 1 && BASES[basis_vect[1]]==0 && return collect([-bandwidths[1]÷2:-1; 1:bandwidths[1]÷2-1])

    bandwidths = reverse(bandwidths)
    basis_vect = reverse(basis_vect)
    tmp = Vector{Vector{Int64}}()
    for (idx, s) in enumerate(basis_vect)
        if BASES[s]>0
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
 * `basis_vect::Vector{String}`

# Output:
 * `freq::Array{Int}` ... all frequencies of the full cube
"""
function nfmt_index_set(bandwidths::Vector{Int}, basis_vect::Vector{String})::Array{Int}
    d = length(bandwidths)
    d == 0 && return [0]
    d == 1 && BASES[basis_vect[1]]>0 && return collect([0:0; 1:bandwidths[1]-1])
    d == 1 && BASES[basis_vect[1]]==0 && return collect([-bandwidths[1]÷2:0; 1:bandwidths[1]÷2-1])

    bandwidths = reverse(bandwidths)
    basis_vect = reverse(basis_vect)
    tmp = Vector{Vector{Int64}}()
    for i = range(1,d)
        if BASES[basis_vect[i]]>0
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
 * `basis_vect::Vector{String}`

# Output:
 * `mask::BitArray{1}` ... mask with size of the full cube having zeros whereever a frequency has at least one zero-element and vice-versa
"""
function nfmt_mask(bandwidths::Vector{Int}, basis_vect::Vector{String})::BitArray{1}
    freq = nfmt_index_set(bandwidths, basis_vect)
    nfmt_mask = BitArray{1}
    if length(size(freq)) == 1
        return (freq .!= 0)
    else
        return .![(0 in n) for n in eachcol(freq)]
    end
end

"""
`F = get_transform(bandwidths, X)

# Input:
 * `bandwidths::Vector{Int}`
 * `X::Array{Float64}` ... nodes in |u| x M format
 * `basis_vect::Vector{String}`

# Output:
 * `F::LinearMap{Float64}` ... Linear map of the Fourier-transform implemented by the NFCT
"""
function get_transform(bandwidths::Vector{Int}, X::Array{Float64}, basis_vect::Vector{String})::LinearMap
    if size(X, 1) == 1
        X = vec(X)
        d = 1
        M = length(X)
    else
        (d, M) = size(X)
    end

    if bandwidths == []
        return LinearMap{ComplexF64}(fhat -> fill(fhat[1], M), f -> [sum(f)], M, 1)
    end

    mask = nfmt_mask(bandwidths, basis_vect)

    b = copy(bandwidths)
    for (idx, s) in enumerate(basis_vect)
        if (BASES[s]>0)
            b[idx] *= 2
        end
    end

    N2 = Tuple(b)
    plan = NFMT(Tuple(basis_vect), N2, M, Tuple(2 * collect(N2)), 5)
    plan.x = X

    function trafo(fhat::Vector{ComplexF64})::Vector{ComplexF64}
        fh = zeros(ComplexF64, length(mask))
        fh[mask] = fhat
        plan.fhat = fh
        nfmt_trafo(plan)
        return plan.f
    end

    function adjoint(f::Vector{ComplexF64})::Vector{ComplexF64}
        plan.f = f
        nfmt_adjoint(plan)
        return plan.fhat[mask]
    end

    N = prod(bandwidths .- 1)
    return LinearMap{ComplexF64}(trafo, adjoint, M, N)
end

function get_phi(x::Vector{Float64}, k::Vector{Int64}, basis_vect::Vector{String})::ComplexF64
    p = 1
    for (idx, s) in enumerate(basis_vect)
        if (BASES[s]==1)
            if k[idx] ≠ 0
                p *= sqrt(2.0)*cos(pi*k[idx]*x[idx])
            end
        elseif (BASES[s]==2)
            if k[idx] ≠ 0
                p *= sqrt(2.0)*cos(k[idx]*acos(2*x[idx]-1))
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
 * `basis_vect::Vector{String}`

# Output:
 * `F::Array{ComplexF64}` ... Matrix of the Fourier-transform
"""
function get_matrix(bandwidths::Vector{Int}, X::Array{Float64}, basis_vect::Vector{String})::Array{ComplexF64}
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
        freq = nfmt_index_set_without_zeros(bandwidths, basis_vect)
        F_direct = [get_phi(append!(Vector{Float64}(),x), append!(Vector{Int}(),n), basis_vect) for x in vec(X), n in freq]
    else
        freq = nfmt_index_set_without_zeros(bandwidths, basis_vect)
        F_direct = [get_phi(Vector(x), Vector(n), basis_vect) for x in eachcol(X), n in eachcol(freq)]
    end

    return F_direct
end 

end