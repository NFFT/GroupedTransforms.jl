export CWWTtools

module CWWTtools

using LinearMaps
using Combinatorics
using SparseArrays
using LinearAlgebra


"""
`N = datalength(bandwidths)`

# Input:
 * `bandwidths::Vector{Int}`

# Output:
 * `N::Int` ... length of a wavelet-coefficient with the given bandwidths
"""
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
    s = 0
    for i =0:n
        s += 2^i*binomial(i+d-1,d-1)
    end
    return s
end
end



"""
trafos:Vector{SparseMatrixCSC}
This vector is local to the module on every worker.  It stores the transformations in order to access them later.
"""
trafos = Vector{SparseMatrixCSC{Float64, Int}}(undef,1)

"""
`F = get_transform(bandwidths, X, order)

# Input:
 * `bandwidths::Vector{Int}`
 * `X::Array{Float64}` ... nodes in |u| x M format

# Output:
 * `F::LinearMap{ComplexF64}` ... Linear maps of the sparse Matrices
"""
function get_transform(bandwidths::Vector{Int}, X::Array{Float64}, m::Int)::Int64
     #m = 2

  if size(X, 1) == 1
    X = vec(X)
    d = 1
    M = length(X)
  else
    (d, M) = size(X)
  end

  freq = cwwt_index_set(bandwidths)  #all indices j
  """
  Create sparse matrix:
  I: row indices
  J: column indices
  V: Values
  """

  if bandwidths == []       #0-dimensional terms
    idx = length(trafos)
    I = collect(1:M)
    J = ones(M)
    V = ones(M)
    trafos[idx] = sparse(I,J,V)
    append!( trafos, Vector{SparseMatrixCSC{Float64, Int64}}(undef,1) )
    return idx
elseif d == 1    #1-dimensional terms
  I = collect(1:M)
  J = Int.(ones(M))
  V = vec(Chui_periodic(X, m, 0, [0] ))
  for j in 1:bandwidths[1]
      num = min(2^j,2*m-1)
      k = Int.(mod.((floor.(2^j*X).-2*m.+2)*ones(1,num)+ ones(M,1) * transpose(collect((0:num-1))),2^j))
      I = vcat(I,kron(collect(1:M),ones(num)))
      J = vcat(J,vec(reshape(transpose(k.+2^j),:,1)))
      V = vcat(V,vec(reshape(transpose(Chui_periodic(X,m,j,k)),:,1)))
  end

  idx = length(trafos)
  trafos[idx] = sparse(I,J,V,M,2^(bandwidths[1]+1)-1)
  append!( trafos, Vector{SparseMatrixCSC{Float64, Int64}}(undef,1) )
  return idx

elseif d == 2
    # j = [0,0]:
     I = collect(1:M)
     J = Int.(ones(M))
     V = vec(Chui_periodic(X, m, [0;0], cat([0],[0],dims=[3]) ))
     freq = freq[:,2:end]
     ac_co = 2                       # actual column index
     for j in eachcol(freq)
         num1 = min(2^j[1],2*m-1)
         num2 = min(2^j[2],2*m-1)
         k1 = Int.(mod.((floor.(2^j[1]*X[1,:]).-2*m.+2)*ones(1,num1)+ ones(M,1) * transpose(collect((0:num1-1))),2^j[1]))
         k2 = Int.(mod.((floor.(2^j[2]*X[2,:]).-2*m.+2)*ones(1,num2)+ ones(M,1) * transpose(collect((0:num2-1))),2^j[2]))
         k_ind= zeros(M,num1*num2)
         k = zeros(Int,M,num1*num2 ,d)
         for kk1 in 1:num1
            k_ind[:,(kk1-1)*num2+1:kk1*num2 ]= 2^j[2]*k1[:,kk1] .+k2
            k[:,(kk1-1)*num2+1:kk1*num2 ,:] = cat(k1[:,kk1]*Int.(ones(num2)'),k2,dims =3)
        end
        I = vcat(I,kron(collect(1:M),ones(num1*num2)))
        J = vcat(J,vec(reshape(transpose(k_ind .+ac_co),:,1)))
        V = vcat(V,vec(reshape(transpose(Chui_periodic(X,m,[j[1],j[2]],k)),:,1)  ))
        ac_co = ac_co + prod(2 .^j)
     end
     idx = length(trafos)
     trafos[idx] = sparse(I,J,V,M,ac_co-1)
     append!( trafos, Vector{SparseMatrixCSC{Float64, Int64}}(undef,1) )
     return idx

 elseif d == 3 #3-dimensional terms
     I = collect(1:M)
     J = Int.(ones(M))
     V = vec(Chui_periodic(X, m, [0;0;0], cat([0],[0],[0],dims=[3]) ))
     freq = freq[:,2:end]
     ac_co = 2                       # actual column index
     for j in eachcol(freq)
         num1 = min(2^j[1],2*m-1)
         num2 = min(2^j[2],2*m-1)
         num3 = min(2^j[3],2*m-1)
         k1 = Int.(mod.((floor.(2^j[1]*X[1,:]).-2*m.+2)*ones(1,num1)+ ones(M,1) * transpose(collect((0:num1-1))),2^j[1]))
         k2 = Int.(mod.((floor.(2^j[2]*X[2,:]).-2*m.+2)*ones(1,num2)+ ones(M,1) * transpose(collect((0:num2-1))),2^j[2]))
         k3 = Int.(mod.((floor.(2^j[3]*X[3,:]).-2*m.+2)*ones(1,num3)+ ones(M,1) * transpose(collect((0:num3-1))),2^j[3]))
         k_ind = zeros(M,num1*num2*num3)
         k = zeros(Int,M,num1*num2*num3 ,d)
         for kk1 in 1:num1
             for kk2 in 1:num2
                 k_ind[:,  collect((kk1-1)*num2*num3+(kk2-1)*num3 +1:((kk1-1)*num2*num3+(kk2)*num3)) ]= 2^(j[2]+j[3])*k1[:,kk1] + 2^(j[3])*k2[:,kk2] .+k3
                 k[:,collect((kk1-1)*num2*num3+(kk2-1)*num3+1:(kk1-1)*num2*num3+(kk2)*num3) ,:] = cat(k1[:,kk1]*Int.(ones(num3)'),k2[:,kk2]*Int.(ones(num3)'),k3,dims =3)
            end
        end
        I = vcat(I,kron(collect(1:M),ones(num1*num2*num3)))
        J = vcat(J,vec(reshape(transpose(k_ind .+ac_co),:,1)))
        V = vcat(V,vec(reshape(transpose(Chui_periodic(X,m,[j[1],j[2],j[3]],k)),:,1)  ))
        ac_co = ac_co + 2^sum(j)
     end
     idx = length(trafos)
     trafos[idx] = sparse(I,J,V,M,ac_co-1)
     append!( trafos, Vector{SparseMatrixCSC{Float64, Int64}}(undef,1) )
     return idx

 elseif d == 4 #4-dimensional terms
     I = collect(1:M)
     J = Int.(ones(M))
     V = vec(Chui_periodic(X, m, [0;0;0;0], cat([0],[0],[0],[0],dims=[3]) ))
     freq = freq[:,2:end]
     ac_co = 2                       # actual column index
     for j in eachcol(freq)
         num1 = min(2^j[1],2*m-1)
         num2 = min(2^j[2],2*m-1)
         num3 = min(2^j[3],2*m-1)
         num4 = min(2^j[4],2*m-1)
         k1 = Int.(mod.((floor.(2^j[1]*X[1,:]).-2*m.+2)*ones(1,num1)+ ones(M,1) * transpose(collect((0:num1-1))),2^j[1]))
         k2 = Int.(mod.((floor.(2^j[2]*X[2,:]).-2*m.+2)*ones(1,num2)+ ones(M,1) * transpose(collect((0:num2-1))),2^j[2]))
         k3 = Int.(mod.((floor.(2^j[3]*X[3,:]).-2*m.+2)*ones(1,num3)+ ones(M,1) * transpose(collect((0:num3-1))),2^j[3]))
         k4 = Int.(mod.((floor.(2^j[4]*X[4,:]).-2*m.+2)*ones(1,num4)+ ones(M,1) * transpose(collect((0:num4-1))),2^j[4]))
         k_ind = zeros(M,num1*num2*num3*num4)
         k = zeros(Int,M,num1*num2*num3*num4 ,d)
         for kk1 in 1:num1
             for kk2 in 1:num2
                 for kk3 in 1:num3
                 k_ind[:,  collect((kk1-1)*num2*num3*num4+(kk2-1)*num3*num4+ (kk3-1)*num4+1:((kk1-1)*num2*num3*num4+(kk2-1)*num3*num4+(kk3)*num4)) ]= 2^(j[2]+j[3]+j[4])*k1[:,kk1] + 2^(j[3]+j[4])*k2[:,kk2] +2^(j[4])*k3[:,kk3] .+ k4
                 k[:,collect((kk1-1)*num2*num3*num4+(kk2-1)*num3*num4+ (kk3-1)*num4+1:((kk1-1)*num2*num3*num4+(kk2-1)*num3*num4+(kk3)*num4))  ,:] = cat(k1[:,kk1]*Int.(ones(num4)'),k2[:,kk2]*Int.(ones(num4)'),k3[:,kk3]*Int.(ones(num4)'),k4,dims =3)
                end
            end
        end
        I = vcat(I,kron(collect(1:M),ones(num1*num2*num3*num4)))
        J = vcat(J,vec(reshape(transpose(k_ind .+ac_co),:,1)))
        V = vcat(V,vec(reshape(transpose(Chui_periodic(X,m,[j[1],j[2],j[3],j[4]],k)),:,1)  ))
        ac_co = ac_co + 2^sum(j)
     end
     idx = length(trafos)
     trafos[idx] = sparse(I,J,V,M,ac_co-1)
     append!( trafos, Vector{SparseMatrixCSC{Float64, Int64}}(undef,1) )
     return idx

 elseif d == 5 #5-dimensional terms
     I = collect(1:M)
     J = Int.(ones(M))
     V = vec(Chui_periodic(X, m, [0;0;0;0;0], cat([0],[0],[0],[0],[0],dims=[3]) ))
     freq = freq[:,2:end]
     ac_co = 2                       # actual column index
     for j in eachcol(freq)
         num1 = min(2^j[1],2*m-1)
         num2 = min(2^j[2],2*m-1)
         num3 = min(2^j[3],2*m-1)
         num4 = min(2^j[4],2*m-1)
         num5 = min(2^j[5],2*m-1)
         k1 = Int.(mod.((floor.(2^j[1]*X[1,:]).-2*m.+2)*ones(1,num1)+ ones(M,1) * transpose(collect((0:num1-1))),2^j[1]))
         k2 = Int.(mod.((floor.(2^j[2]*X[2,:]).-2*m.+2)*ones(1,num2)+ ones(M,1) * transpose(collect((0:num2-1))),2^j[2]))
         k3 = Int.(mod.((floor.(2^j[3]*X[3,:]).-2*m.+2)*ones(1,num3)+ ones(M,1) * transpose(collect((0:num3-1))),2^j[3]))
         k4 = Int.(mod.((floor.(2^j[4]*X[4,:]).-2*m.+2)*ones(1,num4)+ ones(M,1) * transpose(collect((0:num4-1))),2^j[4]))
         k5 = Int.(mod.((floor.(2^j[5]*X[5,:]).-2*m.+2)*ones(1,num5)+ ones(M,1) * transpose(collect((0:num5-1))),2^j[5]))
         k_ind = zeros(M,num1*num2*num3*num4*num5)
         k = zeros(Int,M,num1*num2*num3*num4*num5 ,d)
         for kk1 in 1:num1
             for kk2 in 1:num2
                 for kk3 in 1:num3
                     for kk4 in 1:num4
                 k_ind[:,  collect((kk1-1)*num2*num3*num4*num5+(kk2-1)*num3*num4*num5+ (kk3-1)*num4*num5+(kk4-1)*num5+1:((kk1-1)*num2*num3*num4*num5+(kk2-1)*num3*num4*num5+(kk3-1)*num4*num5+kk4*num5)) ]= 2^(j[2]+j[3]+j[4]+j[5])*k1[:,kk1] + 2^(j[3]+j[4]+j[5])*k2[:,kk2] +2^(j[4]+j[5])*k3[:,kk3] + 2^(j[5])*k4[:,kk4] .+ k5
                 k[:,collect((kk1-1)*num2*num3*num4*num5+(kk2-1)*num3*num4*num5+ (kk3-1)*num4*num5+(kk4-1)*num5+1:((kk1-1)*num2*num3*num4*num5+(kk2-1)*num3*num4*num5+(kk3-1)*num4*num5+kk4*num5))  ,:] = cat(k1[:,kk1]*Int.(ones(num5)'),k2[:,kk2]*Int.(ones(num5)'),k3[:,kk3]*Int.(ones(num5)'),k4[:,kk4]*Int.(ones(num5)'),k5,dims =3)
                    end
                end
            end
        end
        I = vcat(I,kron(collect(1:M),ones(num1*num2*num3*num4*num5)))
        J = vcat(J,vec(reshape(transpose(k_ind .+ac_co),:,1)))
        V = vcat(V,vec(reshape(transpose(Chui_periodic(X,m,[j[1],j[2],j[3],j[4],j[5]],k)),:,1)  ))
        ac_co = ac_co + 2^sum(j)
     end
     idx = length(trafos)
     trafos[idx] = sparse(I,J,V,M,ac_co-1)
     append!( trafos, Vector{SparseMatrixCSC{Float64, Int64}}(undef,1) )
     return idx


 end  #if d == ...

end #function


"""
`freq = cwwt_index_set(bandwidths)`

# Input:
 * `n::Vector{Int}`

# Output:
 * `freq::Array{Int}` ... all frequencies with |j|<n
"""
function cwwt_index_set(n::Vector{Int})::Array{Int}
  d = length(n)
  d == 0 && return [0,]
  d == 1 && return collect(Int.(0:n[1]))
  freq = Int.(zeros(d,1))
  for j = 1:n[end]
      for x in partitions(d+j , d)
             x = x .- 1
                if all(x .<= j)
                    ys = Set(permutations(x))
                        for y in ys
                            freq = hcat(freq,y)
                         end
                end
        end
    end
  return freq
end



"periodic 1-d-wavelet for one j and different k:"
function  Chui_periodic(x::Vector{Float64},m:: Int,j::Int,k::Array{Int}) :: Array{Float64}
"""
# INPUT:
# x ... x-values: should be a vector
# m ... order of Wavelet
# j ... dilation parameter j>=-1
# k ... translation parameter, k in [0,2^j-1]
#       M x mm  (mm is number of different k's)
# OUTPUT :
# y ... function value
"""

if j ==-1
    y = ones(size(x))
    return y

else
    mm = size(k,2)
    y = zeros(length(x),mm)
    if 2^j >2*m-1
        for i = 1:mm
            l = ceil.(k[:,i]/2^j .- x)
            y[:,i] = 2^(j/2)*CWWTtools.Chui_wavelet(2^j*(x .+l ) .-k[:,i],m)
        end
    else

        for i = 1:mm
            for ll = 0:2*m-2
                l = ceil.(k[:,i]/2^j .- x).+ll
                y[:,i] = y[:,i] + 2^(j/2)*CWWTtools.Chui_wavelet(2^j*(x .+l ) .-k[:,i],m)
            end
        end

 end
return y #reshape(y,size(x,1),mm)
end
end

"periodic muliti-d-wavelet :"
function  Chui_periodic(x::Array{Float64},m:: Int,j::Vector{Int},k::Array{Int,3}) :: Array{Float64}
"""
# INPUT:
# x ... x-values: in d x M -format
# m ... order of Wavelet
# j ... dilation parameter j>=-1
# k ... translation parameter, k in [0,2^j-1]
#       M x mm x d  (mm is number of different k's)
# OUTPUT :
# y ... function value
"""
(d,M) = size(x)
mm = size(k,2)
y = ones(M,mm)
for i = 1:mm
    for dd = 1:d
        y[:,i] = y[:,i].*Chui_periodic(x[dd,:],m,j[dd],k[:,i,dd])
    end
end
return y
end


"Chui-Wang-Wavelet Function for different orders m :"
function  Chui_wavelet(x::Array{Float64},m::Int )::Array{Float64}
"""
% periodic Chui-Wang-Wavelets,
% Chui Wang has support [0,2m-1],
% INPUT:
% x ... x-values
% m ... order
%
%
% OUTPUT:
% psi ... function values at x
"""

" calculate coefficients q: q(i) = q_{i-1} Matlab begins with 1"
q = zeros(3*m-1,1)
n = collect(0:3*m-2)
for ell = 0:m
    q = q + binomial(m,ell).*CWWTtools.cardinal_bspline(Array{Float64}(n.+1 .-ell .-m),2*m)  #-m, since cardinal B-Spline
end
q = q .* (-1).^n *(2^(1.0-m))

psi = zeros(size(x))

for i = 1:length(x)
    xx = x[i]
    psi[i] =  psi[i] + sum(q.*CWWTtools.cardinal_bspline(2*xx .- n .-m/2,m))
end
#psi = reshape(psi,size(x))
return psi
end

"Cardinal B-Spline :"
function cardinal_bspline(v::Array{Float64},order::Int)::Array{Float64}
"""
% evaluates the centered cardinal B-spline of the given order at v
INPUT :
v     ... x-values
order ... order of B-Spline
y     ... function values
"""
f = zeros(size(v))
for i = 1:length(v)


    x = v[i];

        if x<-order/2
            j = 0;
        elseif x>=order/2
            j = 0;
        else
            j=ceil(x+order/2);
        end

        if order ==1
            if j == 1
                y = 1.0000000000000000e+00
            else
                y = 0
            end
        elseif order == 2
        if j == 1
            y = 1.0000000000000000e+00 + x*( 1.0000000000000000e+00)
        elseif j ==2
            y = 1.0000000000000000e+00 + x*( -1.0000000000000000e+00)
        else
            y = 0;
        end
        elseif order == 3
        if j == 1
            y = 1.1250000000000000e+00 + x*( 1.5000000000000000e+00 + x*( 5.0000000000000000e-01))
        elseif j == 2
            y = 7.5000000000000000e-01 + x*( 0.0000000000000000e+00 + x*( -1.0000000000000000e+00))
        elseif j == 3
            y = 1.1250000000000000e+00 + x*( -1.5000000000000000e+00 + x*( 5.0000000000000000e-01))
        else
            y = 0
        end
        elseif order ==4
        if j ==1
             y = 1.3333333333333333e+00 + x*( 2.0000000000000000e+00 + x*( 1.0000000000000000e+00 + x*( 1.6666666666666666e-01)))
        elseif j == 2
             y = 6.6666666666666663e-01 + x*( 0.0000000000000000e+00 + x*( -1.0000000000000000e+00 + x*( -5.0000000000000000e-01)))
        elseif j == 3
             y = 6.6666666666666663e-01 + x*( 0.0000000000000000e+00 + x*( -1.0000000000000000e+00 + x*( 5.0000000000000000e-01)))
        elseif j == 4
             y = 1.3333333333333333e+00 + x*( -2.0000000000000000e+00 + x*( 1.0000000000000000e+00 + x*( -1.6666666666666666e-01)))
        else
             y = 0;
        end
    elseif order == 5
    if j == 1
         y = 1.6276041666666665e+00 + x*( 2.6041666666666665e+00 + x*( 1.5625000000000000e+00 + x*( 4.1666666666666663e-01 + x*( 4.1666666666666664e-02))))
     elseif j == 2
         y = 5.7291666666666663e-01 + x*( -2.0833333333333343e-01 + x*( -1.2500000000000000e+00 + x*( -8.3333333333333337e-01 + x*( -1.6666666666666666e-01))))
     elseif j == 3
         y = 5.9895833333333326e-01 + x*( 0.0000000000000000e+00 + x*( -6.2500000000000000e-01 + x*( 0.0000000000000000e+00 + x*( 2.5000000000000000e-01))))
     elseif j == 4
         y = 5.7291666666666663e-01 + x*( 2.0833333333333343e-01 + x*( -1.2500000000000000e+00 + x*( 8.3333333333333337e-01 + x*( -1.6666666666666666e-01))))
     elseif j == 5
         y = 1.6276041666666665e+00 + x*( -2.6041666666666665e+00 + x*( 1.5625000000000000e+00 + x*( -4.1666666666666663e-01 + x*( 4.1666666666666664e-02))))
     else
         y = 0
     end
 elseif order == 6
    if j == 1
         y = 2.0249999999999999e+00 + x*( 3.3750000000000000e+00 + x*( 2.2500000000000000e+00 + x*( 7.5000000000000000e-01 + x*( 1.2500000000000000e-01 + x*( 8.3333333333333332e-03)))))
     elseif j == 2
         y = 4.2499999999999999e-01 + x*( -6.2500000000000033e-01 + x*( -1.7500000000000000e+00 + x*( -1.2500000000000000e+00 + x*( -3.7499999999999994e-01 + x*( -4.1666666666666664e-02)))))
     elseif j == 3
         y = 5.5000000000000004e-01 + x*( -1.1102230246251565e-16 + x*( -4.9999999999999983e-01 + x*( 0.0000000000000000e+00 + x*( 2.5000000000000006e-01 + x*( 8.3333333333333329e-02)))))
     elseif j == 4
         y = 5.5000000000000004e-01 + x*( 1.1102230246251565e-16 + x*( -5.0000000000000000e-01 + x*( 0.0000000000000000e+00 + x*( 2.5000000000000000e-01 + x*( -8.3333333333333329e-02)))))
     elseif j == 5
         y = 4.2499999999999999e-01 + x*( 6.2500000000000022e-01 + x*( -1.7500000000000000e+00 + x*( 1.2500000000000000e+00 + x*( -3.7500000000000000e-01 + x*( 4.1666666666666664e-02)))))
     elseif j == 6
         y = 2.0249999999999999e+00 + x*( -3.3750000000000000e+00 + x*( 2.2500000000000000e+00 + x*( -7.5000000000000000e-01 + x*( 1.2500000000000000e-01 + x*( -8.3333333333333332e-03)))))
     else
         y = 0;
    end
  elseif order == 7
    if j == 1
         y = 2.5531467013888887e+00 + x*( 4.3768229166666659e+00 + x*( 3.1263020833333335e+00 + x*( 1.1909722222222221e+00 + x*( 2.5520833333333331e-01 + x*( 2.9166666666666664e-02 + x*( 1.3888888888888889e-03))))))
     elseif j == 2
         y = 1.7955729166666637e-01 + x*( -1.3197916666666669e+00 + x*( -2.5703125000000000e+00 + x*( -1.8472222222222221e+00 + x*( -6.5625000000000000e-01 + x*( -1.1666666666666665e-01 + x*( -8.3333333333333332e-03))))))
     elseif j == 3
         y = 5.1178385416666694e-01 + x*( 9.1145833333331847e-03 + x*( -3.5546874999999956e-01 + x*( 1.2152777777777778e-01 + x*( 3.2812500000000006e-01 + x*( 1.4583333333333334e-01 + x*( 2.0833333333333332e-02))))))
     elseif j == 4
         y = 5.1102430555555556e-01 + x*( -1.2952601953960160e-16 + x*( -4.0104166666666652e-01 + x*( -7.4014868308343765e-17 + x*( 1.4583333333333334e-01 + x*( 0.0000000000000000e+00 + x*( -2.7777777777777776e-02))))))
     elseif j == 5
         y = 5.1178385416666694e-01 + x*( -9.1145833333330373e-03 + x*( -3.5546875000000000e-01 + x*( -1.2152777777777778e-01 + x*( 3.2812500000000000e-01 + x*( -1.4583333333333334e-01 + x*( 2.0833333333333332e-02))))))
     elseif j == 6
         y = 1.7955729166666645e-01 + x*( 1.3197916666666669e+00 + x*( -2.5703125000000000e+00 + x*( 1.8472222222222221e+00 + x*( -6.5625000000000000e-01 + x*( 1.1666666666666665e-01 + x*( -8.3333333333333332e-03))))))
     elseif j == 7
         y = 2.5531467013888887e+00 + x*( -4.3768229166666659e+00 + x*( 3.1263020833333335e+00 + x*( -1.1909722222222221e+00 + x*( 2.5520833333333331e-01 + x*( -2.9166666666666664e-02 + x*( 1.3888888888888889e-03))))))
     else
        y = 0;
    end
  elseif order == 8
    if j == 1
         y = 3.2507936507936499e+00 + x*( 5.6888888888888891e+00 + x*( 4.2666666666666666e+00 + x*( 1.7777777777777775e+00 + x*( 4.4444444444444448e-01 + x*( 6.6666666666666666e-02 + x*( 5.5555555555555558e-03 + x*( 1.9841269841269841e-04)))))))
     elseif j ==2
         y = -2.2063492063492082e-01 + x*( -2.4111111111111110e+00 + x*( -3.8333333333333330e+00 + x*( -2.7222222222222219e+00 + x*( -1.0555555555555556e+00 + x*( -2.3333333333333334e-01 + x*( -2.7777777777777780e-02 + x*( -1.3888888888888889e-03)))))))
     elseif j == 3
         y = 4.9047619047619057e-01 + x*( 7.7777777777777932e-02 + x*( -9.9999999999999520e-02 + x*( 3.8888888888888917e-01 + x*( 5.0000000000000000e-01 + x*( 2.3333333333333336e-01 + x*( 4.9999999999999996e-02 + x*( 4.1666666666666666e-03)))))))
     elseif j == 4
         y = 4.7936507936507977e-01 + x*( -4.1930744590753681e-16 + x*( -3.3333333333333293e-01 + x*( -1.3481279584734043e-16 + x*( 1.1111111111111112e-01 + x*( 0.0000000000000000e+00 + x*( -2.7777777777777783e-02 + x*( -6.9444444444444432e-03)))))))
     elseif j == 5
         y = 4.7936507936507972e-01 + x*( -9.9127055770103257e-19 + x*( -3.3333333333333298e-01 + x*( -8.7231809077690869e-17 + x*( 1.1111111111111109e-01 + x*( -1.5860328923216521e-17 + x*( -2.7777777777777780e-02 + x*( 6.9444444444444432e-03)))))))
     elseif j == 6
         y = 4.9047619047619057e-01 + x*( -7.7777777777777682e-02 + x*( -1.0000000000000012e-01 + x*( -3.8888888888888901e-01 + x*( 5.0000000000000000e-01 + x*( -2.3333333333333336e-01 + x*( 4.9999999999999996e-02 + x*( -4.1666666666666666e-03)))))))
     elseif j == 7
         y = -2.2063492063492077e-01 + x*( 2.4111111111111114e+00 + x*( -3.8333333333333330e+00 + x*( 2.7222222222222219e+00 + x*( -1.0555555555555556e+00 + x*( 2.3333333333333334e-01 + x*( -2.7777777777777773e-02 + x*( 1.3888888888888889e-03)))))))
     elseif j == 8
         y = 3.2507936507936499e+00 + x*( -5.6888888888888891e+00 + x*( 4.2666666666666666e+00 + x*( -1.7777777777777775e+00 + x*( 4.4444444444444448e-01 + x*( -6.6666666666666666e-02 + x*( 5.5555555555555558e-03 + x*( -1.9841269841269841e-04)))))))
     else
         y = 0
    end
 elseif order == 9
    if j ==1
         y = 4.1704180036272316e+00 + x*( 7.4140764508928569e+00 + x*( 5.7665039062499996e+00 + x*( 2.5628906249999996e+00 + x*( 7.1191406250000011e-01 + x*( 1.2656250000000002e-01 + x*( 1.4062500000000000e-02 + x*( 8.9285714285714294e-04 + x*( 2.4801587301587302e-05))))))))
     elseif j == 2
         y = -8.5608956473214359e-01 + x*( -4.0750837053571427e+00 + x*( -5.7226562500000000e+00 + x*( -4.0023437500000014e+00 + x*( -1.6328125000000000e+00 + x*( -4.0937499999999993e-01 + x*( -6.2499999999999993e-02 + x*( -5.3571428571428572e-03 + x*( -1.9841269841269841e-04))))))))
     elseif j == 3
         y = 5.0630231584821439e-01 + x*( 2.8457031250000059e-01 + x*( 3.8085937500000017e-01 + x*( 8.8046875000000069e-01 + x*( 8.0859374999999989e-01 + x*( 3.7187500000000001e-01 + x*( 9.3749999999999986e-02 + x*( 1.2499999999999999e-02 + x*( 6.9444444444444447e-04))))))))
     elseif j == 4
         y = 4.5290876116071449e-01 + x*( -1.9531250000014036e-04 + x*( -2.8359374999999992e-01 + x*( -5.4687499999998678e-03 + x*( 7.0312499999999861e-02 + x*( -2.1874999999999967e-02 + x*( -3.7500000000000019e-02 + x*( -1.2500000000000001e-02 + x*( -1.3888888888888887e-03))))))))
     elseif j == 5
         y = 4.5292096819196492e-01 + x*( -4.1199682554449168e-16 + x*( -2.8222656249999944e-01 + x*( -2.1510571102112408e-16 + x*( 8.3984374999999944e-02 + x*( -1.3877787807814457e-17 + x*( -1.5625000000000014e-02 + x*( 8.6736173798840355e-19 + x*( 1.7361111111111108e-03))))))))
     elseif j == 6
         y = 4.5290876116071466e-01 + x*( 1.9531249999965074e-04 + x*( -2.8359374999999992e-01 + x*( 5.4687499999997776e-03 + x*( 7.0312499999999875e-02 + x*( 2.1874999999999933e-02 + x*( -3.7500000000000006e-02 + x*( 1.2499999999999999e-02 + x*( -1.3888888888888887e-03))))))))
     elseif j == 7
         y = 5.0630231584821439e-01 + x*( -2.8457031249999992e-01 + x*( 3.8085937500000017e-01 + x*( -8.8046875000000024e-01 + x*( 8.0859374999999989e-01 + x*( -3.7187500000000001e-01 + x*( 9.3750000000000000e-02 + x*( -1.2499999999999997e-02 + x*( 6.9444444444444447e-04))))))))
     elseif j == 8
         y = -8.5608956473214382e-01 + x*( 4.0750837053571427e+00 + x*( -5.7226562500000000e+00 + x*( 4.0023437500000005e+00 + x*( -1.6328125000000000e+00 + x*( 4.0937499999999993e-01 + x*( -6.2499999999999986e-02 + x*( 5.3571428571428563e-03 + x*( -1.9841269841269841e-04))))))))
     elseif j == 9
         y = 4.1704180036272316e+00 + x*( -7.4140764508928569e+00 + x*( 5.7665039062499996e+00 + x*( -2.5628906249999996e+00 + x*( 7.1191406250000011e-01 + x*( -1.2656250000000002e-01 + x*( 1.4062500000000000e-02 + x*( -8.9285714285714294e-04 + x*( 2.4801587301587302e-05))))))))
     else
         y = 0
    end
 elseif order == 10
    if j == 1
         y = 5.3822889109347445e+00 + x*( 9.6881200396825395e+00 + x*( 7.7504960317460307e+00 + x*( 3.6168981481481475e+00 + x*( 1.0850694444444446e+00 + x*( 2.1701388888888895e-01 + x*( 2.8935185185185189e-02 + x*( 2.4801587301587300e-03 + x*( 1.2400793650793653e-04 + x*( 2.7557319223985893e-06)))))))))
     elseif j == 2
         y = -1.8416969797178138e+00 + x*( -6.5658482142857135e+00 + x*( -8.5034722222222232e+00 + x*( -5.8645833333333339e+00 + x*( -2.4704861111111116e+00 + x*( -6.7187500000000000e-01 + x*( -1.1921296296296295e-01 + x*( -1.3392857142857140e-02 + x*( -8.6805555555555572e-04 + x*( -2.4801587301587302e-05)))))))))
     elseif j == 3
         y = 5.9915123456790131e-01 + x*( 7.5669642857142883e-01 + x*( 1.2599206349206369e+00 + x*( 1.7291666666666661e+00 + x*( 1.3263888888888888e+00 + x*( 5.9375000000000000e-01 + x*( 1.6203703703703703e-01 + x*( 2.6785714285714281e-02 + x*( 2.4801587301587300e-03 + x*( 9.9206349206349220e-05)))))))))
     elseif j == 4
         y = 4.2983906525573179e-01 + x*( -5.2083333333328057e-03 + x*( -2.6388888888888967e-01 + x*( -4.8611111111110529e-02 + x*( -6.9444444444446956e-03 + x*( -7.2916666666666644e-02 + x*( -6.0185185185185203e-02 + x*( -2.0833333333333336e-02 + x*( -3.4722222222222220e-03 + x*( -2.3148148148148149e-04)))))))))
     elseif j == 5
         y = 4.3041776895943618e-01 + x*( -2.4454782334950824e-18 + x*( -2.4305555555555569e-01 + x*( 1.3010426069826053e-16 + x*( 6.5972222222221863e-02 + x*( 2.8526563827174158e-17 + x*( -1.1574074074074106e-02 + x*( -1.5419764230904951e-18 + x*( 1.7361111111111110e-03 + x*( 3.4722222222222213e-04)))))))))
     elseif j == 6
         y = 4.3041776895943623e-01 + x*( -9.9439409253128842e-16 + x*( -2.4305555555555530e-01 + x*( -5.7168775886080107e-16 + x*( 6.5972222222221932e-02 + x*( -5.7053127654348317e-17 + x*( -1.1574074074074105e-02 + x*( 3.0839528461809902e-18 + x*( 1.7361111111111106e-03 + x*( -3.4722222222222213e-04)))))))))
     elseif j == 7
         y = 4.2983906525573234e-01 + x*( 5.2083333333329627e-03 + x*( -2.6388888888888878e-01 + x*( 4.8611111111110640e-02 + x*( -6.9444444444445291e-03 + x*( 7.2916666666666644e-02 + x*( -6.0185185185185182e-02 + x*( 2.0833333333333336e-02 + x*( -3.4722222222222216e-03 + x*( 2.3148148148148149e-04)))))))))
     elseif j == 8
         y = 5.9915123456790087e-01 + x*( -7.5669642857142860e-01 + x*( 1.2599206349206351e+00 + x*( -1.7291666666666665e+00 + x*( 1.3263888888888888e+00 + x*( -5.9375000000000000e-01 + x*( 1.6203703703703703e-01 + x*( -2.6785714285714288e-02 + x*( 2.4801587301587300e-03 + x*( -9.9206349206349220e-05)))))))))
     elseif j == 9
         y = -1.8416969797178138e+00 + x*( 6.5658482142857135e+00 + x*( -8.5034722222222232e+00 + x*( 5.8645833333333330e+00 + x*( -2.4704861111111112e+00 + x*( 6.7187499999999989e-01 + x*( -1.1921296296296295e-01 + x*( 1.3392857142857140e-02 + x*( -8.6805555555555551e-04 + x*( 2.4801587301587302e-05)))))))))
     elseif j == 10
         y = 5.3822889109347445e+00 + x*( -9.6881200396825395e+00 + x*( 7.7504960317460307e+00 + x*( -3.6168981481481475e+00 + x*( 1.0850694444444446e+00 + x*( -2.1701388888888895e-01 + x*( 2.8935185185185189e-02 + x*( -2.4801587301587300e-03 + x*( 1.2400793650793653e-04 + x*( -2.7557319223985893e-06)))))))))
     else
         y = 0
    end

        end
        f[i] = y
    end
    return f
end


"""
# function from index set of wavelets to natural numbers, i.e.
# (j,k) maps to N
# input:
#       j in  0 ... n
#       k in  0 ... 2^j-1
# output:
#       out  in N
# creates row vector with entry for every column vector in k
"""

function indextoN(j::Array{Int},k::Array{Int})::Array{Int}

d = length(j);  #dimension

d2 = size(k,1)
s = size(k,2)
if d!=d2
    error("j and k have to have same length, k has to be column vector.")
end
out = zeros(1,s)


# begin_index(\ell) = 1+\sum_i^\ell 2^(i-1)*i
# begin of indices with |j|_1=\ell
#                 0   1   2   3   4     5   6   7   8     9    10      11    12    13      14      15      16   17         18      19
#begin_index2d =[ 1   2    6   18  50  130 322 770 1794 4098   9218  20482 45058  98306  212994  458754 983042  2097154  4456450  9437186];

    if d == 1
        for i =1:s
         out[i] = 2 .^(j)+k[:,i]
        end

    elseif d ==2
      level = sum(j)
          for i = 1:s
        out[i] = begin_index2d(level)+j(1)*2^level+2^j[2]*k[1,i]+k[2,i]
          end
      end
    return out
end



"""
2-dimensional index sets,
begin of indices with |j|_1=\ell
"""
function  begin_index2d(j::Int):: Int
    if j == 0
        ind =1;
    elseif j ==1
        ind = 2;
    else
        ind = 2^(j)*(j-1)+2;
    end
    end


end #module
