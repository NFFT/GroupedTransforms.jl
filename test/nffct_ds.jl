using LinearAlgebra
using GroupedTransforms
using Test

d = 4

ds = 3

dcos = [true,false,true,false]

M = 1_000
X = rand(d, M)

X[.!dcos,:] = X[.!dcos,:] .- 0.5

# set up transform ###################################################

F = GroupedTransform("expcos", d, ds, [2^6, 2^4, 4], X, dcos)
for s = F.setting
    println("######")
    println(s[:bandwidths])
    println(s[:mode].datalength(s[:bandwidths]))
end
F_direct = get_matrix(F)

# compute transform with NFFT ########################################

fhat = GroupedCoefficients(F.setting)
for i = 1:length(F.setting)
    u = F.setting[i][:u]
    fhat[u] = rand(ComplexF64, size(fhat[u]))
end

# arithmetic tests ###################################################

ghat = GroupedCoefficients(F.setting)
for i = 1:length(F.setting)
    u = F.setting[i][:u]
    ghat[u] = rand(ComplexF64, size(ghat[u]))
end

fhat[1]
fhat[1] = 1.0 + 1.0 * im
2 * fhat
fhat + ghat
fhat - ghat
#F[[1, 2]]
GroupedTransforms.set_data!(fhat, ghat.data)

###

#f = F * fhat

# compute transform without NFFT #####################################

f_direct = F_direct * vec(fhat)

# compare results ####################################################

error = norm(f - f_direct)
@test error < 1e-5
