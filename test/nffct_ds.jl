using LinearAlgebra
using GroupedTransforms
using Test

d = 4

ds = 3

dcos = ["exp", "alg", "cos", "alg"]

M = 1_000
X = rand(d, M)
X[1,:] = X[1,:] .- 0.5

# set up transform ###################################################

F = GroupedTransform("mixed", d, ds, [2^6, 2^4, 4], X, dcos)

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

f = F * fhat

# compare results ####################################################

error = norm(f - f_direct)

# generate random function values ####################################

y = rand(ComplexF64, M)

# compute adjoint transform with NFFT ################################

fhat = F' * y

