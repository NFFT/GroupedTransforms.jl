using LinearAlgebra
using GroupedTransforms

d = 8
ds = 3

M = 1_0000
X = rand(d, M) .- 0.5

# set up transform ###################################################

F = GroupedTransform("exp", d, ds, [2^13, 2^7, 2^5], X)

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

@time f = F * fhat

# generate random function values ####################################

y = rand(ComplexF64, M)

# compute adjoint transform with NFFT ################################

fhat = F' * y

