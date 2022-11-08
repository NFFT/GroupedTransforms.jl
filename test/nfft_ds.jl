using LinearAlgebra

d = 4
ds = 3

M = 1_000
X = rand(d, M) .- 0.5

# set up transform ###################################################

F = GroupedTransform("exp", d, ds, [2^12, 2^6, 2^4], X)
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

f = F * fhat

# compute transform without NFFT #####################################

f_direct = F_direct * vec(fhat)

# compare results ####################################################

error = norm(f - f_direct)
@test error < 1e-5

# generate random function values ####################################

y = rand(ComplexF64, M)

# compute adjoint transform with NFFT ################################

fhat = F' * y

# compute adjoint transform without NFFT #############################

fhat_direct = F_direct' * y

# compare results ####################################################

error = norm(vec(fhat) - fhat_direct)
@test error < 1e-5
