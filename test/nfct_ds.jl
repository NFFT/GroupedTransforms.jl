using LinearAlgebra

d = 4
ds = 3

M = 1_000
X = 0.5 .* rand(d, M)

# set up transform ###################################################

F = GroupedTransform("cos", d, ds, [2^12, 2^6, 2^4], X)
get_NumFreq(F.setting)
get_IndexSet(F.setting, d)
F_direct = get_matrix(F)

# compute transform with NFFT ########################################

fhat = GroupedCoefficients(F.setting)
for i = 1:length(F.setting)
    u = F.setting[i][:u]
    fhat[u] = rand(Float64, size(fhat[u]))
end

# arithmetic tests ###################################################

ghat = GroupedCoefficients(F.setting)
for i = 1:length(F.setting)
    u = F.setting[i][:u]
    ghat[u] = rand(Float64, size(ghat[u]))
end

fhat[1]
fhat[1] = 1.0
2 * fhat
fhat + ghat
fhat - ghat
F[[1, 2]]
GroupedTransforms.set_data!(fhat, ghat.data)
norms(fhat)
norms(fhat, ghat)
###

f = F * fhat

# compute transform without NFFT #####################################

f_direct = F_direct * vec(fhat)

# compare results ####################################################

error = norm(f - f_direct)
@test error < 1e-8

# generate random function values ####################################

y = rand(Float64, M)

# compute adjoint transform with NFFT ################################

fhat = F' * y

# compute adjoint transform without NFFT #############################

fhat_direct = F_direct' * y

# compare results ####################################################

error = norm(vec(fhat) - fhat_direct)
@test error < 1e-8
