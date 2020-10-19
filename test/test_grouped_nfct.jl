d = 4
ds = 3
U = GroupedTransforms.get_superposition_set(d, ds)

tmp = [0, 2^12, 2^6, 2^4]
bandwidths = [ fill(tmp[length(u)+1], length(u)) for u in U ]

setting = [ (u = U[idx], mode = NFCTtools, bandwidths = bandwidths[idx]) for idx in 1:length(U) ]

M = 1_000
X = 2*(rand(d, M).-0.5)

# set up transform ####################################################################################################

F = GroupedTransform(setting, X)
F_direct = get_matrix(F)

# compute transform with NFFT ########################################

fhat = GroupedCoeff(setting)
for u in U
  fhat[u] = convert.(ComplexF64,rand(Float64, size(fhat[u])))
end

f = F*fhat

# compute transform without NFFT #####################################

f_direct = F_direct*vec(fhat)

# compare results ####################################################

error = sum(abs.(f.-f_direct).^2)
@test error < 1e-16

# generate random function values ####################################

y = convert.(ComplexF64,rand(Float64, M))

# compute adjoint transform with NFFT ################################

fhat = F'*y

# compute adjoint transform without NFFT #############################

fhat_direct = F_direct'*y

# compare results ####################################################

error = sqrt(sum(abs.(vec(fhat)-fhat_direct).^2))
@test error < 1e-8