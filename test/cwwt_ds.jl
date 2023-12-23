using LinearAlgebra
using GroupedTransforms
using BenchmarkTools

d = 8
ds = 3

M = 1_000
X = rand(d, M) .- 0.5

# set up transform ###################################################

F = GroupedTransform("chui3", d, ds, [3, 2, 1], X)
#F_direct = get_matrix(F)

# compute transform with CWWT ########################################

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
#F[[1, 2]]
GroupedTransforms.set_data!(fhat, ghat.data)

###

@btime f = F * fhat

y = rand(Float64, M)

@btime fhat = F' * y

A = F[[1, 2]]