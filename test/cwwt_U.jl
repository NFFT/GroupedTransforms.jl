using LinearAlgebra

d = 4
ds = 3

M = 1_000
X = rand(d, M) .- 0.5

U = Vector{Vector{Int64}}(undef, 3)
U[1] = []
U[2] = [1]
U[3] = [1, 2]

# set up transform ###################################################

F = GroupedTransform("chui2", U, [3, 2, 1], X)

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
F[[1, 2]]
GroupedTransforms.set_data!(fhat, ghat.data)

###
f = F * fhat



# generate random function values ###################################

y = rand(Float64, M)
fhat = F' * y
