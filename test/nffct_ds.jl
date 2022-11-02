using LinearAlgebra
using GroupedTransforms

d = 4

ds = 3

dcos = [true,false,true,false]

M = 1_000
X = rand(d, M)

X[.!dcos,:] = X[.!dcos,:] .- 0.5

F = GroupedTransform("expcos", d, ds, [2^10, 2^6, 2^4, 4], X)
F_direct = get_matrix(F)