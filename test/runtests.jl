using GroupedTransforms
using Test
using Aqua

Aqua.test_all(GroupedTransforms)

tests = ["nfct_ds", "nfct_U", "nfft_ds", "nfft_U"]

for t in tests
    include("$(t).jl")
end
