using GroupedTransforms
using Test
using Aqua

Aqua.test_all(GroupedTransforms, ambiguities = false)
Aqua.test_ambiguities(GroupedTransforms)

tests = ["cwwt_ds", "cwwt_U", "nfmt_ds", "nfmt_U", "nfct_ds", "nfct_U", "nfft_ds", "nfft_U"]

for t in tests
    include("$(t).jl")
    GC.gc()
end
