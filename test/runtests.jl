using GroupedTransforms
using Test

tests = ["nfct_ds", "nfct_U", "nfft_ds", "nfft_U"]

for t in tests
    include("$(t).jl")
end
