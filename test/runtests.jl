using GroupedTransforms
using Test

tests = ["test_grouped_nfft", "test_grouped_nfct"]

for t in tests
    include("$(t).jl")
end
