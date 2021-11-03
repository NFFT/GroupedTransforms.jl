# GroupedTransforms.jl

Fast Grouped Transformations as introduced in [Bartel, Potts, Schmischke, 2021](https://arxiv.org/abs/2010.10199) .

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://nfft.github.io/GroupedTransforms.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://nfft.github.io/GroupedTransforms.jl/dev)
[![ci](https://github.com/NFFT/GroupedTransforms.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/NFFT/GroupedTransforms.jl/actions?query=workflow%3ACI+branch%3Amain)
[![codecov](https://codecov.io/gh/NFFT/GroupedTransforms.jl/branch/main/graph/badge.svg?token=FFYB0NSKHT)](https://codecov.io/gh/NFFT/GroupedTransforms.jl)
[![Aqua QA](https://img.shields.io/badge/Aqua.jl-%F0%9F%8C%A2-aqua.svg)](https://github.com/JuliaTesting/Aqua.jl)

`GroupedTransforms.jl` provides the following fast algorithms:
- nonequispaced fast transformation with exponential functions for grouped index sets based on the NFFT (non-equispaced fast Fourier transform)
- nonequispaced fast transformation with cosine functions for grouped index sets based on the NFCT (non-equispaced fast cosine transform)

## Getting started

In Julia you can get started by typing

```julia
] add GroupedTransforms
```

then checkout the [documentation](https://nfft.github.io/GroupedTransforms.jl/stable/).
