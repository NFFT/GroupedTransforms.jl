var documenterSearchIndex = {"docs":
[{"location":"about.html#About","page":"About","title":"About","text":"","category":"section"},{"location":"about.html","page":"About","title":"About","text":"GroupedTransforms.jl inherited its name from the paper Bartel, Potts, Schmischke, 2021. It is currently maintained by Michael Schmischke (michael.schmischke@math.tu-chemnitz.de) with contributions from Felix Bartel.","category":"page"},{"location":"about.html","page":"About","title":"About","text":"If you want to contribute or have any questions, visit the GitHub repository to clone/fork the repository or open an issue.","category":"page"},{"location":"transform.html#GroupedTransform","page":"GroupedTransform","title":"GroupedTransform","text":"","category":"section"},{"location":"transform.html","page":"GroupedTransform","title":"GroupedTransform","text":"    CurrentModule = GroupedTransforms","category":"page"},{"location":"transform.html","page":"GroupedTransform","title":"GroupedTransform","text":"Modules = [GroupedTransforms]\nPages   = [\"GroupedTransform.jl\"]","category":"page"},{"location":"transform.html#GroupedTransforms.GroupedTransform","page":"GroupedTransform","title":"GroupedTransforms.GroupedTransform","text":"GroupedTransform\n\nA struct to describe a GroupedTransformation\n\nFields\n\nsystem::String - choice of \"exp\" or \"cos\" or \"chui1\" or \"chui2\" or \"chui3\" or \"chui4\"\nsetting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}} - vector of the dimensions, mode, and bandwidths for each term/group, see also get_setting(system::String,d::Int,ds::Int,N::Vector{Int})::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}} and get_setting(system::String,U::Vector{Vector{Int}},N::Vector{Int})::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}}\nX::Array{Float64} - array of nodes\ntransforms::Vector{Tuple{Int64,Int64}} - holds the low-dimensional sub transformations\n\nConstructor\n\nGroupedTransform( system, setting, X )\n\nAdditional Constructor\n\nGroupedTransform( system, d, ds, N::Vector{Int}, X )\nGroupedTransform( system, U, N, X )\n\n\n\n\n\n","category":"type"},{"location":"transform.html#Base.:*-Tuple{GroupedTransform, GroupedCoefficients}","page":"GroupedTransform","title":"Base.:*","text":"*( F::GroupedTransform, fhat::GroupedCoefficients )::Vector{<:Number}\n\nOverloads the * notation in order to achieve f = F*fhat.\n\n\n\n\n\n","category":"method"},{"location":"transform.html#Base.:*-Tuple{GroupedTransform, Vector{var\"#s25\"} where var\"#s25\"<:Number}","page":"GroupedTransform","title":"Base.:*","text":"*( F::GroupedTransform, f::Vector{<:Number} )::GroupedCoefficients\n\nOverloads the * notation in order to achieve the adjoint transform f = F*f.\n\n\n\n\n\n","category":"method"},{"location":"transform.html#Base.adjoint-Tuple{GroupedTransform}","page":"GroupedTransform","title":"Base.adjoint","text":"adjoint( F::GroupedTransform )::GroupedTransform\n\nOverloads the F' notation and gives back the same GroupdTransform. GroupedTransform decides by the input if it is the normal trafo or the adjoint so this is only for convinience.\n\n\n\n\n\n","category":"method"},{"location":"transform.html#Base.getindex-Tuple{GroupedTransform, Vector{Int64}}","page":"GroupedTransform","title":"Base.getindex","text":"F::GroupedTransform[u::Vector{Int}]::LinearMap{<:Number} or SparseArray\n\nThis function overloads getindex of GroupedTransform such that you can do F[[1,3]] to obtain the transform of the corresponding ANOVA term defined by u.\n\n\n\n\n\n","category":"method"},{"location":"transform.html#GroupedTransforms.get_matrix-Tuple{GroupedTransform}","page":"GroupedTransform","title":"GroupedTransforms.get_matrix","text":"get_matrix( F::GroupedTransform )::Matrix{<:Number}\n\nThis function returns the actual matrix of the transformation. This is not available for the wavelet basis\n\n\n\n\n\n","category":"method"},{"location":"index.html#Welcome-to-GroupedTransforms.jl","page":"Home","title":"Welcome to GroupedTransforms.jl","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"The nonequispaced Grouped Transformations have been introduced in Bartel, Potts, Schmischke, 2021 to provide a fast method for the multiplication of matrices with the Fourier or the cosine system supported on frequency index sets with low-dimensional support.","category":"page"},{"location":"index.html#Literature","page":"Home","title":"Literature","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"<ul>\n<li id=\"BaPoSc\">[<a>Bartel, Potts, Schmischke, 2021</a>]\n  F. Bartel, D. Potts, M. Schmischke. Grouped transformations in high-dimensional explainable ANOVA approximation.</emph>\n  Preprint 2021.\n  arXiv: <a href=\"https://arxiv.org/abs/2010.10199\">2010.10199</a>.\n</li>\n</ul>","category":"page"},{"location":"coeffs.html#GroupedCoefficients","page":"GroupedCoefficients","title":"GroupedCoefficients","text":"","category":"section"},{"location":"coeffs.html","page":"GroupedCoefficients","title":"GroupedCoefficients","text":"    CurrentModule = GroupedTransforms","category":"page"},{"location":"coeffs.html","page":"GroupedCoefficients","title":"GroupedCoefficients","text":"Modules = [GroupedTransforms]\nPages   = [\"GroupedCoefficients.jl\"]","category":"page"},{"location":"coeffs.html#GroupedTransforms.GroupedCoefficientsComplex","page":"GroupedCoefficients","title":"GroupedTransforms.GroupedCoefficientsComplex","text":"GroupedCoefficientsComplex\n\nA struct to hold complex coefficients belonging to indices in a grouped index set\n\n    mathcalI_pmbN(U) = left pmbk in Z^d  mathrmsupp pmbk in U pmbk_mathrmsupp pmbk in - fracN_mathrmsupp pmbk 2 fracN_ mathrmsupp pmbk 2 - 1 ) right\n\nFields\n\nsetting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}} - uniquely describes the setting such as the bandlimits N_pmb u, see also get_setting(system::String,d::Int,ds::Int,N::Vector{Int})::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}} and get_setting(system::String,U::Vector{Vector{Int}},N::Vector{Int})::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}}\ndata::Union{Vector{ComplexF64},Nothing} - the vector of coefficients\n\nConstructor\n\nGroupedCoefficientsComplex( setting, data = nothing )\n\nAdditional Constructor\n\nGroupedCoefficients( setting, data = nothing )\n\n\n\n\n\n","category":"type"},{"location":"coeffs.html#GroupedTransforms.GroupedCoefficientsReal","page":"GroupedCoefficients","title":"GroupedTransforms.GroupedCoefficientsReal","text":"GroupedCoefficientsReal\n\nA struct to hold real valued coefficients belonging to indices in a grouped index set\n\n    mathcalI_pmbN(U) = left pmbk in Z^d  mathrmsupp pmbk in U pmbk_mathrmsupp pmbk in 0 N_mathrmsupp pmbk  - 1  right\n\nFields\n\nsetting::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}} - uniquely describes the setting such as the bandlimits N_pmb u, see also get_setting(system::String,d::Int,ds::Int,N::Vector{Int})::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}} and get_setting(system::String,U::Vector{Vector{Int}},N::Vector{Int})::Vector{NamedTuple{(:u, :mode, :bandwidths),Tuple{Vector{Int},Module,Vector{Int}}}}\ndata::Union{Vector{Float64},Nothing} - the vector of coefficients\n\nConstructor\n\nGroupedCoefficientsReal( setting, data = nothing )\n\nAdditional Constructor\n\nGroupedCoefficients( setting, data = nothing )\n\n\n\n\n\n","category":"type"},{"location":"coeffs.html#Base.:*-Tuple{Number, GroupedCoefficients}","page":"GroupedCoefficients","title":"Base.:*","text":"*( z::Number, fhat::GroupedCoefficients )::GroupedCoefficients\n\nThis function defines the multiplication of a number with a GroupedCoefficients object.\n\n\n\n\n\n","category":"method"},{"location":"coeffs.html#Base.:+-Tuple{GroupedCoefficients, GroupedCoefficients}","page":"GroupedCoefficients","title":"Base.:+","text":"+( z::Number, fhat::GroupedCoefficients )::GroupedCoefficients\n\nThis function defines the addition of two GroupedCoefficients objects.\n\n\n\n\n\n","category":"method"},{"location":"coeffs.html#Base.:--Tuple{GroupedCoefficients, GroupedCoefficients}","page":"GroupedCoefficients","title":"Base.:-","text":"-( z::Number, fhat::GroupedCoefficients )::GroupedCoefficients\n\nThis function defines the subtraction of two GroupedCoefficients objects.\n\n\n\n\n\n","category":"method"},{"location":"coeffs.html#Base.getindex-Tuple{GroupedCoefficients, Int64}","page":"GroupedCoefficients","title":"Base.getindex","text":"fhat::GroupedCoefficients[idx::Int]\n\nThis function overloads getindex of GroupedCoefficients such that you can do fhat[1] to obtain the basis coefficient determined by idx.\n\n\n\n\n\n","category":"method"},{"location":"coeffs.html#Base.getindex-Tuple{GroupedCoefficients, Vector{Int64}}","page":"GroupedCoefficients","title":"Base.getindex","text":"fhat::GroupedCoefficients[u::Vector{Int}]\n\nThis function overloads getindex of GroupedCoefficients such that you can do fhat[[1,3]] to obtain the basis coefficients of the corresponding ANOVA term defined by u.\n\n\n\n\n\n","category":"method"},{"location":"coeffs.html#Base.setindex!-Tuple{GroupedCoefficients, Number, Int64}","page":"GroupedCoefficients","title":"Base.setindex!","text":"fhat::GroupedCoefficients[idx::Int] = z::Number\n\nThis function overloads setindex of GroupedCoefficients such that you can do fhat[1] = 3 to set the basis coefficient determined by idx.\n\n\n\n\n\n","category":"method"},{"location":"coeffs.html#Base.setindex!-Tuple{GroupedCoefficients, Union{Vector{ComplexF64}, Vector{Float64}}, Vector{var\"#s26\"} where var\"#s26\"<:Integer}","page":"GroupedCoefficients","title":"Base.setindex!","text":"fhat::GroupedCoefficients[u::Vector{Int}] = fhatu::Union{Vector{ComplexF64},Vector{Float64}}\n\nThis function overloads setindex of GroupedCoefficients such that you can do fhat[[1,3]] = [1 2 3] to set the basis coefficients of the corresponding ANOVA term defined by u.\n\n\n\n\n\n","category":"method"},{"location":"coeffs.html#Base.vec-Tuple{GroupedCoefficients}","page":"GroupedCoefficients","title":"Base.vec","text":"vec( fhat::GroupedCoefficients )::Vector{<:Number}\n\nThis function returns the vector of the basis coefficients of fhat. This is useful for working with lsqr or similar.\n\n\n\n\n\n","category":"method"},{"location":"coeffs.html#GroupedTransforms.variances-Tuple{Int64, Int64}","page":"GroupedCoefficients","title":"GroupedTransforms.variances","text":"matrix of variances between two basis functions, needed for wavelet basis, since they are not orthonormal\n\n\n\n\n\n","category":"method"}]
}
