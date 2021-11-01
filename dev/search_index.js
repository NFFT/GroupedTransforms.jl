var documenterSearchIndex = {"docs":
[{"location":"about.html#About","page":"About","title":"About","text":"","category":"section"},{"location":"about.html","page":"About","title":"About","text":"NFFT3.jl inherited its name from the NFFT3, a C subroutine library . It is currently maintained by Michael Schmischke (michael.schmischke@math.tu-chemnitz.de) with contributions from Tom-Christian Riemer, Toni Volkmer, and Felix Bartel.","category":"page"},{"location":"about.html","page":"About","title":"About","text":"If you want to contribute or have any questions, visit the GitHub repository to clone/fork the repository or open an issue.","category":"page"},{"location":"index.html#Welcome-to-GroupedTransforms.jl","page":"Home","title":"Welcome to GroupedTransforms.jl","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"The nonequispaced fast Fourier transform or NFFT, see [Keiner, Kunis, Potts, 2006] and [Plonka, Potts, Steidl, Tasche, 2018], overcomes one of the main shortcomings of the FFT - the need for an equispaced sampling grid. Considering a d-dimensional trigonometric polynomial ","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"  \tf(pmbx) coloneqq sum_ pmbk in I_pmbN^d hatf_pmbk  mathrme^-2pimathrmipmbkcdotpmbx","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"with an index set I_pmbN^d coloneqq  pmbk in mathbbZ^d -fracN_i2 leq pmbk_i leq fracN_i2-1 i=12ldotsd  where pmbN in (2mathbbN)^d is the multibandlimit, the nonequispaced fast Fourier transform (NDFT) is its evaluation at M in mathbbN nonequispaced points pmbx_j in mathbbT^d for j = 12 ldots M,","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"  \tf(pmbx_j) =sum_pmbk in I_pmbN^d hatf_pmbk  mathrme^-2 pi mathrmi  pmbk cdot pmbx_j","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"with given coefficients hatf_pmbk in mathbbC where we identify the smooth manifold of the torus mathbbT with -12 12). The NFFT is an algorithm for the fast evaluation of the sums f(pmbx_j) as well as the adjoint problem, the fast evaluation of","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"\thath_pmbk = sum_j = 1^M f_j  mathrme^2 pi mathrmi  pmbk cdot pmbx_j pmbk in I_pmbN^d","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"for given coefficients f_j in mathbbC. The available NFFT3 library [Keiner, Kunis, Potts, NFFT3] provides C routines for the NFFT as well as applications such as the fast evaluation of sums","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"  \tg(pmby_j) coloneqq sum_k=1^N alpha_k  K(lVert pmby_j - pmbx_k rVert_2) j = 1 ldots M","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"for given coefficients alpha_k in mathbbC, nodes pmbx_kpmby_j in R^d  and a radial kernel function K 0infty) to 0infty), and generalizations such as the NNFFT for nonequispaced nodes in time and frequency domain. ","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"The NFFT3 C library has been developed at the Mathematical Institute of the University of Luebeck, at the Mathematical Institute of the University Osnabrueck and at the Faculty of Mathematics of the Chemnitz University of Technology by Jens Keiner, Stefan Kunis and Daniel Potts. Further contributions, in particular applications, are due to Dr. Markus Fenn, Steffen Klatt, Tobias Knopp and Antje Vollrath. The support for OpenMP was developed by Toni Volkmer. Many contributions to the release 3.3.* and later have been done by Toni Volkmer, Michael Quellmalz, and Michael Schmischke.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"This package offers a Julia wrapper for the NFFT, NFCT, NFST, and fastsum algorithms, see [Schmischke, 2018].","category":"page"},{"location":"index.html#Literature","page":"Home","title":"Literature","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"<ul>\n<li id=\"PlonkaPottsSteidlTasche2018\">[<a>Plonka, Potts, Steidl, Tasche, 2018</a>]\n  G. Plonka, D. Potts, G. Steidl and M. Tasche. Numerical Fourier Analysis: Theory and Applications.</emph>\n  Springer Nature Switzerland AG, 2018.\n  doi: <a href=\"https://doi.org/10.1007/978-3-030-04306-3\">10.1007/978-3-030-04306-3</a>.\n</li>\n</ul>","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"<ul>\n<li id=\"Schmischke2018\">[<a>Schmischke, 2018</a>]\n  M. Schmischke. Nonequispaced Fast Fourier Transform (NFFT) Interface for Julia.</emph>\n  2018.\n  arXiv: <a href=\"https://arxiv.org/abs/1810.09891\">1512.02814</a>.\n</li>\n</ul>","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"<ul>\n<li id=\"KeinerKunisPotts2006\">[<a>Keiner, Kunis, Potts, 2006</a>]\n  J. Keiner, S. Kunis, and D. Potts. Fast summation of radial functions on the sphere. </emph>\n  Computing, 78:1--15, 2006.\n  doi: <a href=\"https://doi.org/10.1007/s00607-006-0169-z\">1512.02814</a>.\n</li>\n</ul>","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"<ul>\n<li id=\"KeinerKunisPottsNFFT3\">[<a>Keiner, Kunis, Potts, NFFT3</a>]\n  J. Keiner, S. Kunis, and D. Potts. NFFT 3.0, C subroutine library. </emph>\n  url: <a href=\"http://www.tu-chemnitz.de/~potts/nfft\">1512.02814</a>.\n</li>\n</ul>","category":"page"}]
}
