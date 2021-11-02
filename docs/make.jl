using Documenter, GroupedTransforms

makedocs(
    sitename = "GroupedTransforms.jl",
    format = Documenter.HTML(; prettyurls = false),
    modules = [GroupedTransforms],
    pages = ["Home" => "index.md", "About" => "about.md"],
)

deploydocs(
    repo = "github.com/NFFT/GroupedTransforms.jl.git",
    devbranch = "main",
    devurl = "dev",
    versions = ["stable" => "v^", "v#.#", "dev" => "dev"],
)
