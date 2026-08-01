# Standard stuff
CI = get(ENV, "CI", nothing) == "true" || get(ENV, "GITHUB_TOKEN", nothing) !== nothing
using CairoMakie, Documenter, Literate
using DocumenterTools: Themes
using DocumenterCitations
ENV["JULIA_DEBUG"] = "Documenter"

# Packages specific to these docs
using FastHydrology

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:authoryear
)

Literate.markdown(
    joinpath(@__DIR__, "src", "examples", "Kazmierczak2024.jl"),
    joinpath(@__DIR__, "src", "examples");
    config = Dict("credit" => false)
)

Literate.markdown(
    joinpath(@__DIR__, "src", "examples", "HAB.jl"),
    joinpath(@__DIR__, "src", "examples");
    config = Dict("credit" => false)
)

Literate.markdown(
    joinpath(@__DIR__, "src", "examples", "ShaktiCoupling.jl"),
    joinpath(@__DIR__, "src", "examples");
    config = Dict("credit" => false)
)

Literate.markdown(
    joinpath(@__DIR__, "src", "examples", "ArrayGrid.jl"),
    joinpath(@__DIR__, "src", "examples");
    config = Dict("credit" => false)
)

example_pages = [
    "examples/Kazmierczak2024.md",
    "examples/HAB.md",
    "examples/ShaktiCoupling.md",
    "examples/ArrayGrid.md",
]

ref_pages = ["API_public.md", "references.md"]

# %% Build docs
PAGES = [
    "index.md",
    "Examples" => example_pages,
    "References" => ref_pages,
]

include(joinpath(@__DIR__, "style.jl"))

makedocs(
    modules = [FastHydrology],
    format = Documenter.HTML(
        prettyurls = CI,
        assets = [
            asset("https://fonts.googleapis.com/css?family=Montserrat|Source+Code+Pro&display=swap", class=:css),
        ],
        collapselevel = 2,
    ),
    sitename = "FastHydrology.jl",
    authors = "Takis Angelides",
    pages = PAGES,
    doctest = CI,
    draft = false,
    plugins = [bib],
    checkdocs = :none,
    warnonly = true,
)

deploydocs(
    repo = "github.com/TakisAngelides/FastHydrology.jl",
)