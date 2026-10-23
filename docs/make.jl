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

# Literate pages: the tutorial is static (its input file is too big to ship), the Shakti example
# runs live. The Antarctica, Greenland, Thwaites, Helheim and Drang Drung pages are plain
# markdown with figures made by make_figures.jl.
Literate.markdown(
    joinpath(@__DIR__, "src", "tutorial.jl"),
    joinpath(@__DIR__, "src");
    config = Dict("credit" => false, "documenter" => false)
)

for file in ("ShaktiCoupling.jl",)
    Literate.markdown(
        joinpath(@__DIR__, "src", "examples", file),
        joinpath(@__DIR__, "src", "examples");
        config = Dict("credit" => false)
    )
end

example_pages = [
    "examples/Antarctica.md",
    "examples/Greenland.md",
    "examples/Thwaites.md",
    "examples/Helheim.md",
    "examples/DrangDrung.md",
    "examples/ShaktiCoupling.md",
]

ref_pages = ["API_public.md", "references.md"]

# %% Build docs
PAGES = [
    "index.md",
    "tutorial.md",
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
        size_threshold = 1_000_000,
    ),
    sitename = "FastHydrology.jl",
    authors = "Takis Angelides",
    pages = PAGES,
    doctest = CI,
    draft = false,
    plugins = [bib],
    checkdocs = :exports,
    warnonly = true,
)

deploydocs(
    repo = "github.com/TakisAngelides/FastHydrology.jl",
)