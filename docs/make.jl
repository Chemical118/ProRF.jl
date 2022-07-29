using Documenter, ProRF

makedocs(
    sitename = "ProRF.jl",
    authors = "Ryu Hyunwoo",
    # format = Documenter.LaTeX(),
    doctest = true,
    modules = [ProRF],
    pages = [
        "Introduction" => "index.md",
        "Library" => [
            "Data preprocess" => "library/datapro.md",
            "Random Forest" => "library/ranfor.md",
            "Toolbox" => "library/other.md",
            "Index" => "library/findex.md"
            ],
    ]
)

deploydocs(
    repo = "github.com/Chemical118/ProRF.jl.git",
)