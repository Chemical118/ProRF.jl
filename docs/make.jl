using Documenter, ProRF

makedocs(
    sitename = "ProRF.jl",
    authors = "Ryu Hyunwoo",
    doctest = true,
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