using Pkg
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()

using Documenter
using BreakpointProfiles

makedocs(
    sitename = "BreakpointProfiles.jl",
    format = Documenter.HTML(),
    modules = [BreakpointProfiles],
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "Examples" => "examples.md"
    ],
    remotes = nothing
)
