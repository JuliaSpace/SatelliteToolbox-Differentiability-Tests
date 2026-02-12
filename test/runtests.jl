using SatelliteToolboxDifferentiability
using Test

using DifferentiationInterface, DifferentiationInterfaceTest
using Enzyme, FiniteDiff, ForwardDiff, GTPSA, Mooncake, PolyesterForwardDiff, TaylorDiff, Zygote

const _BACKENDS = [
    AutoEnzyme(),
    AutoForwardDiff(),
    AutoGTPSA(),
    AutoMooncake(; config=nothing),
    AutoPolyesterForwardDiff(),
    AutoZygote(),
]

@testset "SatelliteToolboxDifferentiability.jl" begin
    include("space_indices.jl")
end
