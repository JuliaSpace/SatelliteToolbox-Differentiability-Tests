using SatelliteToolboxDifferentiability
using Test

using DifferentiationInterface, DifferentiationInterfaceTest
using Enzyme, FiniteDiff, ForwardDiff, GTPSA, ImplicitDifferentiation, Mooncake, PolyesterForwardDiff, TaylorDiff, Zygote

const _BACKENDS = [
    AutoEnzyme(),
    AutoForwardDiff(),
    AutoGTPSA(),
    AutoMooncake(; config=nothing),
    AutoPolyesterForwardDiff(),
    AutoZygote(),
]

# Enzyme with runtime activity for models that require it.
const _BACKENDS_RTA = map(
    b -> b isa AutoEnzyme ? AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Forward)) : b,
    _BACKENDS,
)

# Backends excluding GTPSA (for models where GTPSA's C library cannot handle the computation).
const _BACKENDS_NO_GTPSA = filter(b -> !isa(b, AutoGTPSA), _BACKENDS)
const _BACKENDS_RTA_NO_GTPSA = filter(b -> !isa(b, AutoGTPSA), _BACKENDS_RTA)

@testset "SatelliteToolboxDifferentiability.jl" begin
    include("space_indices.jl")
    include("atmospheric_models.jl")
    include("geomagnetic_models.jl")
end
