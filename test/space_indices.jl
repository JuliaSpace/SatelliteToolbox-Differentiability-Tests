## Description #############################################################################
#
# Tests for differentiation across the space indices.
#
# Currently Supported & Tested:
#
#   Enzyme, ForwardDiff, FiniteDiff, GTPSA, Mooncake, PolyesterForwardDiff,
#   TaylorDiff, Zygote
#
############################################################################################

const _INDICES = [
    :F10obs
    :F10obs_avg_center81
    :F10obs_avg_last81
    :F10adj
    :F10adj_avg_center81
    :F10adj_avg_last81
    :Ap
    :Ap_daily
    :Kp
    :Kp_daily
    :Cp
    :C9
    :ISN
    :BSRN
    :ND
    :DTC
    :S10
    :M10
    :Y10
    :S81a
    :M81a
    :Y81a
#    :Hp30
#    :Hp60
#    :Ap30
#    :Ap60
#    :Dst           # Not on main branch yet
#    :DTC_Dst       # Not on main branch yet
]

@testset "Space Index Differentiability" begin
    SpaceIndices.init()
    #SpaceIndices.init(SpaceIndices.Dst)  # Dst excluded from default init
    dt = DateTime(2020, 6, 19, 9, 30, 0)
    jd = datetime2julian(dt)

    # For hourly-cadence indices (Dst, DTC_Dst, DTC), FiniteDiff's default relative step
    # size (√ε × |jd| ≈ 0.04 days ≈ 1 hour) is comparable to the knot spacing,
    # causing the FD derivative to straddle multiple intervals. Use a manual central
    # difference with a step small enough to stay within one interpolation interval.
    _HOURLY_INDICES = (:DTC,) #:Dst, :DTC_Dst)

    # Build Scenarios with reference derivatives computed via finite differences.
    scenarios = Scenario[]

    for index in _INDICES
        # The `.+ 0 * x` ensures Zygote returns a numeric 0.0 pullback instead of
        # `nothing` for constant-valued indices. The mathematical result is unchanged.
        # See https://github.com/JuliaDiff/DifferentiationInterface.jl/pull/604
        fn = (x) -> reduce(vcat, space_index(Val(index), x)) .+ 0 * x

        if index ∈ _HOURLY_INDICES
            h_fd = 1e-6  # ~0.086 s — well within 1-hour knots
            ref_deriv = (fn(jd + h_fd) - fn(jd - h_fd)) / (2h_fd)
        else
            _, ref_deriv = value_and_derivative(fn, AutoFiniteDiff(), jd)
        end

        # Override prep_args so that preparation uses the actual Julian date instead of
        # zero(jd) = 0.0, which falls outside the valid data range for all indices.
        push!(scenarios,
            Scenario{:derivative,:out}(
                fn, jd;
                res1=ref_deriv,
                name=string(index),
                prep_args=(; x=jd, contexts=()),
            )
        )
    end

    # Test all DI backends against finite difference references
    test_differentiation(
        _BACKENDS,
        scenarios;
        correctness=true,
        rtol=1e-2,
        detailed=true,
    )

    # TaylorDiff is not supported by DifferentiationInterface, so test it manually.
    @testset "TaylorDiff" begin
        for index in _INDICES
            fn = (x) -> reduce(vcat, space_index(Val(index), x)) .+ 0 * x

            if index ∈ _HOURLY_INDICES
                h_fd = 1e-6
                f_fd  = fn(jd)
                df_fd = (fn(jd + h_fd) - fn(jd - h_fd)) / (2h_fd)
            else
                f_fd, df_fd = value_and_derivative(fn, AutoFiniteDiff(), jd)
            end

            f_td  = fn(jd)
            df_td = TaylorDiff.derivative(fn, jd, Val(1))

            @test f_fd == f_td
            @test df_fd ≈ df_td rtol=1e-2
        end
    end

    SpaceIndices.destroy()

end
