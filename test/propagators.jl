## Description #############################################################################
#
# Tests related to differentiation of the SatelliteToolboxPropagators orbit propagators.
#
############################################################################################

@testset "Propagator Differentiability" begin

    epoch = datetime2julian(DateTime("2024-01-01T00:00:00"))

    # Input: [Δt (s), epoch (JD), a (m), e, i (rad), Ω (rad), ω (rad), f (rad)]
    #
    # Three cases spanning eccentricity range to exercise the Kepler equation solver:
    #   near-circular, moderate, and highly-elliptical.
    test_cases = [
        (
            name  = "Near-circular LEO",
            input = [1800.0, epoch, 6778.0e3, 0.0002, deg2rad(51.6), deg2rad(120.0), deg2rad(80.0), deg2rad(45.0)],
        ),
        (
            name  = "Eccentric LEO",
            input = [2700.0, epoch, 7200.0e3, 0.03, deg2rad(28.5), deg2rad(45.0), deg2rad(270.0), deg2rad(150.0)],
        ),
        (
            name  = "Molniya-like HEO",
            input = [10800.0, epoch, 26600.0e3, 0.74, deg2rad(63.4), deg2rad(280.0), deg2rad(270.0), deg2rad(10.0)],
        ),
    ]

    # ======================== Two-Body Propagation ========================
    # Differentiate [r_i; v_i] w.r.t. [Δt, epoch, a, e, i, Ω, ω, f].
    # Output is a 6×8 Jacobian covering the full state sensitivity.

    let scenarios = Scenario[]
        for tc in test_cases
            fn = (x) -> begin
                orb = KeplerianElements(x[2], x[3], x[4], x[5], x[6], x[7], x[8])
                r, v, _ = twobody(x[1], orb)
                return collect(vcat(r, v))
            end

            _, ref_jac = value_and_jacobian(fn, AutoFiniteDiff(), tc.input)

            push!(scenarios, Scenario{:jacobian,:out}(fn, tc.input;
                res1=ref_jac, name=tc.name,
                prep_args=(; x=tc.input, contexts=()),
            ))
        end

        test_differentiation(_BACKENDS_RTA_NO_GTPSA, scenarios;
            correctness=true, rtol=1e-2, detailed=true,
            testset_name="Two-Body Propagation",
        )
    end

    # ======================== J2 Propagation ========================
    # Differentiate [r_i; v_i] w.r.t. [Δt, epoch, a, e, i, Ω, ω, f].

    let scenarios = Scenario[]
        for tc in test_cases
            fn = (x) -> begin
                orb = KeplerianElements(x[2], x[3], x[4], x[5], x[6], x[7], x[8])
                r, v, _ = j2(x[1], orb)
                return collect(vcat(r, v))
            end

            _, ref_jac = value_and_jacobian(fn, AutoFiniteDiff(), tc.input)

            push!(scenarios, Scenario{:jacobian,:out}(fn, tc.input;
                res1=ref_jac, name=tc.name,
                prep_args=(; x=tc.input, contexts=()),
            ))
        end

        test_differentiation(_BACKENDS_RTA_NO_GTPSA, scenarios;
            correctness=true, rtol=1e-2, detailed=true,
            testset_name="J2 Propagation",
        )
    end

    # ======================== J2 Osculating Propagation ========================
    # Differentiate [r_i; v_i] w.r.t. [Δt, epoch, a, e, i, Ω, ω, f].

    let scenarios = Scenario[]
        for tc in test_cases
            fn = (x) -> begin
                orb = KeplerianElements(x[2], x[3], x[4], x[5], x[6], x[7], x[8])
                r, v, _ = j2osc(x[1], orb)
                return collect(vcat(r, v))
            end

            _, ref_jac = value_and_jacobian(fn, AutoFiniteDiff(), tc.input)

            push!(scenarios, Scenario{:jacobian,:out}(fn, tc.input;
                res1=ref_jac, name=tc.name,
                prep_args=(; x=tc.input, contexts=()),
            ))
        end

        test_differentiation(_BACKENDS_RTA_NO_GTPSA, scenarios;
            correctness=true, rtol=1e-2, detailed=true,
            testset_name="J2 Osculating Propagation",
        )
    end

    # ======================== J4 Propagation ========================
    # Differentiate [r_i; v_i] w.r.t. [Δt, epoch, a, e, i, Ω, ω, f].

    let scenarios = Scenario[]
        for tc in test_cases
            fn = (x) -> begin
                orb = KeplerianElements(x[2], x[3], x[4], x[5], x[6], x[7], x[8])
                r, v, _ = j4(x[1], orb)
                return collect(vcat(r, v))
            end

            _, ref_jac = value_and_jacobian(fn, AutoFiniteDiff(), tc.input)

            push!(scenarios, Scenario{:jacobian,:out}(fn, tc.input;
                res1=ref_jac, name=tc.name,
                prep_args=(; x=tc.input, contexts=()),
            ))
        end

        test_differentiation(_BACKENDS_RTA_NO_GTPSA, scenarios;
            correctness=true, rtol=1e-2, detailed=true,
            testset_name="J4 Propagation",
        )
    end

    # ======================== J4 Osculating Propagation ========================
    # Differentiate [r_i; v_i] w.r.t. [Δt, epoch, a, e, i, Ω, ω, f].

    let scenarios = Scenario[]
        for tc in test_cases
            fn = (x) -> begin
                orb = KeplerianElements(x[2], x[3], x[4], x[5], x[6], x[7], x[8])
                r, v, _ = j4osc(x[1], orb)
                return collect(vcat(r, v))
            end

            _, ref_jac = value_and_jacobian(fn, AutoFiniteDiff(), tc.input)

            push!(scenarios, Scenario{:jacobian,:out}(fn, tc.input;
                res1=ref_jac, name=tc.name,
                prep_args=(; x=tc.input, contexts=()),
            ))
        end

        test_differentiation(_BACKENDS_RTA_NO_GTPSA, scenarios;
            correctness=true, rtol=1e-2, detailed=true,
            testset_name="J4 Osculating Propagation",
        )
    end
end
