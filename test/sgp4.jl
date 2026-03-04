## Description #############################################################################
#
# Tests related to differentiation of the SGP4 orbit propagator.
#
############################################################################################

@testset "SGP4 Differentiability" begin

    epoch = datetime2julian(DateTime("2024-01-01T00:00:00"))

    # Input: [Δt (min), epoch (JD), n₀ (rad/min), e₀, i₀ (rad), Ω₀ (rad), ω₀ (rad), M₀ (rad), bstar]
    test_cases = [
        # :sgp4 — near-earth, perigee ≥ 220 km
        (
            name  = "ISS-like 30min",
            input = [30.0, epoch, 15.5 * 2π / 1440, 0.0002, deg2rad(51.6), deg2rad(120.0), deg2rad(80.0), deg2rad(45.0), 5e-5],
        ),
        (
            name  = "ISS-like 90min",
            input = [90.0, epoch, 15.5 * 2π / 1440, 0.0002, deg2rad(51.6), deg2rad(120.0), deg2rad(80.0), deg2rad(45.0), 5e-5],
        ),
        (
            name  = "Sun-sync 60min",
            input = [60.0, epoch, 14.5 * 2π / 1440, 0.001, deg2rad(98.2), deg2rad(200.0), deg2rad(90.0), deg2rad(30.0), 3e-5],
        ),
        (
            name  = "Eccentric LEO 45min",
            input = [45.0, epoch, 14.8 * 2π / 1440, 0.03, deg2rad(28.5), deg2rad(45.0), deg2rad(270.0), deg2rad(150.0), 1e-4],
        ),
        # :sgp4_lowper — near-earth, perigee < 220 km
        (
            name  = "Low-perigee decay 30min",
            input = [30.0, epoch, 16.1 * 2π / 1440, 0.012, deg2rad(51.6), deg2rad(120.0), deg2rad(80.0), deg2rad(45.0), 1e-3],
        ),
        # :sdp4 — deep space, non-resonant
        (
            name  = "MEO non-resonant 60min",
            input = [60.0, epoch, 6.0 * 2π / 1440, 0.01, deg2rad(55.0), deg2rad(100.0), deg2rad(90.0), deg2rad(30.0), 1e-5],
        ),
        # :sdp4 — deep space, 24h synchronous resonance
        (
            name  = "GEO 24h-sync 120min",
            input = [120.0, epoch, 1.0027 * 2π / 1440, 0.0001, deg2rad(0.5), deg2rad(75.0), deg2rad(10.0), deg2rad(180.0), 1e-5],
        ),
        # :sdp4 — deep space, 12h geopotential resonance
        (
            name  = "Molniya 12h-resonance 180min",
            input = [180.0, epoch, 2.006 * 2π / 1440, 0.74, deg2rad(63.4), deg2rad(280.0), deg2rad(270.0), deg2rad(10.0), 5e-5],
        ),
    ]

    # ======================== SGP4 Propagation ========================
    # Differentiate [r_teme; v_teme] w.r.t. [Δt, epoch, n₀, e₀, i₀, Ω₀, ω₀, M₀, bstar].
    # Output is a 6×9 Jacobian covering the full state sensitivity.

    let scenarios = Scenario[]
        for tc in test_cases
            fn = (x) -> begin
                r, v, _ = sgp4(x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9])
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
            testset_name="SGP4 Propagation",
        )
    end

    # ======================== ML-∂SGP4 Propagation ========================
    # The default model is a const `MLdSGP4FrozenModel` (all SMatrix/SVector),
    # so the closure captures nothing mutable — safe for every AD backend.

    let scenarios = Scenario[]
        for tc in test_cases
            fn = (x) -> begin
                r, v, _ = ml_dsgp4(x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9])
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
            testset_name="ML-∂SGP4 Propagation",
        )
    end
end
