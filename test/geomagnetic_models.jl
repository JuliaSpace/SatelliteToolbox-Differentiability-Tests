## Description #############################################################################
#
# Tests related to differentiation of the Geomagnetic Field Models.
#
############################################################################################

@testset "Geomagnetic Model Differentiability" begin

    # ======================== Dipole Model ========================
    # geomagnetic_dipole_field(r_e, year) → SVector{3}
    # Differentiate w.r.t. the 3D position vector → 3×3 Jacobian.

    let scenarios = Scenario[]
        year = 2020.0
        R = 6.671e6

        positions = [
            [R, 0.0, 0.0],
            [0.0, R, 0.0],
            [0.0, 0.0, R],
            [R / sqrt(3.0), R / sqrt(3.0), R / sqrt(3.0)],
        ]

        for (i, pos) in enumerate(positions)
            fn = (x) -> collect(geomagnetic_dipole_field(x, year)) .+ sum(0 .* x)
            _, ref_jac = value_and_jacobian(fn, AutoFiniteDiff(), pos)

            push!(scenarios, Scenario{:jacobian,:out}(fn, pos;
                res1=ref_jac, name="position $i",
                prep_args=(; x=pos, contexts=()),
            ))
        end

        test_differentiation(_BACKENDS, scenarios;
            correctness=true, rtol=1e-2, detailed=true,
            testset_name="Dipole Model",
        )
    end

    # ======================== IGRF (Geocentric) ========================
    # igrf(date, r, λ, Ω) → SVector{3}
    # Differentiate w.r.t. [date, r, λ, Ω] → 3×4 Jacobian.
    let scenarios = Scenario[]
        igrf_inputs = [
            [2020.0, 6.671e6,  deg2rad(45),   deg2rad(-45)],
            [2020.0, 6.671e6,  0.0,           0.0],
            [2020.0, 6.671e6,  deg2rad(-30),  deg2rad(120)],
            [2020.0, 7.0e6,    deg2rad(60),   deg2rad(-90)],
        ]

        for (i, input) in enumerate(igrf_inputs)
            fn = (x) -> collect(igrf(x[1], x[2], x[3], x[4]; show_warnings=false, verbosity=Val(false)))
            _, ref_jac = value_and_jacobian(fn, AutoFiniteDiff(), input)

            push!(scenarios, Scenario{:jacobian,:out}(fn, input;
                res1=ref_jac, name="geocentric $i",
                prep_args=(; x=input, contexts=()),
            ))
        end

        test_differentiation(_BACKENDS_RTA_NO_GTPSA, scenarios;
            correctness=true, rtol=1e-2, detailed=true,
            testset_name="IGRF (Geocentric)",
        )
    end

    # ======================== IGRF (Geodetic) ========================
    # igrf(date, h, λ, Ω, Val(:geodetic)) → SVector{3}
    # Differentiate w.r.t. [date, h, λ, Ω] → 3×4 Jacobian.

    let scenarios = Scenario[]
        igrf_inputs = [
            [2020.0, 300e3,   deg2rad(45),   deg2rad(-45)],
            [2020.0, 200e3,   0.0,           0.0],
            [2020.0, 500e3,   deg2rad(-30),  deg2rad(-60)],
        ]

        for (i, input) in enumerate(igrf_inputs)
            fn = (x) -> collect(igrf(x[1], x[2], x[3], x[4], Val(:geodetic); show_warnings=false, verbosity=Val(false)))
            _, ref_jac = value_and_jacobian(fn, AutoFiniteDiff(), input)

            push!(scenarios, Scenario{:jacobian,:out}(fn, input;
                res1=ref_jac, name="geodetic $i",
                prep_args=(; x=input, contexts=()),
            ))
        end

        test_differentiation(_BACKENDS_NO_GTPSA, scenarios;
            correctness=true, rtol=1e-2, detailed=true,
            testset_name="IGRF (Geodetic)",
        )
    end

    # ======================== TaylorDiff (manual) ========================
    # TaylorDiff is not supported by DifferentiationInterface, so test it manually.
    # For vector→vector functions, compute the Jacobian column-by-column via
    # directional derivatives along each basis vector.

    #TODO: TAYLORDIFF FAILING
    #TODO: 1. IN REFERENCE FRAME ROTATIONS -- angle_to_dcm
    #TODO: 2. Fails in round() call igrf.jl:320
    # @testset "TaylorDiff" begin

    #     @testset "Dipole" begin
    #         year = 2020.0
    #         R = 6.671e6

    #         positions = [
    #             [R, 0.0, 0.0],
    #             [0.0, R, 0.0],
    #             [0.0, 0.0, R],
    #             [R / sqrt(3.0), R / sqrt(3.0), R / sqrt(3.0)],
    #         ]

    #         for pos in positions
    #             fn = (x) -> collect(geomagnetic_dipole_field(x, year)) .+ sum(0 .* x)
    #             _, ref_jac = value_and_jacobian(fn, AutoFiniteDiff(), pos)

    #             n = length(pos)
    #             jac_td = hcat([
    #                 TaylorDiff.derivative(fn, pos, [k == j ? 1.0 : 0.0 for j in 1:n], Val(1))
    #                 for k in 1:n
    #             ]...)

    #             @test ref_jac ≈ jac_td rtol=1e-2
    #         end
    #     end

    #     @testset "IGRF (Geocentric)" begin
    #         igrf_inputs = [
    #             [2020.0, 6.671e6,  deg2rad(45),   deg2rad(-45)],
    #             [2020.0, 6.671e6,  0.0,           0.0],
    #             [2020.0, 6.671e6,  deg2rad(-30),  deg2rad(120)],
    #             [2020.0, 7.0e6,    deg2rad(60),   deg2rad(-90)],
    #         ]

    #         for input in igrf_inputs
    #             fn = (x) -> collect(igrf(x...; show_warnings=false)) .+ sum(0 .* x)
    #             _, ref_jac = value_and_jacobian(fn, AutoFiniteDiff(), input)

    #             n = length(input)
    #             jac_td = hcat([
    #                 TaylorDiff.derivative(fn, input, [k == j ? 1.0 : 0.0 for j in 1:n], Val(1))
    #                 for k in 1:n
    #             ]...)

    #             @test ref_jac ≈ jac_td rtol=1e-2
    #         end
    #     end

    #     @testset "IGRF (Geodetic)" begin
    #         igrf_inputs = [
    #             [2020.0, 300e3,   deg2rad(45),   deg2rad(-45)],
    #             [2020.0, 200e3,   0.0,           0.0],
    #             [2020.0, 500e3,   deg2rad(-30),  deg2rad(-60)],
    #         ]

    #         for input in igrf_inputs
    #             fn = (x) -> collect(igrf(x..., Val(:geodetic); show_warnings=false)) .+ sum(0 .* x)
    #             _, ref_jac = value_and_jacobian(fn, AutoFiniteDiff(), input)

    #             n = length(input)
    #             jac_td = hcat([
    #                 TaylorDiff.derivative(fn, input, [k == j ? 1.0 : 0.0 for j in 1:n], Val(1))
    #                 for k in 1:n
    #             ]...)

    #             @test ref_jac ≈ jac_td rtol=1e-2
    #         end
    #     end

    # end

end
