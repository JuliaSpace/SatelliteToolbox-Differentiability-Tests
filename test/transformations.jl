## Description #############################################################################
#
# Tests related to differentiation of the Reference Frame Transformations, Geodetic /
# Geocentric conversions, and Time conversions from SatelliteToolboxTransformations.
#
############################################################################################

@testset "Transformation Differentiability" begin

    eop_iau1980  = fetch_iers_eop()
    eop_iau2000a = fetch_iers_eop(Val(:IAU2000A))

    jd_utc = date_to_jd(2004, 4, 6, 7, 51, 28.386009)

    # ======================== EOP Functions ========================

    let scenarios = Scenario[]
        eop_functions = [
            ("IAU1980 x",  eop_iau1980.x),
            ("IAU1980 y",  eop_iau1980.y),
            ("IAU2000A x", eop_iau2000a.x),
            ("IAU2000A y", eop_iau2000a.y),
        ]

        for (name, f) in eop_functions
            fn = (x) -> f(x) + 0 * x
            _, ref = value_and_derivative(fn, AutoFiniteDiff(), jd_utc)
            push!(scenarios, Scenario{:derivative,:out}(fn, jd_utc;
                res1=ref, name=name,
                prep_args=(; x=jd_utc, contexts=()),
            ))
        end

        test_differentiation(_BACKENDS_NO_GTPSA, scenarios;
            correctness=true, rtol=1e-4, detailed=true,
            testset_name="EOP Functions",
        )
    end

    # ======================== ECEF to ECEF ========================

    let scenarios = Scenario[]
        frame_sets = (
            ("ITRF→PEF",  ITRF(), PEF(),  eop_iau1980),
            ("PEF→ITRF",  PEF(),  ITRF(), eop_iau1980),
            ("ITRF→TIRS", ITRF(), TIRS(), eop_iau2000a),
            ("TIRS→ITRF", TIRS(), ITRF(), eop_iau2000a),
        )

        for (name, from, to, eop) in frame_sets
            fn = (x) -> Array(r_ecef_to_ecef(from, to, x, eop)) .+ 0 * x
            _, ref = value_and_derivative(fn, AutoFiniteDiff(), jd_utc)
            push!(scenarios, Scenario{:derivative,:out}(fn, jd_utc;
                res1=ref, name=name,
                prep_args=(; x=jd_utc, contexts=()),
            ))
        end

        test_differentiation(_BACKENDS_RTA_NO_GTPSA, scenarios;
            correctness=true, rtol=1e-4, detailed=true,
            testset_name="ECEF to ECEF",
        )
    end

    # ======================== ECEF to ECI ========================

    let scenarios = Scenario[]
        frame_sets = (
            ("ITRF→GCRF IAU1980",    ITRF(), GCRF(),   eop_iau1980),
            ("ITRF→TEME IAU1980",    ITRF(), TEME(),   eop_iau1980),
            ("ITRF→GCRF IAU2000A",   ITRF(), GCRF(),   eop_iau2000a),
            ("ITRF→MJ2000 IAU2000A", ITRF(), MJ2000(), eop_iau2000a),
        )

        for (name, from, to, eop) in frame_sets
            fn = (x) -> Array(r_ecef_to_eci(from, to, x, eop)) .+ 0 * x
            _, ref = value_and_derivative(fn, AutoFiniteDiff(), jd_utc)
            push!(scenarios, Scenario{:derivative,:out}(fn, jd_utc;
                res1=ref, name=name,
                prep_args=(; x=jd_utc, contexts=()),
            ))
        end

        test_differentiation(_BACKENDS_RTA_NO_GTPSA, scenarios;
            correctness=true, rtol=2e-1, detailed=true,
            testset_name="ECEF to ECI",
        )
    end

    # ======================== ECI to ECEF ========================

    let scenarios = Scenario[]
        frame_sets = (
            ("GCRF→ITRF IAU1980",    GCRF(),   ITRF(), eop_iau1980),
            ("TEME→ITRF IAU1980",    TEME(),   ITRF(), eop_iau1980),
            ("GCRF→ITRF IAU2000A",   GCRF(),   ITRF(), eop_iau2000a),
            ("MJ2000→ITRF IAU2000A", MJ2000(), ITRF(), eop_iau2000a),
        )

        for (name, from, to, eop) in frame_sets
            fn = (x) -> Array(r_eci_to_ecef(from, to, x, eop)) .+ 0 * x
            _, ref = value_and_derivative(fn, AutoFiniteDiff(), jd_utc)
            push!(scenarios, Scenario{:derivative,:out}(fn, jd_utc;
                res1=ref, name=name,
                prep_args=(; x=jd_utc, contexts=()),
            ))
        end

        test_differentiation(_BACKENDS_RTA_NO_GTPSA, scenarios;
            correctness=true, rtol=2e-1, detailed=true,
            testset_name="ECI to ECEF",
        )
    end

    # ======================== ECI to ECI (Single Epoch) ========================

    let scenarios = Scenario[]
        frame_sets = (
            ("GCRF→TOD",    GCRF(),   TOD(),    eop_iau1980),
            ("TOD→GCRF",    TOD(),    GCRF(),   eop_iau1980),
            ("GCRF→TEME",   GCRF(),   TEME(),   eop_iau1980),
            ("TEME→GCRF",   TEME(),   GCRF(),   eop_iau1980),
            ("GCRF→CIRS",   GCRF(),   CIRS(),   eop_iau2000a),
            ("CIRS→GCRF",   CIRS(),   GCRF(),   eop_iau2000a),
            ("GCRF→MJ2000", GCRF(),   MJ2000(), eop_iau2000a),
            ("MJ2000→GCRF", MJ2000(), GCRF(),   eop_iau2000a),
            ("GCRF→ERS",    GCRF(),   ERS(),    eop_iau2000a),
            ("ERS→GCRF",    ERS(),    GCRF(),   eop_iau2000a),
        )

        for (name, from, to, eop) in frame_sets
            fn = (x) -> Array(r_eci_to_eci(from, to, x, eop)) .+ 0 * x
            _, ref = value_and_derivative(fn, AutoFiniteDiff(), jd_utc)
            push!(scenarios, Scenario{:derivative,:out}(fn, jd_utc;
                res1=ref, name=name,
                prep_args=(; x=jd_utc, contexts=()),
            ))
        end

        test_differentiation(_BACKENDS_RTA_NO_GTPSA, scenarios;
            correctness=true, rtol=2e-1, detailed=true,
            testset_name="ECI to ECI (Single Epoch)",
        )
    end

    # ======================== ECI to ECI (Two Epochs) ========================

    let scenarios = Scenario[]
        frame_sets = (
            ("MOD→TOD",   MOD(),   TOD(),   eop_iau1980),
            ("TOD→MOD",   TOD(),   MOD(),   eop_iau1980),
            ("TOD→TEME",  TOD(),   TEME(),  eop_iau1980),
            ("TEME→TOD",  TEME(),  TOD(),   eop_iau1980),
            ("CIRS→CIRS", CIRS(),  CIRS(),  eop_iau2000a),
            ("ERS→MOD06", ERS(),   MOD06(), eop_iau2000a),
            ("MOD06→ERS", MOD06(), ERS(),   eop_iau2000a),
        )

        for (name, from, to, eop) in frame_sets
            fn = (x) -> Array(r_eci_to_eci(from, x, to, x, eop)) .+ 0 * x
            _, ref = value_and_derivative(fn, AutoFiniteDiff(), jd_utc)
            push!(scenarios, Scenario{:derivative,:out}(fn, jd_utc;
                res1=ref, name=name,
                prep_args=(; x=jd_utc, contexts=()),
            ))
        end

        test_differentiation(_BACKENDS_RTA_NO_GTPSA, scenarios;
            correctness=true, atol=1e-4, detailed=true,
            testset_name="ECI to ECI (Two Epochs)",
        )
    end

    # ======================== Geodetic / Geocentric ========================

    let scenarios = Scenario[]
        ecef_pos = [7000e3, 0.0, 7000e3]

        fn = (x) -> collect(ecef_to_geocentric(x)) .+ sum(0 .* x)
        _, ref = value_and_jacobian(fn, AutoFiniteDiff(), ecef_pos)
        push!(scenarios, Scenario{:jacobian,:out}(fn, ecef_pos;
            res1=ref, name="ECEF→Geocentric",
            prep_args=(; x=ecef_pos, contexts=()),
        ))

        geocentric_state = [deg2rad(45.0), deg2rad(0.0), 7000 * √2]
        fn = (x) -> collect(geocentric_to_ecef(x)) .+ sum(0 .* x)
        _, ref = value_and_jacobian(fn, AutoFiniteDiff(), geocentric_state)
        push!(scenarios, Scenario{:jacobian,:out}(fn, geocentric_state;
            res1=ref, name="Geocentric→ECEF",
            prep_args=(; x=geocentric_state, contexts=()),
        ))

        fn = (x) -> collect(ecef_to_geodetic(x)) .+ sum(0 .* x)
        _, ref = value_and_jacobian(fn, AutoFiniteDiff(), ecef_pos)
        push!(scenarios, Scenario{:jacobian,:out}(fn, ecef_pos;
            res1=ref, name="ECEF→Geodetic",
            prep_args=(; x=ecef_pos, contexts=()),
        ))

        geodetic_state = [deg2rad(45.0), deg2rad(0.0), 400.0]
        fn = (x) -> collect(geodetic_to_ecef(x)) .+ sum(0 .* x)
        _, ref = value_and_jacobian(fn, AutoFiniteDiff(), geodetic_state)
        push!(scenarios, Scenario{:jacobian,:out}(fn, geodetic_state;
            res1=ref, name="Geodetic→ECEF",
            prep_args=(; x=geodetic_state, contexts=()),
        ))

        geocentric_state2 = [deg2rad(45.0), 7000 * √2]
        fn = (x) -> collect(geocentric_to_geodetic(x)) .+ sum(0 .* x)
        _, ref = value_and_jacobian(fn, AutoFiniteDiff(), geocentric_state2)
        push!(scenarios, Scenario{:jacobian,:out}(fn, geocentric_state2;
            res1=ref, name="Geocentric→Geodetic",
            prep_args=(; x=geocentric_state2, contexts=()),
        ))

        geodetic_state2 = [deg2rad(45.0), 400.0]
        fn = (x) -> collect(geodetic_to_geocentric(x)) .+ sum(0 .* x)
        _, ref = value_and_jacobian(fn, AutoFiniteDiff(), geodetic_state2)
        push!(scenarios, Scenario{:jacobian,:out}(fn, geodetic_state2;
            res1=ref, name="Geodetic→Geocentric",
            prep_args=(; x=geodetic_state2, contexts=()),
        ))

        test_differentiation(_BACKENDS_NO_GTPSA, scenarios;
            correctness=true, rtol=2e-1, detailed=true,
            testset_name="Geodetic / Geocentric",
        )
    end

    # ======================== Leap Seconds ========================

    let scenarios = Scenario[]
        fn = (x) -> get_Δat(x) + 0 * x
        _, ref = value_and_derivative(fn, AutoFiniteDiff(), jd_utc)
        push!(scenarios, Scenario{:derivative,:out}(fn, jd_utc;
            res1=ref, name="get_Δat",
            prep_args=(; x=jd_utc, contexts=()),
        ))

        test_differentiation(_BACKENDS_NO_GTPSA, scenarios;
            correctness=true, detailed=true,
            testset_name="Leap Seconds",
        )
    end

    # ======================== Time: UTC ↔ UT1 ========================

    let scenarios = Scenario[]
        ΔUT1 = -0.463326

        for (fname, func) in [("jd_utc_to_ut1", jd_utc_to_ut1), ("jd_ut1_to_utc", jd_ut1_to_utc)]
            fn = (x) -> func(x, ΔUT1) + 0 * x
            _, ref = value_and_derivative(fn, AutoFiniteDiff(), jd_utc)
            push!(scenarios, Scenario{:derivative,:out}(fn, jd_utc;
                res1=ref, name="$fname(jd, ΔUT1) w.r.t. jd",
                prep_args=(; x=jd_utc, contexts=()),
            ))

            fn = (x) -> func(jd_utc, x) + 0 * x
            _, ref = value_and_derivative(fn, AutoFiniteDiff(), ΔUT1)
            push!(scenarios, Scenario{:derivative,:out}(fn, ΔUT1;
                res1=ref, name="$fname(jd, ΔUT1) w.r.t. ΔUT1",
                prep_args=(; x=ΔUT1, contexts=()),
            ))

            fn = (x) -> func(x, eop_iau1980) + 0 * x
            _, ref = value_and_derivative(fn, AutoFiniteDiff(), jd_utc)
            push!(scenarios, Scenario{:derivative,:out}(fn, jd_utc;
                res1=ref, name="$fname(jd, eop_iau1980)",
                prep_args=(; x=jd_utc, contexts=()),
            ))

            fn = (x) -> func(x, eop_iau2000a) + 0 * x
            _, ref = value_and_derivative(fn, AutoFiniteDiff(), jd_utc)
            push!(scenarios, Scenario{:derivative,:out}(fn, jd_utc;
                res1=ref, name="$fname(jd, eop_iau2000a)",
                prep_args=(; x=jd_utc, contexts=()),
            ))
        end

        test_differentiation(_BACKENDS_RTA_NO_GTPSA, scenarios;
            correctness=true, atol=1e-4, detailed=true,
            testset_name="Time UTC ↔ UT1",
        )
    end

    # ======================== Time: UTC ↔ TT ========================

    let scenarios = Scenario[]
        for (fname, func) in [("jd_utc_to_tt", jd_utc_to_tt), ("jd_tt_to_utc", jd_tt_to_utc)]
            fn = (x) -> func(x) + 0 * x
            _, ref = value_and_derivative(fn, AutoFiniteDiff(), jd_utc)
            push!(scenarios, Scenario{:derivative,:out}(fn, jd_utc;
                res1=ref, name=fname,
                prep_args=(; x=jd_utc, contexts=()),
            ))
        end

        test_differentiation(_BACKENDS_NO_GTPSA, scenarios;
            correctness=true, rtol=1e-4, detailed=true,
            testset_name="Time UTC ↔ TT",
        )
    end

end
