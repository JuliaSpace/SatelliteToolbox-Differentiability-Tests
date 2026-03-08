## Description #############################################################################
#
# Tests related to differentiation of the Gravity Models from
# SatelliteToolboxGravityModels.
#
############################################################################################

@testset "Gravity Model Differentiability" begin

    grav_model = GravityModels.load(IcgemFile, fetch_icgem_file(:EGM96))

    lat   = 27.5
    lon   = 235.3
    r_itrf = Array(geodetic_to_ecef(lat, lon, 0))
    time  = 0.0

    _BACKENDS_CONST_NO_GTPSA = map(
        b -> b isa AutoEnzyme ? AutoEnzyme(; function_annotation=Enzyme.Const) : b,
        _BACKENDS_NO_GTPSA,
    )

    # ======================== State Differentiation ========================

    let scenarios = Scenario[]
        fn_accel = (x) -> Array(GravityModels.gravitational_acceleration(grav_model, x)) .+ sum(0 .* x)
        _, ref = value_and_jacobian(fn_accel, AutoFiniteDiff(), r_itrf)
        push!(scenarios, Scenario{:jacobian,:out}(fn_accel, r_itrf;
            res1=ref, name="gravitational_acceleration(model, x)",
            prep_args=(; x=r_itrf, contexts=()),
        ))

        fn_pot = (x) -> GravityModels.gravitational_potential(grav_model, x) + sum(0 .* x)
        _, ref = value_and_gradient(fn_pot, AutoFiniteDiff(), r_itrf)
        push!(scenarios, Scenario{:gradient,:out}(fn_pot, r_itrf;
            res1=ref, name="gravitational_potential(model, x)",
            prep_args=(; x=r_itrf, contexts=()),
        ))

        fn_deriv = (x) -> collect(GravityModels.gravitational_field_derivative(grav_model, x)) .+ sum(0 .* x)
        _, ref = value_and_jacobian(fn_deriv, AutoFiniteDiff(), r_itrf)
        push!(scenarios, Scenario{:jacobian,:out}(fn_deriv, r_itrf;
            res1=ref, name="gravitational_field_derivative(model, x)",
            prep_args=(; x=r_itrf, contexts=()),
        ))

        test_differentiation(_BACKENDS_CONST_NO_GTPSA, scenarios;
            correctness=true, rtol=1e-4, detailed=true,
            testset_name="State Differentiation",
        )
    end

    # ======================== Time Differentiation ========================

    let scenarios = Scenario[]
        fn_accel = (x) -> Array(GravityModels.gravitational_acceleration(grav_model, r_itrf, x)) .+ 0 * x
        _, ref = value_and_derivative(fn_accel, AutoFiniteDiff(), time)
        push!(scenarios, Scenario{:derivative,:out}(fn_accel, time;
            res1=ref, name="gravitational_acceleration(model, r, t)",
            prep_args=(; x=time, contexts=()),
        ))

        fn_pot = (x) -> GravityModels.gravitational_potential(grav_model, r_itrf, x) + 0 * x
        _, ref = value_and_derivative(fn_pot, AutoFiniteDiff(), time)
        push!(scenarios, Scenario{:derivative,:out}(fn_pot, time;
            res1=ref, name="gravitational_potential(model, r, t)",
            prep_args=(; x=time, contexts=()),
        ))

        fn_deriv = (x) -> collect(GravityModels.gravitational_field_derivative(grav_model, r_itrf, x)) .+ 0 * x
        _, ref = value_and_derivative(fn_deriv, AutoFiniteDiff(), time)
        push!(scenarios, Scenario{:derivative,:out}(fn_deriv, time;
            res1=ref, name="gravitational_field_derivative(model, r, t)",
            prep_args=(; x=time, contexts=()),
        ))

        test_differentiation(_BACKENDS_CONST_NO_GTPSA, scenarios;
            correctness=true, rtol=1e-4, detailed=true,
            testset_name="Time Differentiation",
        )
    end

end
