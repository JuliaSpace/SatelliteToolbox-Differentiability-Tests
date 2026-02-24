## Description #############################################################################
#
# Tests related to differentiation of the Atmospheric Models.
#
############################################################################################

@testset "Atmospheric Model Differentiability" begin
    SpaceIndices.init()

    hs = collect(90:200:1090) .* 1000.0
    hs_hp = collect(100:200:900) .* 1000.0  # Harris-Priester valid range: 100-1000 km

    # ======================== Exponential Atmosphere ========================

    let scenarios = Scenario[]
        for h in hs
            fn = (x) -> AtmosphericModels.exponential(x) + 0 * x
            _, ref = value_and_derivative(fn, AutoFiniteDiff(), h)
            push!(scenarios, Scenario{:derivative,:out}(fn, h;
                res1=ref, name="h=$(Int(h÷1000))km",
            ))
        end

        test_differentiation(_BACKENDS, scenarios;
            correctness=true, rtol=1e-4, detailed=true,
            testset_name="Exponential Atmosphere",
        )
    end

    # ===================== Jacchia-Roberts 1971 =====================

    let scenarios = Scenario[]
        for h in hs
            instant = datetime2julian(DateTime("2023-01-01T10:00:00"))
            ϕ_gd    = deg2rad(-23)
            λ       = deg2rad(-45)
            F10     = 152.6
            F10ₐ    = 159.12345679012347
            Kp      = 2.667

            input   = [instant; ϕ_gd; λ; h; F10; F10ₐ; Kp]
            input_r = input[1:4]

            fn = (x) -> AtmosphericModels.jr1971(x[1], x[2], x[3], x[4], x[5], x[6], x[7]; verbose=Val(false)).total_density
            fn_r = (x) -> AtmosphericModels.jr1971(x[1], x[2], x[3], x[4]; verbose=Val(false)).total_density

            _, ref   = value_and_gradient(fn, AutoFiniteDiff(), input)
            _, ref_r = value_and_gradient(fn_r, AutoFiniteDiff(), input_r)

            push!(scenarios, Scenario{:gradient,:out}(fn, input;
                res1=ref, name="full h=$(Int(h÷1000))km",
                prep_args=(; x=input, contexts=()),
            ))
            push!(scenarios, Scenario{:gradient,:out}(fn_r, input_r;
                res1=ref_r, name="reduced h=$(Int(h÷1000))km",
                prep_args=(; x=input_r, contexts=()),
            ))
        end

        test_differentiation(_BACKENDS_RTA_NO_GTPSA, scenarios;
            correctness=true, atol=5e-1, rtol=5e-1, detailed=true,
            testset_name="JR1971 Atmosphere",
        )
    end

    # ======================== NRLMSISE-00 ========================

    let scenarios = Scenario[]
        for h in hs
            instant = datetime2julian(DateTime("2023-01-01T10:00:00"))
            ϕ_gd    = deg2rad(-23)
            λ       = deg2rad(-45)
            F10     = 121.0
            F10ₐ    = 80.0
            ap      = 7.0

            input   = [instant; h; ϕ_gd; λ; F10ₐ; F10; ap]
            input_r = input[1:4]

            fn   = (x) -> AtmosphericModels.nrlmsise00(x[1], x[2], x[3], x[4], x[5], x[6], x[7]).total_density
            fn_r = (x) -> AtmosphericModels.nrlmsise00(x[1], x[2], x[3], x[4]; verbose=Val(false)).total_density

            _, ref   = value_and_gradient(fn, AutoFiniteDiff(), input)
            _, ref_r = value_and_gradient(fn_r, AutoFiniteDiff(), input_r)

            push!(scenarios, Scenario{:gradient,:out}(fn, input;
                res1=ref, name="full h=$(Int(h÷1000))km",
                prep_args=(; x=input, contexts=()),
            ))
            push!(scenarios, Scenario{:gradient,:out}(fn_r, input_r;
                res1=ref_r, name="reduced h=$(Int(h÷1000))km",
                prep_args=(; x=input_r, contexts=()),
            ))
        end

        test_differentiation(_BACKENDS_RTA_NO_GTPSA, scenarios;
            correctness=true, atol=1e-5, detailed=true,
            testset_name="NRLMSISE-00 Atmosphere",
        )
    end

    # ===================== Jacchia-Bowman 2008 =====================

    let scenarios = Scenario[]
        for h in hs
            input = [
                datetime2julian(DateTime("2023-01-01T10:00:00"));
                0.0; 0.0; h;
                100.0; 100.0; 100.0; 100.0; 100.0; 100.0; 100.0; 100.0; 85.0
            ]
            input_r = input[1:4]

            fn = (x) -> AtmosphericModels.jb2008(x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]; verbose=Val(false)).total_density
            fn_r = (x) -> AtmosphericModels.jb2008(x[1], x[2], x[3], x[4]; verbose=Val(false)).total_density

            _, ref   = value_and_gradient(fn, AutoFiniteDiff(), input)
            _, ref_r = value_and_gradient(fn_r, AutoFiniteDiff(), input_r)

            push!(scenarios, Scenario{:gradient,:out}(fn, input;
                res1=ref, name="full h=$(Int(h÷1000))km",
                prep_args=(; x=input, contexts=()),
            ))
            push!(scenarios, Scenario{:gradient,:out}(fn_r, input_r;
                res1=ref_r, name="reduced h=$(Int(h÷1000))km",
                prep_args=(; x=input_r, contexts=()),
            ))
        end

        test_differentiation(_BACKENDS_NO_GTPSA, scenarios;
            correctness=true, rtol=2e-1, detailed=true,
            testset_name="JB2008 Atmosphere",
        )
    end

    # ===================== Harris-Priester =====================

    # let scenarios = Scenario[]
    #     for h in hs_hp
    #         instant = datetime2julian(DateTime("2023-01-01T10:00:00"))
    #         ϕ_gd    = deg2rad(-23)
    #         λ       = deg2rad(-45)

    #         input = [instant; ϕ_gd; λ; h]

    #         fn = (x) -> AtmosphericModels.harrispriester(x[1], x[2], x[3], x[4]).total_density

    #         _, ref = value_and_gradient(fn, AutoFiniteDiff(), input)

    #         push!(scenarios, Scenario{:gradient,:out}(fn, input;
    #             res1=ref, name="h=$(Int(h÷1000))km",
    #             prep_args=(; x=input, contexts=()),
    #         ))
    #     end

    #     test_differentiation(_BACKENDS_RTA, scenarios;
    #         correctness=true, rtol=2e-1, detailed=true,
    #         testset_name="Harris-Priester Atmosphere",
    #     )
    # end

    # # ================ Harris-Priester Modified ================

    # let scenarios = Scenario[]
    #     for h in hs_hp
    #         instant = datetime2julian(DateTime("2023-01-01T10:00:00"))
    #         ϕ_gd    = deg2rad(-23)
    #         λ       = deg2rad(-45)
    #         F10ₐ    = 150.0

    #         input   = [instant; ϕ_gd; λ; h; F10ₐ]
    #         input_r = input[1:4]

    #         fn = (x) -> AtmosphericModels.harrispriester_modified(x[1], x[2], x[3], x[4], x[5]).total_density
    #         fn_r = (x) -> AtmosphericModels.harrispriester_modified(x[1], x[2], x[3], x[4]).total_density

    #         _, ref   = value_and_gradient(fn, AutoFiniteDiff(), input)
    #         _, ref_r = value_and_gradient(fn_r, AutoFiniteDiff(), input_r)

    #         push!(scenarios, Scenario{:gradient,:out}(fn, input;
    #             res1=ref, name="full h=$(Int(h÷1000))km",
    #             prep_args=(; x=input, contexts=()),
    #         ))
    #         push!(scenarios, Scenario{:gradient,:out}(fn_r, input_r;
    #             res1=ref_r, name="reduced h=$(Int(h÷1000))km",
    #             prep_args=(; x=input_r, contexts=()),
    #         ))
    #     end

    #     test_differentiation(_BACKENDS_RTA, scenarios;
    #         correctness=true, rtol=2e-1, detailed=true,
    #         testset_name="Harris-Priester Modified Atmosphere",
    #     )
    # end

    #TODO: TaylorDiff tests failing in multiple spots (usually around round() or rem2pi calls)

#     # ===================== TaylorDiff (manual) =====================
#     # TaylorDiff is not supported by DifferentiationInterface, so test it manually.
#     # For vector-input functions, compute the gradient via directional derivatives
#     # along each basis vector.

#     @testset "TaylorDiff" begin

#         @testset "Exponential" begin
#             for h in hs
#                 fn = (x) -> AtmosphericModels.exponential(x) + 0 * x
#                 _, df_fd = value_and_derivative(fn, AutoFiniteDiff(), h)
#                 df_td = TaylorDiff.derivative(fn, h, Val(1))
#                 @test df_fd ≈ df_td rtol=1e-4
#             end
#         end

#         @testset "JR1971" begin
#             for h in hs
#                 instant = datetime2julian(DateTime("2023-01-01T10:00:00"))
#                 ϕ_gd    = deg2rad(-23)
#                 λ       = deg2rad(-45)
#                 F10     = 152.6
#                 F10ₐ    = 159.12345679012347
#                 Kp      = 2.667

#                 input   = [instant; ϕ_gd; λ; h; F10; F10ₐ; Kp]
#                 input_r = input[1:4]

#                 fn = (x) -> AtmosphericModels.jr1971(x...; verbose=Val(false)).total_density + sum(0 .* x)

#                 _, df_fd = value_and_gradient(fn, AutoFiniteDiff(), input)
#                 df_td = [TaylorDiff.derivative(fn, input, [i == j ? 1.0 : 0.0 for j in eachindex(input)], Val(1)) for i in eachindex(input)]
#                 @test df_fd ≈ df_td atol=5e-1

#                 _, df_fd_r = value_and_gradient(fn, AutoFiniteDiff(), input_r)
#                 df_td_r = [TaylorDiff.derivative(fn, input_r, [i == j ? 1.0 : 0.0 for j in eachindex(input_r)], Val(1)) for i in eachindex(input_r)]
#                 @test df_fd_r ≈ df_td_r rtol=5e-1
#             end
#         end

#         @testset "NRLMSISE-00" begin
#             for h in hs
#                 instant = datetime2julian(DateTime("2023-01-01T10:00:00"))
#                 ϕ_gd    = deg2rad(-23)
#                 λ       = deg2rad(-45)
#                 F10     = 121.0
#                 F10ₐ    = 80.0
#                 ap      = 7.0

#                 input   = [instant; h; ϕ_gd; λ; F10ₐ; F10; ap]
#                 input_r = input[1:4]

#                 fn   = (x) -> AtmosphericModels.nrlmsise00(x...).total_density + sum(0 .* x)
#                 fn_r = (x) -> AtmosphericModels.nrlmsise00(x...; verbose=Val(false)).total_density + sum(0 .* x)

#                 _, df_fd = value_and_gradient(fn, AutoFiniteDiff(), input)
#                 df_td = [TaylorDiff.derivative(fn, input, [i == j ? 1.0 : 0.0 for j in eachindex(input)], Val(1)) for i in eachindex(input)]
#                 @test df_fd ≈ df_td atol=1e-5

#                 _, df_fd_r = value_and_gradient(fn_r, AutoFiniteDiff(), input_r)
#                 df_td_r = [TaylorDiff.derivative(fn_r, input_r, [i == j ? 1.0 : 0.0 for j in eachindex(input_r)], Val(1)) for i in eachindex(input_r)]
#                 @test df_fd_r ≈ df_td_r atol=1e-5
#             end
#         end

#         @testset "JB2008" begin
#             for h in hs
#                 input = [
#                     datetime2julian(DateTime("2023-01-01T10:00:00"));
#                     0.0; 0.0; h;
#                     100.0; 100.0; 100.0; 100.0; 100.0; 100.0; 100.0; 100.0; 85.0
#                 ]
#                 input_r = input[1:4]

#                 fn = (x) -> AtmosphericModels.jb2008(x...; verbose=Val(false)).total_density + sum(0 .* x)

#                 _, df_fd = value_and_gradient(fn, AutoFiniteDiff(), input)
#                 df_td = [TaylorDiff.derivative(fn, input, [i == j ? 1.0 : 0.0 for j in eachindex(input)], Val(1)) for i in eachindex(input)]
#                 @test df_fd ≈ df_td rtol=2e-1

#                 _, df_fd_r = value_and_gradient(fn, AutoFiniteDiff(), input_r)
#                 df_td_r = [TaylorDiff.derivative(fn, input_r, [i == j ? 1.0 : 0.0 for j in eachindex(input_r)], Val(1)) for i in eachindex(input_r)]
#                 @test df_fd_r ≈ df_td_r rtol=2e-1
#             end
#         end

#         # @testset "Harris-Priester" begin
#         #     for h in hs_hp
#         #         instant = datetime2julian(DateTime("2023-01-01T10:00:00"))
#         #         ϕ_gd    = deg2rad(-23)
#         #         λ       = deg2rad(-45)

#         #         input = [instant; ϕ_gd; λ; h]

#         #         fn = (x) -> AtmosphericModels.harrispriester(x...) + sum(0 .* x)

#         #         _, df_fd = value_and_gradient(fn, AutoFiniteDiff(), input)
#         #         df_td = [TaylorDiff.derivative(fn, input, [i == j ? 1.0 : 0.0 for j in eachindex(input)], Val(1)) for i in eachindex(input)]
#         #         @test df_fd ≈ df_td rtol=2e-1
#         #     end
#         # end

#         # @testset "Harris-Priester Modified" begin
#         #     for h in hs_hp
#         #         instant = datetime2julian(DateTime("2023-01-01T10:00:00"))
#         #         ϕ_gd    = deg2rad(-23)
#         #         λ       = deg2rad(-45)
#         #         F10ₐ    = 150.0

#         #         input   = [instant; ϕ_gd; λ; h; F10ₐ]
#         #         input_r = input[1:4]

#         #         fn = (x) -> AtmosphericModels.harrispriester_modified(x...) + sum(0 .* x)

#         #         _, df_fd = value_and_gradient(fn, AutoFiniteDiff(), input)
#         #         df_td = [TaylorDiff.derivative(fn, input, [i == j ? 1.0 : 0.0 for j in eachindex(input)], Val(1)) for i in eachindex(input)]
#         #         @test df_fd ≈ df_td rtol=2e-1

#         #         _, df_fd_r = value_and_gradient(fn, AutoFiniteDiff(), input_r)
#         #         df_td_r = [TaylorDiff.derivative(fn, input_r, [i == j ? 1.0 : 0.0 for j in eachindex(input_r)], Val(1)) for i in eachindex(input_r)]
#         #         @test df_fd_r ≈ df_td_r rtol=2e-1
#         #     end
#         # end

#     end

#     SpaceIndices.destroy()
end
