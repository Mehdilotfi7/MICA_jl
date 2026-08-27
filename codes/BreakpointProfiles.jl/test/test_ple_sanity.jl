using Test
using BreakpointProfiles
using Optim
using Random

@testset "PLE Sanity Tests" begin

    # =========================================================================
    # Test 1: Known quadratic model — PLE has an analytical solution
    # =========================================================================
    @testset "Quadratic model: known CI" begin
        # Model: loss(θ) = (θ₁ - 3.0)² + (θ₂ - 5.0)²
        # Profile θ₁: min_{θ₂} loss(θ₁, θ₂) = (θ₁ - 3.0)²
        # CI: |θ₁ - 3| ≤ √3.84 ≈ 1.96
        # So CI = [3 - 1.96, 3 + 1.96] = [1.04, 4.96]

        function quad_objective(params, cps)
            return (params[1] - 3.0)^2 + (params[2] - 5.0)^2
        end

        prob = ODEChangepointPLEProblem(
            objective = quad_objective,
            data = zeros(1, 1),
            loss_fn = (s, d) -> 0.0,
            changepoints = Int[],
            best_params = [3.0, 5.0],
            best_loss = 0.0,
            lb = [-10.0, -10.0],
            ub = [20.0, 20.0],
            param_names = ["θ₁", "θ₂"],
            n_global = 2,
            n_segment_specific = 0,
            n_obs = 1
        )

        opt = MultiStartBOBYQAConfig(n_starts=2, maxeval=500)
        options = ProfileOptions(samplesize=30, polish=false)
        result = profile_parameter(prob, 1; options=options, optimizer=opt)

        @test result.parameter == "θ₁"
        @test result.best_value ≈ 3.0
        @test result.best_loss ≈ 0.0 atol=1e-6

        # CI should be approximately [1.04, 4.96]
        @test result.ci_lower ≈ 1.04 atol=0.15
        @test result.ci_upper ≈ 4.96 atol=0.15
        @test result.identifiable == true
        @test result.n_failed == 0

        # Profile at best must give ≈ 0
        best_idx = findfirst(v -> abs(v - 3.0) < 0.01, result.values)
        if best_idx !== nothing
            @test result.losses[best_idx] ≈ 0.0 atol=0.01
        end
    end

    # =========================================================================
    # Test 2: Non-identifiable parameter — flat profile
    # =========================================================================
    @testset "Non-identifiable parameter" begin
        # Model: loss(θ) = (θ₁ - 2.0)²
        # θ₂ does not appear in the loss → non-identifiable
        # Profile θ₂: min_{θ₁} loss = 0 for all θ₂ → flat profile

        function flat_objective(params, cps)
            return (params[1] - 2.0)^2
        end

        prob = ODEChangepointPLEProblem(
            objective = flat_objective,
            data = zeros(1, 1),
            loss_fn = (s, d) -> 0.0,
            changepoints = Int[],
            best_params = [2.0, 0.0],
            best_loss = 0.0,
            lb = [-5.0, -5.0],
            ub = [10.0, 10.0],
            param_names = ["θ₁", "θ₂"],
            n_global = 2,
            n_segment_specific = 0,
            n_obs = 1
        )

        opt = MultiStartBOBYQAConfig(n_starts=2, maxeval=500)
        options = ProfileOptions(samplesize=20, polish=false)
        result = profile_parameter(prob, 2; options=options, optimizer=opt)

        # Profile should be flat → not identifiable
        @test result.identifiable == false

        # All losses should be ≈ 0
        finite_losses = filter(isfinite, result.losses)
        @test all(l -> l < 0.1, finite_losses)
    end

    # =========================================================================
    # Test 3: Self-consistency — profile at θ* must recover best_loss
    # =========================================================================
    @testset "Self-consistency at θ*" begin
        function rosenbrock2d(params, cps)
            x, y = params
            return (1 - x)^2 + 100 * (y - x^2)^2
        end

        prob = ODEChangepointPLEProblem(
            objective = rosenbrock2d,
            data = zeros(1, 1),
            loss_fn = (s, d) -> 0.0,
            changepoints = Int[],
            best_params = [1.0, 1.0],
            best_loss = 0.0,
            lb = [-5.0, -5.0],
            ub = [5.0, 5.0],
            param_names = ["x", "y"],
            n_global = 2,
            n_segment_specific = 0,
            n_obs = 1
        )

        opt = MultiStartBOBYQAConfig(n_starts=3, maxeval=1000)
        options = ProfileOptions(samplesize=20, polish=false)
        result = profile_parameter(prob, 1; options=options, optimizer=opt)

        # Should find the minimum at x=1
        @test result.best_value ≈ 1.0
        @test result.best_loss ≈ 0.0 atol=1e-4

        # Profile should be monotonically increasing away from minimum (locally)
        perm = sortperm(result.values)
        svals = result.values[perm]
        sloss = result.losses[perm]
        best_pos = argmin(abs.(svals .- 1.0))

        # Check monotonicity to the left of best (first 3 points)
        if best_pos > 2
            for i in (best_pos-2):(best_pos-1)
                @test sloss[i] >= sloss[i+1] - 0.1  # allow small tolerance
            end
        end
    end

    # =========================================================================
    # Test 4: GaussianLogNLL correctness
    # =========================================================================
    @testset "GaussianLogNLL computation" begin
        sim = [10.0 100.0; 20.0 200.0]  # 2 channels, 2 time points
        data = [10.0 100.0; 20.0 200.0]  # perfect match

        # With perfect match and any σ, NLL should be 0
        nll = GaussianLogNLL([1.0, 1.0])
        @test nll(sim, data) ≈ 0.0 atol=1e-10

        # With known residual
        sim2 = [exp(1.0) * 10.0 100.0; 20.0 200.0]  # first channel, first point: log(sim) - log(data) = 1.0
        nll_unit = GaussianLogNLL([1.0, 1.0])
        expected = 1.0^2  # only one non-zero residual on -2 log L scale
        @test nll_unit(sim2, data) ≈ expected atol=0.01

        # Single-σ constructor
        nll_single = GaussianLogNLL(2.0)
        @test length(nll_single.sigma_per_channel) == 1
        @test nll_single.sigma_per_channel[1] ≈ 2.0
    end

    # =========================================================================
    # Test 5: MultiStartBOBYQA improves on single-start
    # =========================================================================
    @testset "Multi-start ≤ single-start" begin
        # A function with a local minimum at [0,0] and a global minimum at [3,3]
        function multimodal(x)
            local_min = sum((x .- 0.0) .^ 2)
            global_min = sum((x .- 3.0) .^ 2) - 5.0  # global min at [3,3] with value -5
            return min(local_min, global_min)
        end

        lb = [-5.0, -5.0]
        ub = [10.0, 10.0]
        x0 = [0.0, 0.0]  # start near local minimum

        # Single-start
        single = NLoptPLEConfig(algorithm=:LN_BOBYQA, maxeval=1000)
        loss_single, _ = optimize_ple(multimodal, x0, lb, ub, single)

        # Multi-start (should find global minimum more often)
        multi = MultiStartBOBYQAConfig(n_starts=10, maxeval=1000, perturbation=0.5)
        loss_multi, _ = optimize_ple(multimodal, x0, lb, ub, multi)

        @test loss_multi <= loss_single + 0.01
    end

    # =========================================================================
    # Test 6: Hybrid optimizer configuration
    # =========================================================================
    @testset "Hybrid optimizer (global + local)" begin
        function quad2_objective(params, cps)
            return (params[1] - 3.0)^2 + (params[2] - 5.0)^2
        end

        prob = ODEChangepointPLEProblem(
            objective = quad2_objective,
            data = zeros(1, 1),
            loss_fn = (s, d) -> 0.0,
            changepoints = Int[],
            best_params = [3.0, 5.0],
            best_loss = 0.0,
            lb = [-10.0, -10.0],
            ub = [20.0, 20.0],
            param_names = ["θ₁", "θ₂"],
            n_global = 2,
            n_segment_specific = 0,
            n_obs = 1
        )

        global_opt = MultiStartBOBYQAConfig(n_starts=2, maxeval=300)
        local_opt = OptimPLEConfig(Fminbox(LBFGS()), Optim.Options(show_trace=false, iterations=100))
        hybrid = HybridOptimizerConfig(global_optimizer=global_opt, local_optimizer=local_opt)
        options = ProfileOptions(samplesize=20, polish=false, smooth_jumps=false)
        result = profile_parameter(prob, 1; options=options, optimizer=hybrid)

        @test result.best_value ≈ 3.0
        @test result.ci_lower < 3.0 < result.ci_upper
    end

    # =========================================================================
    # Test 7: ple_ci_interpolated accuracy
    # =========================================================================
    @testset "Interpolated CI" begin
        # Parabolic profile: loss = (θ - 5)² with best_loss = 0
        values = collect(range(0.0, 10.0, length=21))
        losses = [(v - 5.0)^2 for v in values]
        threshold = 0.0 + 3.84

        ci_lo, ci_hi, ident = ple_ci_interpolated(values, losses, 5.0, 0.0, 0.0, 10.0, threshold)

        # Analytical: |θ - 5| ≤ √3.84 ≈ 1.96 → CI = [3.04, 6.96]
        @test ci_lo ≈ 3.04 atol=0.1
        @test ci_hi ≈ 6.96 atol=0.1
        @test ident == true
    end

end
