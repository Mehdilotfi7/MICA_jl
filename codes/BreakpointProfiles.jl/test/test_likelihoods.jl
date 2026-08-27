using BreakpointProfiles
using Test
using Statistics

@testset "Built-in likelihoods" begin
    sim = exp.(0.1:0.1:1.0)
    data = sim .+ 0.01

    gauss = GaussianLogLikelihood(0.5)
    laplace = LaplaceLogLikelihood(0.5)
    @test gauss(sim, data) > 0
    @test laplace(sim, data) > 0
    @test gauss(sim, sim) < gauss(sim, data)
    @test laplace(sim, sim) < laplace(sim, data)

    custom = CustomLoss((s, d) -> mean((s .- d).^2))
    @test custom(sim, data) ≈ mean((sim .- data).^2)

    @test chi2_threshold() ≈ 3.8414588206941285
end

@testset "log_transform" begin
    v = [0.0, 0.5, 1.0, -1.0]
    @test all(log_transform(v; threshold=1.0) .== 0.0)
    @test log_transform([10.0, 100.0]; threshold=1.0) ≈ [log(10.0), log(100.0)]
end

@testset "evaluate_loss via objective / simulator / ode_function" begin
    data = reshape(collect(0.0:0.1:1.0), 1, :)

    # objective path
    prob_obj = ODEChangepointPLEProblem(
        objective = (params, cps) -> params[1]^2,
        data = data,
        loss_fn = (sim, data) -> 0.0,
        changepoints = Int[],
        best_params = [2.0],
        lb = [0.0],
        ub = [10.0],
        n_global = 1,
        n_segment_specific = 0
    )
    @test BreakpointProfiles.evaluate_loss(prob_obj, [2.0]) ≈ 4.0

    # simulator + loss_fn path
    prob_sim = ODEChangepointPLEProblem(
        simulator = (params, cps) -> data,
        data = data,
        loss_fn = (sim, data) -> sum((sim .- data).^2),
        changepoints = Int[],
        best_params = [1.0],
        lb = [0.0],
        ub = [10.0],
        n_global = 1,
        n_segment_specific = 0
    )
    @test BreakpointProfiles.evaluate_loss(prob_sim, [1.0]) ≈ 0.0

    # generic ODE solver path (u' = r, constant rate, no changepoint)
    function const_ode!(du, u, p, t)
        du[1] = p[1]
    end
    prob_ode = ODEChangepointPLEProblem(
        ode_function = const_ode!,
        u0 = [0.0],
        tspan = (0.0, 10.0),
        data = data,
        loss_fn = (sim, data) -> sum((sim .- data).^2),
        changepoints = Int[],
        best_params = [0.1],
        lb = [0.0],
        ub = [10.0],
        n_global = 1,
        n_segment_specific = 0
    )
    sim = BreakpointProfiles.solve_segments(prob_ode, [0.1])
    @test size(sim, 2) == 11
    @test sim[1, 1] ≈ 0.0
    @test sim[1, end] ≈ 1.0
    @test BreakpointProfiles.evaluate_loss(prob_ode, [0.1]) < 1.0
end
