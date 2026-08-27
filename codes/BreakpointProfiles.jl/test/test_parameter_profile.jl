using BreakpointProfiles
using Test
using CSV, DataFrames, Statistics

# Analytic piecewise-linear ODE with one changepoint.
# u(t) = u0 + r1 * t                     for t < cp
# u(t) = u0 + r1 * cp + r2 * (t - cp)    for t >= cp
function piecewise_linear(t, cp, u0, r1, r2)
    if t < cp
        return u0 + r1 * t
    else
        return u0 + r1 * cp + r2 * (t - cp)
    end
end

function simulate_piecewise_linear(params, cps; tspan=(0.0, 50.0))
    cp = isempty(cps) ? tspan[2] : Float64(cps[1])
    u0, r1, r2 = params
    times = collect(tspan[1]:tspan[2])
    vals = [piecewise_linear(t, cp, u0, r1, r2) for t in times]
    return reshape(vals, 1, :)
end

mse_loss(sim, data) = mean((sim .- data).^2)

function make_test_problem(; tspan=(0.0, 50.0), true_cp=20.0, noise=0.0)
    true_params = [5.0, 0.2, 0.5]
    data = simulate_piecewise_linear(true_params, [true_cp]; tspan=tspan)
    if noise > 0
        data = data .+ noise .* randn(size(data))
    end

    lb = [0.0, 0.0, 0.0]
    ub = [10.0, 1.0, 1.0]
    cps = [Int(true_cp)]

    objective = (params, cps) -> mse_loss(simulate_piecewise_linear(params, cps; tspan=tspan), data)

    prob = ODEChangepointPLEProblem(
        objective = objective,
        tspan = tspan,
        data = data,
        loss_fn = mse_loss,
        changepoints = cps,
        best_params = copy(true_params),
        lb = lb,
        ub = ub,
        param_names = ["u0", "r1", "r2"],
        n_global = 1,
        n_segment_specific = 1
    )
    return prob, true_params, cps
end

@testset "profile_parameter adaptive" begin
    prob, true_params, _ = make_test_problem()
    optimizer = NLoptPLEConfig(algorithm=:LN_BOBYQA, xtol_rel=1e-5, maxeval=2000)
    options = ProfileOptions(samplesize=15, polish=false, smooth_jumps=false)
    prof = profile_parameter(prob, 1; options=options, optimizer=optimizer)

    @test prof isa ProfileResult
    @test prof.parameter == "u0"
    @test prof.index == 1
    @test prof.best_value ≈ true_params[1] atol=1e-2
    @test prof.best_loss < 1e-3
    @test prof.ci_lower < prof.best_value < prof.ci_upper
    @test length(prof.values) == length(prof.losses)
    @test prof.values[1] == minimum(prof.values)
    @test prof.values[end] == maximum(prof.values)
end

@testset "profile_parameter fixed" begin
    prob, true_params, _ = make_test_problem()
    optimizer = NLoptPLEConfig(algorithm=:LN_BOBYQA, xtol_rel=1e-5, maxeval=2000)
    prof = profile_parameter(prob, 2; method=:fixed, n_points=10, optimizer=optimizer)

    @test prof.parameter == "r1"
    @test prof.index == 2
    @test prof.best_value ≈ true_params[2] atol=1e-2
    @test prof.best_loss < 1e-3
    @test prof.ci_lower < prof.best_value < prof.ci_upper
end

@testset "ple_ci and is_identifiable" begin
    values = collect(0.0:0.5:3.0)
    losses = [10.0, 5.0, 1.0, 0.0, 1.0, 5.0, 10.0]
    lo, hi, id = ple_ci(values, losses, 1.5, 0.0, 0.0, 3.0, 3.84)
    @test lo == 1.0
    @test hi == 2.0
    @test id

    # Hit lower bound -> non-identifiable
    losses2 = [0.0, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    lo2, hi2, id2 = ple_ci(values, losses2, 0.0, 0.0, 0.0, 3.0, 3.84)
    @test lo2 == 0.0
    @test !id2

    prof = ProfileResult("x", 1, 1.5, 0.0, lo, hi, id, 3.84, values, losses, 0, 0.0, Float64[])
    @test is_identifiable(prof) == id
end
