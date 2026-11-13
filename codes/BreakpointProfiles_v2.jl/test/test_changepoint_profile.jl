using BreakpointProfiles
using Test
using CSV, DataFrames, Statistics

# Use the same analytic piecewise-linear model as the parameter-profile tests.
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

function make_cp_test_problem(; true_cp=20.0, tspan=(0.0, 50.0))
    true_params = [5.0, 0.2, 0.5]
    data = simulate_piecewise_linear(true_params, [true_cp]; tspan=tspan)

    objective = (params, cps) -> mse_loss(simulate_piecewise_linear(params, cps; tspan=tspan), data)

    prob = ODEChangepointPLEProblem(
        objective = objective,
        tspan = tspan,
        data = data,
        loss_fn = mse_loss,
        changepoints = [Int(true_cp)],
        best_params = copy(true_params),
        best_loss = 0.0,
        lb = [0.0, 0.0, 0.0],
        ub = [10.0, 1.0, 1.0],
        param_names = ["u0", "r1", "r2"],
        n_global = 1,
        n_segment_specific = 1
    )
    return prob, true_params
end

@testset "profile_changepoint" begin
    prob, true_params = make_cp_test_problem(true_cp=25.0)
    optimizer = NLoptPLEConfig(algorithm=:LN_BOBYQA, xtol_rel=1e-5, maxeval=2000)
    cp_prof = profile_changepoint(prob, 1; window=5, optimizer=optimizer)

    @test cp_prof isa CPProfileResult
    @test cp_prof.original_cp == 25
    @test cp_prof.window == 5
    @test 25 in cp_prof.candidate_cps
    @test length(cp_prof.candidate_cps) == length(cp_prof.losses)
    @test cp_prof.best_loss < 1e-3

    # The loss at the true changepoint should be minimal
    pos = findfirst(==(25), cp_prof.candidate_cps)
    @test pos !== nothing
    @test cp_prof.losses[pos] ≈ cp_prof.best_loss atol=1e-3
end

@testset "cp_summary and write_cp_profiles" begin
    prob, _ = make_cp_test_problem(true_cp=25.0)
    optimizer = NLoptPLEConfig(algorithm=:LN_BOBYQA, xtol_rel=1e-5, maxeval=2000)
    cp_prof = profile_changepoint(prob, 1; window=5, optimizer=optimizer)

    threshold = 3.84
    df = cp_summary([cp_prof], threshold)
    @test df isa DataFrame
    @test nrow(df) == 1
    @test df.original_cp[1] == 25
    @test df.ci_lower[1] <= df.original_cp[1] <= df.ci_upper[1]

    mktempdir() do tmp
        path = joinpath(tmp, "cp_profile.csv")
        df_long = write_cp_profiles(path, [cp_prof])
        @test isfile(path)
        @test df_long isa DataFrame
        @test nrow(df_long) == length(cp_prof.candidate_cps)
        df_read = CSV.read(path, DataFrame)
        @test names(df_read) == ["cp_index", "original_cp", "candidate_cp", "loss", "delta_loss"]
    end
end
