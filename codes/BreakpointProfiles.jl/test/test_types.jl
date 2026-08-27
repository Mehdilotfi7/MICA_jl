using BreakpointProfiles
using Test

@testset "ODEChangepointPLEProblem construction" begin
    data = ones(1, 10)
    lb = [0.0, 0.0]
    ub = [1.0, 1.0]

    prob = ODEChangepointPLEProblem(
        data = data,
        loss_fn = (sim, data) -> 0.0,
        changepoints = [5],
        best_params = [0.5, 0.5],
        lb = lb,
        ub = ub,
        n_global = 1,
        n_segment_specific = 1
    )

    @test prob isa ODEChangepointPLEProblem
    @test prob.changepoints == [5]
    @test prob.n_global == 1
    @test prob.n_segment_specific == 1
    @test prob.n_obs == 10
    @test length(prob) == 2
    @test prob.param_names == ["par_1", "par_2_seg1", "par_2_seg2"]

    # User-supplied parameter labels
    prob2 = ODEChangepointPLEProblem(
        data = data,
        loss_fn = (sim, data) -> 0.0,
        changepoints = [5],
        best_params = [0.5, 0.5],
        lb = lb,
        ub = ub,
        param_names = ["a", "b"],
        n_global = 1,
        n_segment_specific = 1
    )
    @test prob2.param_names == ["a", "b"]

    @test_throws AssertionError ODEChangepointPLEProblem(
        data = data,
        loss_fn = (sim, data) -> 0.0,
        changepoints = [5],
        best_params = [0.5, 0.5],
        lb = [0.0],
        ub = [1.0, 1.0],
        n_global = 1,
        n_segment_specific = 1
    )

    @test_throws AssertionError ODEChangepointPLEProblem(
        data = data,
        loss_fn = (sim, data) -> 0.0,
        changepoints = [5],
        best_params = [0.5],
        lb = lb,
        ub = ub,
        n_global = 1,
        n_segment_specific = 1
    )
end

@testset "parameter_labels" begin
    @test parameter_labels(2, 1, 1) == ["par_1", "par_2", "par_3_seg1", "par_3_seg2"]
    @test parameter_labels(0, 2, 2) == ["par_1_seg1", "par_2_seg1", "par_1_seg2", "par_2_seg2", "par_1_seg3", "par_2_seg3"]
end
