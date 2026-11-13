using BreakpointProfiles
using Test
using CSV, DataFrames

function synthetic_profile(name, idx, best_value, best_loss, ci_lower, ci_upper, identifiable, values, losses)
    return ProfileResult(
        name, idx, best_value, best_loss,
        ci_lower, ci_upper, identifiable,
        best_loss + 3.8414588206941285,
        values, losses,
        0, best_loss, Float64[]  # n_failed, best_found_loss, best_found_params
    )
end

@testset "ple_summary and write_profiles" begin
    prof1 = synthetic_profile(
        "a", 1, 1.0, 0.0, 0.8, 1.2, true,
        [0.5, 0.8, 1.0, 1.2, 1.5],
        [5.0, 1.0, 0.0, 1.0, 5.0]
    )
    prof2 = synthetic_profile(
        "b", 2, 2.0, 0.0, 0.0, 3.0, false,
        [0.0, 1.0, 2.0, 3.0],
        [10.0, 2.0, 0.0, 2.0]
    )

    df = ple_summary([prof1, prof2])
    @test df isa DataFrame
    @test nrow(df) == 2
    @test df.parameter == ["a", "b"]
    @test df.index == [1, 2]
    @test df.identifiable == [true, false]
    @test df.relative_width[1] ≈ (1.2 - 0.8) / 1.0
    @test df.relative_width[2] ≈ (3.0 - 0.0) / 2.0

    mktempdir() do tmp
        path = joinpath(tmp, "profiles.csv")
        write_profiles(path, [prof1, prof2])
        @test isfile(path)
        df_read = CSV.read(path, DataFrame)
        @test names(df_read) == ["parameter", "index", "best_value", "best_loss", "ci_lower", "ci_upper", "identifiable", "threshold", "n_failed", "best_found_loss"]
        @test nrow(df_read) == 2
    end
end

@testset "to_profile_dataframe" begin
    prof = synthetic_profile(
        "x", 1, 1.0, 0.5, 0.8, 1.2, true,
        [0.8, 1.0, 1.2],
        [4.0, 0.5, 4.0]
    )
    df = to_profile_dataframe([prof])
    @test nrow(df) == 3
    @test df.delta_loss ≈ df.loss .- prof.best_loss
end

@testset "write_summary and print_summary_table" begin
    prof = synthetic_profile(
        "x", 1, 1.0, 0.0, 0.8, 1.2, true,
        [0.8, 1.0, 1.2],
        [1.0, 0.0, 1.0]
    )
    summary_df = ple_summary([prof])
    mktempdir() do tmp
        path = joinpath(tmp, "summary.csv")
        write_summary(path, summary_df)
        @test isfile(path)
    end
    # Just check that printing does not error
    @test_nowarn print_summary_table(summary_df)
end

@testset "plot_profiles requires Plots" begin
    prof = synthetic_profile(
        "x", 1, 1.0, 0.0, 0.8, 1.2, true,
        [0.8, 1.0, 1.2],
        [1.0, 0.0, 1.0]
    )
    @test_throws ErrorException plot_profiles([prof])
end
