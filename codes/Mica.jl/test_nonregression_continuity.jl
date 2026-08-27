using Mica

function boundary_value(model_spec, target, t_start)
    adj_params = adjust_segment_params_for_continuity(model_spec, model_spec.params, target; t_start=t_start)
    manager = ModelManager(model_spec; continuity=true)
    seg_spec = segment_model(manager, adj_params, t_start, t_start + 19, nothing)
    sim = simulate_model(seg_spec)
    return sim[1, 1]
end

t_start = 11

# ETS AAA
target = 2.5
params_ets = [1.0, 0.1, 0.2, 0.3, 0.01]
spec_ets = ETSModelSpec(ets_aaa_model, params_ets, 20, 12)
y1_ets = boundary_value(spec_ets, target, t_start)
println("ETS AAA boundary value at t=$t_start: $y1_ets (target $target)")
@assert abs(y1_ets - target) < 1e-10

# ETS MMM
params_ets_mmm = [1.0, 5.0, 0.2, 0.3, 0.01]
spec_ets_mmm = ETSModelSpec(ets_mmm_model, params_ets_mmm, 20, 12)
y1_ets_mmm = boundary_value(spec_ets_mmm, target, t_start)
println("ETS MMM boundary value at t=$t_start: $y1_ets_mmm (target $target)")
@assert abs(y1_ets_mmm - target) < 1e-10

# Poisson
target = 5.0
params_pois = [0.5, 0.01, 0.0]
spec_pois = CountModelSpec(poisson_model, params_pois, 20, :poisson)
y1_pois = boundary_value(spec_pois, target, t_start)
println("Poisson boundary value at t=$t_start: $y1_pois (target $target)")
@assert abs(y1_pois - target) < 1e-10

# NegBin
params_negbin = [0.5, 0.01, 0.0]
spec_negbin = CountModelSpec(negbin_model, params_negbin, 20, :negbin)
y1_negbin = boundary_value(spec_negbin, target, t_start)
println("NegBin boundary value at t=$t_start: $y1_negbin (target $target)")
@assert abs(y1_negbin - target) < 1e-10

# INGARCH
target = 3.0
params_ingarch = [0.5, 0.2, 0.1, 0.0]
spec_ingarch = CountModelSpec(ingarch_model, params_ingarch, 20, :poisson)
y1_ingarch = boundary_value(spec_ingarch, target, t_start)
println("INGARCH boundary value at t=$t_start: $y1_ingarch (target $target)")
@assert abs(y1_ingarch - target) < 1e-10

# GARCH
target = 1.5
params_garch = [0.5, 0.2, 0.1, 0.7]
spec_garch = VolatilityModelSpec(garch_model, params_garch, 20)
adj_garch = adjust_segment_params_for_continuity(spec_garch, params_garch, target; t_start=t_start)
manager_garch = ModelManager(spec_garch; continuity=true)
seg_garch = segment_model(manager_garch, adj_garch, t_start, t_start + 19, nothing)
sim_garch = simulate_model(seg_garch)
println("GARCH mean boundary value at t=$t_start: $(sim_garch[1, 1]) (target $target)")
@assert abs(sim_garch[1, 1] - target) < 1e-10

println("\nAll non-regression boundary-continuity smoke tests passed.")
