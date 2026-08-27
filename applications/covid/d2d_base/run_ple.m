clear all; close all;
% Set this to your local Data2Dynamics installation.
d2d_path = fullfile(getenv('HOME'), 'Documents', 'Data2Dynamics-d2d-47d53ce');
addpath(genpath(d2d_path));
fprintf('D2D path added: %s\n', d2d_path);

arInit;
arLoadModel('covid_base');
arLoadData('covid_data');
arCompileAll;
arClearEvents;

% CVODES tolerances for the stiff, large-magnitude COVID integrator
ar.config.maxsteps = 200000;
ar.config.rtol = 1e-6;
ar.config.atol = 1e-3;
ar.config.useParallel = false;

% Load initial parameters, bounds and fix flags from converter
run('d2d_initial_params.m');
fprintf('Loaded initial parameters for %d model parameters\n', length(param_names));

% Apply parameter values and bounds
for i = 1:length(param_names)
    if fix_flags(i) == 2
        arSetPars(param_names{i}, values(i), 2, values(i), lb_p(i), ub_p(i));
    else
        arSetPars(param_names{i}, values(i), 1, values(i), lb_p(i), ub_p(i));
    end
end

% Error-model parameters (fixed; not part of structural PLE)
arSetPars('sd_cases',      -1, 2, -1, -3, 0);
arSetPars('sd_hosp',       -1, 2, -1, -3, 0);
arSetPars('sd_icu',        -1, 2, -1, -3, 0);
arSetPars('sd_death',      -1, 2, -1, -3, 0);
arSetPars('sd_vacc',       -1, 2, -1, -3, 0);

% Project parameters back into bounds
for i = 1:length(ar.p)
    ar.p(i) = max(ar.lb(i), min(ar.ub(i), ar.p(i)));
end
arCalcMerit;
fprintf('After setting parameters: chi^2 = %.4f, data chi^2 = %.4f\n', ar.chi2fit, ar.chi2);

% Short warm-start fit before PLE
ar.config.optimizer = 2;
ar.config.optim.MaxIter = 2;
ar.config.optim.TolFun = 1e-6;
ar.config.optim.TolX = 1e-6;

fprintf('\nRunning short warm-start fit (MaxIter=2)...\n');
arFit;
fprintf('Warm-start fit converged: chi^2 = %.4f, data chi^2 = %.4f\n', ar.chi2fit, ar.chi2);

% Save converged workspace
current_time = datetime('now', 'Format', 'yyyyMMdd_HHmmss');
time_str = datestr(current_time, 'yyyymmdd_HHMMSS');
Output_dir = sprintf('./Output_%s', time_str);
mkdir(Output_dir);
arPlot; arPrint;
writematrix(10.^ar.p, sprintf('%s/parameter.csv', Output_dir));
writecell(ar.pLabel, sprintf('%s/label.csv', Output_dir));
save(sprintf('%s/Workspace_converged.mat', Output_dir), 'ar');

% --- PLE with fmincon ---
arPLEInit(true, true, 2, false);
ar.ple.samplesize(:)    = 30;
ar.ple.minstepsize(:)   = 1e-2;
ar.ple.maxstepsize(:)   = 0.5;
ar.ple.allowbetteroptimum = 0;
ar.ple.optimizer        = 2;

fprintf('\nStarting PLE with fmincon...\n');
try
    ple;
    plePrint;
catch ME
    fprintf('PLE error: %s\n', ME.message);
    for k = 1:length(ME.stack)
        fprintf('  %s:%d\n', ME.stack(k).name, ME.stack(k).line);
    end
end

% Save
save(sprintf('%s/Workspace_ple.mat', Output_dir), 'ar');

fprintf('\nResults saved to: %s\n', Output_dir);

exit(0);
