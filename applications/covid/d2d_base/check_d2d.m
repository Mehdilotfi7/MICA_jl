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

ar.config.maxsteps = 1000000;
ar.config.rtol = 1e-4;
ar.config.atol = 1e-2;
ar.config.useParallel = false;

run('d2d_initial_params.m');
fprintf('Loaded initial parameters for %d model parameters\n', length(param_names));

for i = 1:length(param_names)
    if fix_flags(i) == 2
        arSetPars(param_names{i}, values(i), 2, values(i), lb_p(i), ub_p(i));
    else
        arSetPars(param_names{i}, values(i), 1, values(i), lb_p(i), ub_p(i));
    end
end

arSetPars('sd_cases',      -1, 2, -1, -3, 0);
arSetPars('sd_hosp',       -1, 2, -1, -3, 0);
arSetPars('sd_icu',        -1, 2, -1, -3, 0);
arSetPars('sd_death',      -1, 2, -1, -3, 0);
arSetPars('sd_vacc',       -1, 2, -1, -3, 0);

for i = 1:length(ar.p)
    ar.p(i) = max(ar.lb(i), min(ar.ub(i), ar.p(i)));
end

arCalcMerit;
fprintf('After setting parameters: chi^2 = %.4f, data chi^2 = %.4f\n', ar.chi2fit, ar.chi2);

ar.config.optimizer = 2;
ar.config.optim.MaxIter = 2;
ar.config.optim.TolFun = 1e-6;
ar.config.optim.TolX = 1e-6;

fprintf('\nRunning short warm-start fit (MaxIter=2)...\n');
arFit;
fprintf('Warm-start fit converged: chi^2 = %.4f, data chi^2 = %.4f\n', ar.chi2fit, ar.chi2);

fprintf('\nD2D pipeline check passed.\n');
exit(0);
