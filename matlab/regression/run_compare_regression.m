%RUN_COMPARE_REGRESSION  Run several grade-estimation methods on the same samples / split and compare.
%
%   Same cfg and seed for every method -> identical train/test samples and CV folds.
%   Add new methods to the list as they are implemented (e.g. @reg_gpr, @reg_gbm).

clear; clc;
here = fileparts(mfilename('fullpath')); if isempty(here), here = pwd; end
addpath(fileparts(here)); root = mpm_setup();

dataset = 'carlin';                         % 'carlin' (Th) | 'curnamona' (La)
methods = {@reg_svm, @reg_rf, @reg_rvm, @reg_mlp, @reg_xgboost, @reg_lightgbm, @reg_elm};   % <- methods to compare

cfg = reg_demo_config(dataset, root);
cfg.seed = 42;
cfg.show_gui = false;

results = cell(1, numel(methods));
for i = 1:numel(methods)
    results{i} = mpm_regression_run(cfg, methods{i}());
end
if usejava('desktop')
    mpm_compare_viewer(results);
else
    mpm_compare_viewer(results, 'Visible', 'off', 'Close', true, ...
        'ExportDir', fullfile(fileparts(results{end}.output_dir), 'comparison'));
end
