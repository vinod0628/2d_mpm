%RUN_ELM_REGRESSION  Extreme learning machine (ELM ensemble) grade estimation from orebody samples + evidence layers (run this file, F5).
%
%   Regression = grade estimation: the model learns grade at known sample locations from the
%   evidence layers and predicts a grade map. Results open in a GUI (observed vs predicted for
%   training / CV / test, residuals, grade map, cut-off ROC/AUC, importance, tuning).
%
%   YOUR OWN OREBODY DATA: a CSV with columns x, y (in the raster CRS) and a grade column, e.g.
%   drillhole composites or channel samples - set after reg_demo_config below:
%       cfg.points_csv   = 'D:\data\orebody_composites.csv';
%       cfg.target_field = 'Au_gpt';   cfg.target_label = 'Au (g/t)';
%
%   DEMO DATA: the demo sets contain no orebody grades, so surface geochemistry stands in for
%   grade: Carlin Th ppm in NURE stream sediments (linked to the airborne Th survey; see
%   reg_demo_config.m) or Curnamona La ppm in rock samples (weakly linked).
%   Geochemical evidence layers interpolated from the SAME samples are excluded to avoid leakage.
%
%   TEMPLATE: for another method copy this file and change only  M = reg_elm();
%   (see regression/methods/reg_template.m for writing a new method).

clear; clc;
here = fileparts(mfilename('fullpath')); if isempty(here), here = pwd; end
addpath(fileparts(here)); root = mpm_setup();

dataset = 'carlin';          % 'carlin' (Th, NURE stream sediments) | 'curnamona' (La, rock samples)
M = reg_elm();               % <- the only method-specific line

% --- data: layers, samples, grade column (see reg_demo_config.m) -----------------------------
cfg = reg_demo_config(dataset, root);

% --- model / validation settings (defaults: mpm_default_config('regression')) ----------------
cfg.transform = 'log10';                % grades are usually log-normal; 'none' for raw values
cfg.duplicates = 'mean';                % samples in the same cell are averaged
cfg.split.method = 'random';            % 'block' = spatial hold-out (less optimistic):
cfg.split.test_fraction = 0.3;          %   cfg.split.block_size = 5000;
cfg.cutoff.percentile = 75;             % ROC/AUC of "grade >= P75" ranked by predicted grade
% cfg.cutoff.value = 100;               %   ... or a fixed cut-off grade (original units)
cfg.tune.enable = true;                 % k-fold CV grid search on M.grid (metric: RMSE)
cfg.tune.kfold = 5;
cfg.seed = 42;
cfg.show_gui = true;

res = mpm_regression_run(cfg, M);
