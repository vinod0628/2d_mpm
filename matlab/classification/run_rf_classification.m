%RUN_RF_CLASSIFICATION  Random-forest prospectivity mapping + target delineation (run this file, F5).
%
%   Classification = delineation of exploration targets: the model learns deposit vs. background
%   cells from the evidence layers and returns a prospectivity map (0..1) that is cut into target
%   classes. Results open in a GUI (training / test ROC + AUC, success-rate, P-A, maps, ...).
%
%   Data: the co-registered rasters of the shared pre-processed database (demo_data/.../processed)
%   and deposits as CSV (x, y columns) or point shapefile (demo: matlab/data/<dataset>/deposits.csv).
%
%   TEMPLATE: to run another method copy this file and change only  M = clf_rf();
%   (see classification/methods/clf_template.m for writing a new method).

clear; clc;
here = fileparts(mfilename('fullpath')); if isempty(here), here = pwd; end
addpath(fileparts(here)); root = mpm_setup();

dataset = 'carlin';          % 'carlin' (35 Carlin-type Au deposits) | 'curnamona' (7 REE occurrences)
M = clf_rf();                % <- the only method-specific line

% --- data: layers, deposits, split, sampling (see clf_demo_config.m) -------------------------
cfg = clf_demo_config(dataset, root);
% cfg.layers = {'geology', 'aeromag', 'dist_faults'};      % override the evidence layers
% cfg.split.method = 'block'; cfg.split.block_size = 20000; % spatial hold-out
% cfg.negatives.ratio = 2;                                  % more background samples

% --- model / validation (defaults: mpm_default_config('classification')) --------------------
cfg.seed = 42;
cfg.tune.enable = true;                        % k-fold CV grid search on M.grid (metric: AUC)
cfg.tune.kfold = 5;
% cfg.tune.grid = struct('MinLeafSize', [1 5], 'NumPredictorsToSample', [1 2]);  % custom grid
% cfg.params.NumTrees = 500;                                                   % fixed params
cfg.threshold = 0.5;                           % or 'youden'
cfg.targets.levels = [0.05 0.10 0.20];         % target classes: top 5 / 10 / 20 % of the area
cfg.show_gui = true;

res = mpm_classification_run(cfg, M);
