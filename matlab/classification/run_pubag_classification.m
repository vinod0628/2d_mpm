%RUN_PUBAG_CLASSIFICATION  Positive-unlabelled (PU) bagging prospectivity mapping + target delineation (run this file, F5).
%
%   PU learning: only the deposits are labelled. Instead of "negative" background cells (which may
%   hide undiscovered deposits), the model learns the deposits against random UNLABELLED cells of
%   the whole study area (cfg.pu.n_unlabelled, no distance buffer). Each of NumBags classifiers sees
%   all deposits and a small random subsample of the unlabelled cells; their outputs are averaged
%   (Mordelet & Vert 2014). Test deposits vs. test background validate as for the other methods.
%   The score ranks prospectivity but is not a probability -> the threshold is the Youden index.
%
%   Data: the co-registered rasters of the shared pre-processed database (demo_data/.../processed)
%   and deposits as CSV (x, y columns) or point shapefile (demo: matlab/data/<dataset>/deposits.csv).

clear; clc;
here = fileparts(mfilename('fullpath')); if isempty(here), here = pwd; end
addpath(fileparts(here)); root = mpm_setup();

dataset = 'carlin';          % 'carlin' (35 Carlin-type Au deposits) | 'curnamona' (7 REE occurrences)
M = clf_pubag();             % <- the only method-specific line

% --- data: layers, deposits, split, sampling (see clf_demo_config.m) -------------------------
cfg = clf_demo_config(dataset, root);
cfg.pu.n_unlabelled = 10000;                   % random unlabelled cells of the study area
% cfg.positives.buffer_cells = 1;               % more positives around few deposits

% --- model / validation (defaults: mpm_default_config('classification')) --------------------
cfg.seed = 42;
cfg.tune.enable = true;                        % k-fold CV grid search (AUC positives vs. unlabelled)
cfg.tune.kfold = 5;
% base learner: biased SVM (paper, default) or classification tree:
% cfg.params.BaseLearner = 'tree';
% cfg.tune.grid = struct('MinLeafSize', [1 3 5], 'SubsampleRatio', [1 3]);
% cfg.params.NumBags = 200;                     % more bags (paper: little gain beyond ~100)
cfg.threshold = 'youden';                      % PU scores are not probabilities
cfg.targets.levels = [0.05 0.10 0.20];         % target classes: top 5 / 10 / 20 % of the area
cfg.show_gui = true;

res = mpm_classification_run(cfg, M);
