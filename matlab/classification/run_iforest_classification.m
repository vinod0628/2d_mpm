%RUN_IFOREST_CLASSIFICATION  Isolation forest anomaly-detection prospectivity mapping (run this file, F5).
%
%   UNSUPERVISED: the model is fitted on random cells of the whole study area WITHOUT deposit
%   labels - the evidence signature of mineralised ground is sought as a multivariate anomaly.
%   Known deposits are used only to choose the hyper-parameters (AUC of training deposits vs.
%   training background), the threshold (Youden index of the training ROC) and for validation;
%   the test deposits stay independent. Same outputs and GUI as the supervised methods
%   (ROC/AUC, success-rate, P-A, C-A, binary targets, ...); the map shows the anomaly score 2^(-E(h)/c(psi)).
%
%   Data: the co-registered rasters of the shared pre-processed database (demo_data/.../processed)
%   and deposits as CSV (x, y columns) or point shapefile (demo: matlab/data/<dataset>/deposits.csv).

clear; clc;
here = fileparts(mfilename('fullpath')); if isempty(here), here = pwd; end
addpath(fileparts(here)); root = mpm_setup();

dataset = 'carlin';          % 'carlin' (35 Carlin-type Au deposits) | 'curnamona' (7 REE occurrences)
M = clf_iforest();            % <- the only method-specific line

% --- data: layers, deposits, split (see clf_demo_config.m) ----------------------------------
cfg = clf_demo_config(dataset, root);
% cfg.layers = {'geology', 'aeromag', 'dist_faults'};      % override the evidence layers
% cfg.split.method = 'block'; cfg.split.block_size = 20000; % spatial hold-out

% --- unsupervised fit ------------------------------------------------------------------------
cfg.anomaly.fit_cells = 'all';                 % 'all' = whole study area (published practice) |
                                               % 'background' = away from the training deposits
cfg.anomaly.n_fit = 10000;                     % random cells the model is fitted on

% --- model / validation (defaults: mpm_default_config('classification')) --------------------
cfg.seed = 42;
cfg.tune.enable = true;                        % grid search on M.grid (AUC on training samples)
% cfg.tune.grid = struct('SampleSize', [128 256]);                         % custom grid
% cfg.params.NumTrees = 200; cfg.params.HeightLimit = 'full';             % fixed params
cfg.threshold = 'youden';                      % anomaly scores are not probabilities
cfg.targets.levels = [0.05 0.10 0.20];         % target classes: top 5 / 10 / 20 % of the area
cfg.show_gui = true;

res = mpm_classification_run(cfg, M);
