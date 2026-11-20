%RUN_BN_CLASSIFICATION  Bayesian network classifiers (naive / TAN / selective naive Bayes): prospectivity mapping (run this file, F5).
%
%   One network node per evidence layer: categorical layers keep their classes, continuous layers are
%   discretised on the training data (Fayyad & Irani MDL or equal-frequency). The structure is tuned by
%   CV: naive Bayes (= weights of evidence for binary evidence), tree-augmented naive Bayes (Friedman et
%   al. 1997) or selective naive Bayes (only conditionally independent predictors; Porwal et al. 2006).
%   Output: posterior P(deposit | evidence); uncertainty map = Dirichlet-posterior parameter uncertainty.
%
%   Data: the co-registered rasters of the shared pre-processed database (demo_data/.../processed)
%   and deposits as CSV (x, y columns) or point shapefile (demo: matlab/data/<dataset>/deposits.csv).

clear; clc;
here = fileparts(mfilename('fullpath')); if isempty(here), here = pwd; end
addpath(fileparts(here)); root = mpm_setup();

dataset = 'carlin';          % 'carlin' (35 Carlin-type Au deposits) | 'curnamona' (7 REE occurrences)
M = clf_bn();                % <- the only method-specific line

% --- data: layers, deposits, split, sampling (see clf_demo_config.m) -------------------------
cfg = clf_demo_config(dataset, root);

% --- model / validation (defaults: mpm_default_config('classification')) --------------------
cfg.seed = 42;
cfg.tune.enable = true;                        % CV over Structure x Discretization (AUC)
cfg.tune.kfold = 5;
% cfg.params.Structure = 'tan'; cfg.tune.grid = struct();   % fixed structure, no tuning
% cfg.params.NumBins = 5; cfg.params.N0 = 5;                 % equal-frequency classes, Dirichlet smoothing
cfg.threshold = 0.5;                           % or 'youden'
cfg.targets.levels = [0.05 0.10 0.20];         % target classes: top 5 / 10 / 20 % of the area
cfg.show_gui = true;

res = mpm_classification_run(cfg, M);
