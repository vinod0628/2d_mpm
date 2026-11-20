%RUN_CNN_CLASSIFICATION  CNN on image patches: prospectivity mapping + target delineation (run this file, F5).
%
%   Each cell is classified from the w x w window of ALL evidence layers around it, so the network
%   can use spatial context (contacts, structures, gradients near the cell). Training windows come
%   from sliding-window (every window containing a training deposit) and random-drop augmentation;
%   the map is computed fully convolutionally. An ensemble of NumNets CNNs gives the uncertainty map.
%
%   Hardware: runs on any PC (CPU). With an NVIDIA GPU (Parallel Computing Toolbox + CUDA driver) it
%   uses the GPU automatically: cfg.params.ExecutionEnvironment = 'auto' | 'gpu' | 'cpu'.
%
%   Spatial leakage: windows of nearby cells overlap. CV folds keep samples whose windows can overlap
%   together; for an independent test use a spatial block split (cfg.split.method = 'block').
%
%   Data: the co-registered rasters of the shared pre-processed database (demo_data/.../processed)
%   and deposits as CSV (x, y columns) or point shapefile (demo: matlab/data/<dataset>/deposits.csv).

clear; clc;
here = fileparts(mfilename('fullpath')); if isempty(here), here = pwd; end
addpath(fileparts(here)); root = mpm_setup();

dataset = 'carlin';          % 'carlin' (35 Carlin-type Au deposits) | 'curnamona' (7 REE occurrences)
M = clf_cnn();               % <- the only method-specific line

% --- data: layers, deposits, split, sampling (see clf_demo_config.m) -------------------------
cfg = clf_demo_config(dataset, root);
% cfg.split.method = 'block'; cfg.split.block_size = 20000; % spatial hold-out (recommended for patches)

% --- model / validation (defaults: mpm_default_config('classification')) --------------------
cfg.seed = 42;
cfg.tune.enable = true;                        % k-fold CV grid search over PatchSize (AUC)
cfg.tune.kfold = 5;
% cfg.tune.grid = struct('PatchSize', [5 9 13 17]);      % window sizes (cells, odd)
% cfg.params.ExecutionEnvironment = 'gpu';               % force the NVIDIA GPU ('auto' = use if present)
% cfg.params.NumNets = 10; cfg.params.Epochs = 30;       % larger ensemble / longer training
% cfg.params.Rotate = true;                              % rotation / flip augmentation (off: orientation matters)
cfg.threshold = 0.5;                           % or 'youden'
cfg.targets.levels = [0.05 0.10 0.20];         % target classes: top 5 / 10 / 20 % of the area
cfg.show_gui = true;

res = mpm_classification_run(cfg, M);
