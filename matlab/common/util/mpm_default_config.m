function cfg = mpm_default_config(task)
%MPM_DEFAULT_CONFIG Default settings of the classification / regression pipelines.
%
%   cfg = mpm_default_config('classification')   prospectivity / target delineation
%   cfg = mpm_default_config('regression')       grade estimation
%
%   Run scripts only set what differs; MPM_MERGE fills in the rest.

% ---------------------------------------------------------------- data (shared)
cfg.dataset_name = '';            % label for output folders / GUI ('' = folder name)
cfg.data_dir = '';                % Python "processed" folder (layers.csv + rasters/)
cfg.layers = {};                  % evidence layers (names or wildcards); {} = all but mask/reference
cfg.exclude_layers = {};          % removed from the selection (wildcards allowed)
cfg.mask_layer = 'study_mask';    % study-area mask raster ('' = whole grid)
cfg.features = struct( ...
    'categorical', 'onehot', ...  % 'onehot' | 'numeric'
    'as_numeric', {{}}, ...       % categorical layers whose class values are physical numbers
    'missing', 'impute');         % 'impute' | 'exclude'
cfg.points_csv = '';              % deposits (classification) / grade samples (regression)
cfg.x_field = 'x';
cfg.y_field = 'y';
cfg.seed = 42;

% ---------------------------------------------------------------- train / test split (shared)
cfg.split = struct( ...
    'method', 'random', ...       % 'random' | 'block' | 'rule' | 'none'
    'test_fraction', 0.3, ...
    'block_size', 10000, ...      % m, for 'block'
    'test_rule', []);             % @(T) logical vector, for 'rule' (e.g. @(T) T.RANDOM >= 700)

% ---------------------------------------------------------------- model (shared)
cfg.params = struct();            % overrides of the method defaults (M.defaults)
cfg.tune = struct( ...
    'enable', true, ...
    'kfold', 5, ...
    'grid', []);                  % [] = method grid (M.grid); struct() = no tuning
cfg.importance = struct('enable', true, 'repeats', 5);
cfg.chunk = 200000;               % cells per prediction chunk

% ---------------------------------------------------------------- output (shared)
cfg.output_dir = '';              % '' = matlab/results/<dataset>/<task>_<method>_<timestamp>
cfg.save = true;                  % result.mat + CSV tables
cfg.write_geotiff = true;
cfg.save_figures = false;         % PNGs of every GUI tab (also available from the GUI)
cfg.show_gui = true;

switch lower(task)
    case 'classification'
        cfg.positives = struct('buffer_cells', 0);   % >0: also use cells within r cells of each
                                                     % TRAINING deposit as positives
        cfg.negatives = struct( ...
            'strategy', 'random', ...
            'ratio', 1, ...                          % negatives per positive (train and test)
            'min_distance', 2000);                   % m from any known deposit
        cfg.threshold = 0.5;                         % number | 'youden' (from training ROC); Youden for non-probability scores
        % unsupervised (anomaly-detection) methods: cells the model is fitted on, without labels
        cfg.anomaly = struct('fit_cells', 'all', ...            % 'all' (study area) | 'background' (beyond negatives.min_distance of training deposits)
            'n_fit', 10000);                         % random cells
        % positive-unlabelled methods (M.learning = 'pu'): unlabelled cells replacing the negatives
        cfg.pu = struct('n_unlabelled', 10000);      % random cells of the study area (no distance buffer)
        cfg.targets = struct('method', 'area', 'levels', [0.05 0.10 0.20]);
        % binary target grids: 1 = predicted deposit area, 0 = non-deposit (mpm_binary_targets)
        cfg.binary = struct( ...
            'enable', true, ...
            'methods', {{'pa', 'ca'}}, ...           % P-A intersection and/or C-A fractal
            'pa_deposits', 'train', ...              % deposits defining the P-A curve: 'train' | 'all'
            'ca', struct('segments', 3, 'n_thresholds', 100, 'min_points', 6), ...
            'ca_break', 'highest');                  % C-A break used: 'highest' | 'lowest' | k
        cfg.tune.metric = 'auc';
    case 'regression'
        cfg.target_field = '';                       % grade column of the point CSV
        cfg.target_label = '';                       % axis label, e.g. 'Au (g/t)'
        cfg.transform = 'log10';                     % 'log10' | 'log' | 'none' (model scale)
        cfg.nonpositive = 'drop';                    % log transforms: 'drop' | 'halfmin'
        cfg.duplicates = 'mean';                     % samples in the same cell: 'mean' | 'keep'
        cfg.cutoff = struct('value', [], 'percentile', 75);  % grade cut-off for the ROC/AUC check
        cfg.tune.metric = 'rmse';
    otherwise
        error('mpm:config', 'Unknown task "%s".', task);
end
cfg.task = lower(task);
end
