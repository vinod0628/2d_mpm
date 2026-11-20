function res = mpm_classification_run(cfg, M)
%MPM_CLASSIFICATION_RUN Prospectivity mapping / target delineation with any classifier.
%
%   res = mpm_classification_run(cfg, M)
%
%   cfg  settings (see MPM_DEFAULT_CONFIG('classification'); only differences are needed)
%   M    method struct, e.g. clf_svm(), clf_rf()  (contract: classification/methods/clf_template.m)
%
%   Pipeline (identical for every method)
%     1. load the co-registered evidence rasters (Python pre-processing) and build features
%     2. deposits -> cells; split deposits into training / test (random | spatial block | rule)
%     3. optional positive buffer around training deposits; random background negatives beyond
%        cfg.negatives.min_distance from every known deposit (separate sets for train and test)
%     4. z-score with training statistics; k-fold CV grid search (AUC); final model
%     5. prospectivity map for every valid cell
%     6. validation: sample ROC/AUC + confusion metrics (train / test), map ROC/AUC of deposit cells
%        vs. all background cells, success- and prediction-rate curves, P-A plot
%     7. target delineation (area classes), permutation importance per evidence layer
%     8. save result.mat, GeoTIFFs, CSV tables; open the results GUI
%
%   res is a struct consumed by MPM_RESULTS_VIEWER (res = load(...).res also works).

cfg = mpm_merge(mpm_default_config('classification'), cfg);
if ~strcmp(M.task, 'classification'), error('mpm:run', '%s is not a classification method.', M.name); end
cfg.data_dir = mpm_fullpath(cfg.data_dir); cfg.points_csv = mpm_fullpath(cfg.points_csv);
log = mpm_Logger();
rs = RandStream('mt19937ar', 'Seed', cfg.seed);
rng(cfg.seed, 'twister');                 % global stream: TreeBagger, fitPosterior, ...
if isempty(cfg.dataset_name), cfg.dataset_name = mpm_dataset_label(cfg.data_dir); end
log.add('%s | %s | classification', M.name, cfg.dataset_name);

% ------------------------------------------------------------------ 1. evidence + features
S = mpm_load_stack(cfg.data_dir, cfg.layers, cfg.exclude_layers, cfg.mask_layer);
F = mpm_build_features(S, cfg.features);
log.add('%d evidence layers -> %d features, %d valid cells (%dx%d grid, %.0f m)', numel(S.names), ...
    numel(F.names), numel(F.idx), S.R.RasterSize(1), S.R.RasterSize(2), S.R.CellExtentInWorldX);

% ------------------------------------------------------------------ 2. deposits + split
P = mpm_read_points(cfg.points_csv, cfg.x_field, cfg.y_field);
[lin, ~, ~, inside] = mpm_points_to_cells(S.R, P.x, P.y, F.valid);
P.inside = inside; P.cell = lin;
nOut = sum(~inside);
D = P(inside, :);
[~, keep] = unique(D.cell, 'stable');                    % one deposit per cell
nDup = height(D) - numel(keep);
D = D(keep, :);
if height(D) < 3, error('mpm:run', 'Only %d deposits fall in valid cells.', height(D)); end
D.test = mpm_split_points(D, cfg.split, rs);
log.add('%d deposits in %d cells (%d outside study area, %d sharing a cell) -> %d train / %d test', ...
    height(P), height(D), nOut, nDup, sum(~D.test), sum(D.test));

% ------------------------------------------------------------------ 3. samples
trDep = D.cell(~D.test); teDep = D.cell(D.test);
[posTr, gTr] = mpm_buffer_cells(S.R, trDep, cfg.positives.buffer_cells, F.valid);
if cfg.positives.buffer_cells > 0
    k = ~ismember(posTr, teDep); posTr = posTr(k); gTr = gTr(k);
end
pu = isfield(M, 'learning') && strcmpi(M.learning, 'pu');
nNegTr = round(cfg.negatives.ratio * numel(posTr));
nNegTe = round(cfg.negatives.ratio * numel(teDep));
negOpts = cfg.negatives;                                  % never sample inside a positive buffer
negOpts.min_distance = max(negOpts.min_distance, (cfg.positives.buffer_cells + 0.5) * S.R.CellExtentInWorldX);
[negTr, negTe] = mpm_sample_negatives(F, S.R, D.cell, nNegTr, nNegTe, negOpts, rs);
if pu
    % PU: training negatives were drawn as above only so that the test background is identical to
    % that of the other methods; they are replaced here.
    % positive-unlabelled learning: the "negatives" are UNLABELLED random cells of the whole study
    % area (no distance buffer: an unlabelled cell may be an undiscovered deposit). Excluded: the
    % training positives (labelled) and the test background (kept independent). Test deposits are
    % not excluded - in a PU setting they are unknown.
    negTr = pu_cells(F, posTr, negTe, cfg.pu.n_unlabelled, rs);
    log.add('PU learning: %d positive (buffer %d cells) + %d UNLABELLED random cells of the study area; test: %d + %d background', ...
        numel(posTr), cfg.positives.buffer_cells, numel(negTr), numel(teDep), numel(negTe));
else
    log.add('training samples: %d positive (buffer %d cells) + %d negative; test: %d + %d', ...
        numel(posTr), cfg.positives.buffer_cells, numel(negTr), numel(teDep), numel(negTe));
end

linTr = [posTr; negTr]; yTr = [true(numel(posTr), 1); false(numel(negTr), 1)];
grpTr = [gTr; max([gTr; 0]) + (1:numel(negTr))'];
linTe = [teDep; negTe]; yTe = [true(numel(teDep), 1); false(numel(negTe), 1)];
Xtr = F.X(F.pos(linTr), :); Xte = F.X(F.pos(linTe), :);
unsup = isfield(M, 'learning') && strcmpi(M.learning, 'unsupervised');
linFit = [];
if unsup                       % anomaly detection: fitted on unlabelled cells of the study area
    linFit = fit_cells(F, S.R, trDep, cfg.anomaly, negOpts.min_distance, rs);
    sc = mpm_scaler(F.X(F.pos(linFit), :));
    Zfit = sc.apply(F.X(F.pos(linFit), :));
    log.add('unsupervised: model fitted on %d random cells (%s) without labels; deposits only for tuning, threshold and validation', ...
        numel(linFit), cfg.anomaly.fit_cells);
else
    sc = mpm_scaler(Xtr);
end
Ztr = sc.apply(Xtr); Zte = sc.apply(Xte);
mpm_feature_info(struct('names', {F.names}, 'source', F.source, 'layers', {F.layers}, ...
    'onehot', contains(string(F.names), '='), 'mu', sc.mu, 'sigma', sc.sigma));   % layer membership (e.g. clf_bn)
patch =isfield(M, 'input') && strcmpi(M.input, 'patch');
if patch                       % patch methods (CNN): z-scored evidence cube; samples = cell indices
    if unsup || pu, error('mpm:run', 'Patch input is supported for supervised methods only.'); end
    src = mpm_patch_source(F, sc, cfg.chunk);
    mpm_patch_source.active(src);
    Ztr = linTr; Zte = linTe;
    log.add('patch input: %d x %d x %d evidence cube (z-scored, 0 outside the study area)', src.H, src.W, src.nFeat);
end

% ------------------------------------------------------------------ 4. tuning + final model
params = mpm_merge(M.defaults, cfg.params);
grid = cfg.tune.grid;
if isempty(grid), grid = M.grid; end
tuning = table();
if isfield(M, 'spatial_reach') && ~isempty(M.spatial_reach)
    % samples whose windows can overlap must share a CV fold (spatial leakage of patches)
    reach = M.spatial_reach(params, grid);
    grpTr = spatial_groups(linTr, size(F.valid), reach);
    log.add('CV groups: samples within %d cells (Chebyshev) linked -> %d groups (%d with deposits)', ...
        reach, numel(unique(grpTr)), numel(unique(grpTr(yTr))));
end
if cfg.tune.enable && ~isempty(fieldnames(grid)) && unsup
    log.add('tuning (AUC of training deposits vs. training background; fit without labels) over %s', mpm_grid_text(grid));
    [params, tuning] = mpm_tune_unsupervised(M, Zfit, Ztr, yTr, params, grid, log);
    log.add('best: %s  (training AUC = %.4f)', mpm_param_text(params, fieldnames(grid)), tuning.sel_auc(tuning.best));
elseif cfg.tune.enable && ~isempty(fieldnames(grid))
    log.add('CV tuning (%d-fold, %s) over %s', cfg.tune.kfold, cfg.tune.metric, mpm_grid_text(grid));
    if pu, log.add('  PU: CV AUC = positives vs. UNLABELLED (linear in the positive-vs-negative AUC)'); end
    [params, tuning] = mpm_tune_cv(M, Ztr, yTr, params, grid, cfg.tune.kfold, cfg.tune.metric, rs, grpTr, log);
    b = tuning(tuning.best, :);
    log.add('best: %s  (CV %s = %.4f)', mpm_param_text(params, fieldnames(grid)), cfg.tune.metric, ...
        b.(['cv_' cfg.tune.metric]));
end
if unsup
    mdl = M.train(Zfit, [], params, struct('final', true, 'task', 'classification', 'groups', []));
else
    mdl = M.train(Ztr, yTr, params, struct('final', true, 'task', 'classification', 'groups', grpTr));
end
modelText = '';
if isfield(M, 'describe') && ~isempty(M.describe), modelText = M.describe(mdl, params); end
log.add('final model: %s', modelText);
relTr = false(numel(linTr), 1);                           % sparse models: samples the model kept
if isfield(M, 'relevant_samples') && ~isempty(M.relevant_samples)
    relTr(M.relevant_samples(mdl)) = true;
    log.add('relevant training samples: %d deposit + %d background', sum(relTr & yTr), sum(relTr & ~yTr));
end

% ------------------------------------------------------------------ 5. map
log.add('predicting %d cells ...', numel(F.idx));
[scoreMap, uncMap] = mpm_predict_map(M, mdl, F, sc, cfg.chunk, log);

% ------------------------------------------------------------------ 6. validation
sTr = M.predict(mdl, Ztr); sTe = M.predict(mdl, Zte);
thr = cfg.threshold;
if isnumeric(thr) && isfield(M, 'score_is_probability') && ~M.score_is_probability
    log.add('threshold: %s scores are not probabilities -> Youden index of the training ROC', M.name);
    thr = 'youden';
end
if ischar(thr) || isstring(thr)
    C = mpm_roc(yTr, sTr); thr = C.youden_thr;
end
trName = 'training samples'; if pu, trName = 'training samples (positive vs. unlabelled)'; end
ev.samples.train = mpm_eval_binary(trName, yTr, sTr, thr);
if ~isempty(linTe), ev.samples.test = mpm_eval_binary('test samples', yTe, sTe, thr); end
allS = scoreMap(F.idx);
bg = true(size(F.idx)); bg(F.pos(D.cell)) = false;
ev.map.train = mpm_eval_map('training deposits', allS, scoreMap(trDep), allS(bg));
if ~isempty(teDep), ev.map.test = mpm_eval_map('test deposits', allS, scoreMap(teDep), allS(bg)); end
msg = sprintf('AUC  samples: train %.3f', ev.samples.train.auc);
if isfield(ev.samples, 'test'), msg = [msg sprintf(', test %.3f', ev.samples.test.auc)]; end
msg = [msg sprintf('  |  map: train %.3f', ev.map.train.auc)];
if isfield(ev.map, 'test'), msg = [msg sprintf(', test %.3f', ev.map.test.auc)]; end
log.add('%s', msg);

% ------------------------------------------------------------------ 7. targets + importance
[targetMap, targetTable] = mpm_delineate_targets(scoreMap, F.valid, trDep, teDep, cfg.targets);
binary = [];
if cfg.binary.enable                                       % binary target grids (P-A / C-A)
    binary = mpm_binary_targets(scoreMap, F.valid, trDep, teDep, mpm_merge(rmfield(cfg.binary, 'enable'), ...
        struct('cell_area_km2', S.R.CellExtentInWorldX * S.R.CellExtentInWorldY / 1e6)));
    binary.points = D(:, {'x', 'y', 'test'});
    for i = 1:numel(binary.messages), log.add('%s', binary.messages{i}); end
    for i = 1:height(binary.table)
        b = binary.table(i, :);
        log.add('binary %s: threshold %.4g -> %.1f %% of area, captures %.0f %% train / %.0f %% test deposits', ...
            b.method, b.threshold, b.area_pct, b.train_captured_pct, b.test_captured_pct);
    end
end
imp = table(); nat = table();
if cfg.importance.enable
    if ~isempty(linTe) && numel(unique(yTe)) == 2
        Zi = Zte; yi = yTe; impSet = 'test samples';
    else
        Zi = Ztr; yi = yTr; impSet = 'training samples';
    end
    imp = mpm_permutation_importance(M, mdl, Zi, yi, F.source, F.layers, 'auc', cfg.importance.repeats, rs);
    imp.Properties.Description = [imp.Properties.Description ' on ' impSet];
    log.add('permutation importance on %s: top = %s', impSet, imp.layer(1));
end
if isfield(M, 'native_importance') && ~isempty(M.native_importance)
    nat = M.native_importance(mdl, F.names);
end
if patch, mpm_patch_source.active([]); end                % release the evidence cube

% ------------------------------------------------------------------ result struct
res = struct();
res.task = 'classification';
res.method = struct('key', M.key, 'name', M.name, 'description', M.description, 'model_text', modelText);
res.dataset = cfg.dataset_name;
res.created = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
res.cfg = cfg;
res.params = params;
res.grid = struct('R', S.R, 'geokeys', S.geokeys, 'x', S.x, 'y', S.y);
res.features = struct('layers', {F.layers}, 'kinds', {S.kinds}, 'names', {F.names}, ...
    'source', F.source, 'mu', sc.mu, 'sigma', sc.sigma, 'impute', F.impute, 'opts', F.opts);
res.deposits = D;
res.points_all = P;
res.samples = struct('train_cells', linTr, 'train_y', yTr, 'train_score', sTr, 'train_relevant', relTr, ...
    'test_cells', linTe, 'test_y', yTe, 'test_score', sTe, 'fit_cells', linFit, 'unlabelled', pu);
scoreName = 'prospectivity';
if isfield(M, 'score_name') && ~isempty(M.score_name), scoreName = M.score_name; end
res.map = struct('score', scoreMap, 'targets', targetMap, 'uncertainty', uncMap, 'valid', F.valid, ...
    'score_name', scoreName);
res.eval = ev;
res.threshold = thr;
res.targets = targetTable;
res.binary = binary;
res.importance = imp;
res.native_importance = nat;
res.tuning = tuning;
res.model = mdl;
res.log = log.lines;

% ------------------------------------------------------------------ 8. save + GUI
res = mpm_save_result(res, log);
if cfg.show_gui && usejava('desktop')
    mpm_results_viewer(res);
elseif cfg.save_figures && ~isempty(res.output_dir)
    mpm_results_viewer(res, 'Visible', 'off', 'ExportDir', fullfile(res.output_dir, 'figures'), 'Close', true);
end
end

function g = spatial_groups(lin, sz, reach)
%SPATIAL_GROUPS Connected components of samples linked when their Chebyshev distance <= reach.
[r, c] = ind2sub(sz, lin(:));
D = max(abs(r - r'), abs(c - c'));
A = D <= reach;
g = conncomp(graph(A, 'omitselfloops'))';
end

function lin = pu_cells(F, posTr, negTe, n, rs)
%PU_CELLS Unlabelled cells for PU learning: random valid cells of the study area except the
%   training positives and the test background cells.
cand = setdiff(F.idx, [posTr(:); negTe(:)]);
n = min(round(n), numel(cand));
lin = cand(sort(randperm(rs, numel(cand), n)));
end

function lin = fit_cells(F, R, trDep, opts, minDist, rs)
%FIT_CELLS Unlabelled cells an anomaly detector is fitted on (random sample of the study area).
%   opts.fit_cells 'all'        any valid cell (the published MPM practice: the whole study area)
%                  'background' only cells farther than cfg.negatives.min_distance from the
%                               TRAINING deposits (novelty detection w.r.t. barren ground)
%   opts.n_fit     number of cells (all valid cells if larger)
switch lower(opts.fit_cells)
    case 'all'
        cand = F.idx;
    case 'background'
        dep = false(R.RasterSize); dep(trDep) = true;
        d = bwdist(dep) * R.CellExtentInWorldX;
        cand = F.idx(d(F.idx) > minDist);
    otherwise
        error('mpm:run', 'cfg.anomaly.fit_cells must be ''all'' or ''background''.');
end
n = min(round(opts.n_fit), numel(cand));
lin = cand(sort(randperm(rs, numel(cand), n)));
end
