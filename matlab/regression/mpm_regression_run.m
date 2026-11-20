function res = mpm_regression_run(cfg, M)
%MPM_REGRESSION_RUN Grade estimation from orebody samples + evidence layers with any regressor.
%
%   res = mpm_regression_run(cfg, M)
%
%   cfg  settings (see MPM_DEFAULT_CONFIG('regression'); only differences are needed)
%        required: data_dir, points_csv (x, y, <grade column>), target_field
%   M    method struct, e.g. reg_svm(), reg_rf()  (contract: regression/methods/reg_template.m)
%
%   Pipeline (identical for every method)
%     1. evidence rasters + features (same as classification)
%     2. grade samples -> cells; non-positive values handled for log transforms; samples sharing
%        a cell averaged on the model scale (cfg.duplicates = 'mean')
%     3. model scale y = log10(grade) (cfg.transform); train / test split (random | block | rule)
%     4. z-score features; k-fold CV grid search (RMSE on the model scale); final model
%     5. grade map (back-transformed) + uncertainty map if the method provides one
%     6. metrics on model and grade scale (R2, RMSE, MAE, bias, Pearson, Spearman) for training,
%        out-of-fold CV and test samples; cut-off ROC/AUC (samples above a cut-off grade ranked
%        by predicted grade); permutation importance per layer
%     7. save result.mat, GeoTIFFs, CSV tables; open the results GUI
%
%   Note: with a log transform the back-transformed prediction estimates the median grade
%   (geometric mean), not the arithmetic mean.

cfg = mpm_merge(mpm_default_config('regression'), cfg);
if ~strcmp(M.task, 'regression'), error('mpm:run', '%s is not a regression method.', M.name); end
cfg.data_dir = mpm_fullpath(cfg.data_dir); cfg.points_csv = mpm_fullpath(cfg.points_csv);
if isempty(cfg.target_field), error('mpm:run', 'cfg.target_field (grade column) is required.'); end
if isempty(cfg.target_label), cfg.target_label = cfg.target_field; end
log = mpm_Logger();
rs = RandStream('mt19937ar', 'Seed', cfg.seed);
rng(cfg.seed, 'twister');                 % global stream: TreeBagger, fitPosterior, ...
if isempty(cfg.dataset_name), cfg.dataset_name = mpm_dataset_label(cfg.data_dir); end
log.add('%s | %s | regression of %s', M.name, cfg.dataset_name, cfg.target_field);

% ------------------------------------------------------------------ 1. evidence + features
S = mpm_load_stack(cfg.data_dir, cfg.layers, cfg.exclude_layers, cfg.mask_layer);
F = mpm_build_features(S, cfg.features);
log.add('%d evidence layers -> %d features, %d valid cells (%dx%d grid, %.0f m)', numel(S.names), ...
    numel(F.names), numel(F.idx), S.R.RasterSize(1), S.R.RasterSize(2), S.R.CellExtentInWorldX);

% ------------------------------------------------------------------ 2. grade samples
P = mpm_read_points(cfg.points_csv, cfg.x_field, cfg.y_field);
vn = P.Properties.VariableNames;
k = find(strcmpi(vn, cfg.target_field), 1);
if isempty(k), error('mpm:run', 'Column "%s" not found in %s.', cfg.target_field, cfg.points_csv); end
g = double(P.(vn{k}));
n0 = height(P);
ok = isfinite(g);
[fwd, back] = transform_fns(cfg.transform);
if ~strcmp(cfg.transform, 'none')
    np = ok & g <= 0;
    switch cfg.nonpositive
        case 'drop', ok = ok & ~np;
        case 'halfmin', g(np) = 0.5 * min(g(ok & g > 0));
        otherwise, error('mpm:run', 'cfg.nonpositive must be ''drop'' or ''halfmin''.');
    end
    log.add('%d samples <= 0 (%s)', sum(np), cfg.nonpositive);
end
[lin, ~, ~, inside] = mpm_points_to_cells(S.R, P.x, P.y, F.valid);
ok = ok & inside;
log.add('%d samples, %d usable (%d outside the study area / nodata grade)', n0, sum(ok), n0 - sum(ok));
x = P.x(ok); y = P.y(ok); cellIdx = lin(ok); grade = g(ok); ym = fwd(grade);

nSamp = ones(numel(cellIdx), 1);
if strcmp(cfg.duplicates, 'mean')
    [uc, first, gi] = unique(cellIdx, 'stable');
    ym = accumarray(gi, ym, [], @mean);
    nSamp = accumarray(gi, 1);
    x = x(first); y = y(first); cellIdx = uc; grade = back(ym);
    log.add('%d cells with samples (duplicates averaged on the %s scale)', numel(cellIdx), cfg.transform);
end
T = table(x, y, cellIdx, nSamp, grade, ym, 'VariableNames', {'x', 'y', 'cell', 'n_samples', 'observed', 'observed_model'});
if height(T) < 10, error('mpm:run', 'Only %d usable samples.', height(T)); end

% ------------------------------------------------------------------ 3. split
T.test = mpm_split_points(T, cfg.split, rs);
tr = ~T.test; te = T.test;
log.add('%d training / %d test samples (%s split)', sum(tr), sum(te), cfg.split.method);
X = F.X(F.pos(T.cell), :);
sc = mpm_scaler(X(tr, :));
Z = sc.apply(X);

% ------------------------------------------------------------------ 4. tuning + final model
params = mpm_merge(M.defaults, cfg.params);
grid = cfg.tune.grid;
if isempty(grid), grid = M.grid; end
tuning = table(); cvPred = [];
if cfg.tune.enable && ~isempty(fieldnames(grid))
    log.add('CV tuning (%d-fold, %s) over %s', cfg.tune.kfold, cfg.tune.metric, mpm_grid_text(grid));
    [params, tuning, cvPred] = mpm_tune_cv(M, Z(tr, :), T.observed_model(tr), params, grid, ...
        cfg.tune.kfold, cfg.tune.metric, rs, [], log);
    b = tuning(tuning.best, :);
    log.add('best: %s  (CV %s = %.4f)', mpm_param_text(params, fieldnames(grid)), cfg.tune.metric, ...
        b.(['cv_' cfg.tune.metric]));
end
mdl = M.train(Z(tr, :), T.observed_model(tr), params, struct('final', true, 'task', 'regression'));
modelText = '';
if isfield(M, 'describe') && ~isempty(M.describe), modelText = M.describe(mdl, params); end
log.add('final model: %s', modelText);

% ------------------------------------------------------------------ 5. predictions + map
T.predicted_model = predict_all(M, mdl, Z);
T.relevant = false(height(T), 1);                         % sparse models: samples the model kept
if isfield(M, 'relevant_samples') && ~isempty(M.relevant_samples)
    trRows = find(tr);
    T.relevant(trRows(M.relevant_samples(mdl))) = true;
    log.add('relevant training samples: %d of %d', sum(T.relevant), sum(tr));
end
T.predicted = back(T.predicted_model);
T.residual_model = T.observed_model - T.predicted_model;
log.add('predicting %d cells ...', numel(F.idx));
[modelMap, uncMap] = mpm_predict_map(M, mdl, F, sc, cfg.chunk, log);
gradeMap = single(back(double(modelMap)));

% ------------------------------------------------------------------ 6. metrics
ev.train_model = mpm_eval_regression('training (model scale)', T.observed_model(tr), T.predicted_model(tr));
ev.train_grade = mpm_eval_regression('training (grade)', T.observed(tr), T.predicted(tr));
ev.test_model = []; ev.test_grade = []; ev.cv_model = []; ev.cv_grade = [];
if any(te)
    ev.test_model = mpm_eval_regression('test (model scale)', T.observed_model(te), T.predicted_model(te));
    ev.test_grade = mpm_eval_regression('test (grade)', T.observed(te), T.predicted(te));
end
if ~isempty(cvPred)
    yo = T.observed_model(tr);
    ev.cv_model = mpm_eval_regression('training CV out-of-fold (model scale)', yo, cvPred);
    ev.cv_grade = mpm_eval_regression('training CV out-of-fold (grade)', back(yo), back(cvPred));
end
cut = cfg.cutoff.value;
if isempty(cut) && ~isempty(cfg.cutoff.percentile), cut = prctile(T.observed, cfg.cutoff.percentile); end
ev.cutoff = [];
if ~isempty(cut)
    ev.cutoff.value = cut;
    ev.cutoff.train = mpm_roc(T.observed(tr) >= cut, T.predicted(tr));
    if any(te) && numel(unique(T.observed(te) >= cut)) == 2
        ev.cutoff.test = mpm_roc(T.observed(te) >= cut, T.predicted(te));
    end
end
msg = sprintf('R2 (model scale): train %.3f', ev.train_model.r2);
if ~isempty(ev.cv_model), msg = [msg sprintf(', CV %.3f', ev.cv_model.r2)]; end
if ~isempty(ev.test_model), msg = [msg sprintf(', test %.3f', ev.test_model.r2)]; end
if ~isempty(ev.cutoff)
    msg = [msg sprintf('  |  cut-off %.4g AUC: train %.3f', cut, ev.cutoff.train.auc)];
    if isfield(ev.cutoff, 'test'), msg = [msg sprintf(', test %.3f', ev.cutoff.test.auc)]; end
end
log.add('%s', msg);

imp = table(); nat = table();
if cfg.importance.enable
    if sum(te) >= 20, Zi = Z(te, :); yi = T.observed_model(te); impSet = 'test samples';
    else, Zi = Z(tr, :); yi = T.observed_model(tr); impSet = 'training samples'; end
    imp = mpm_permutation_importance(M, mdl, Zi, yi, F.source, F.layers, 'rmse', cfg.importance.repeats, rs);
    imp.Properties.Description = [imp.Properties.Description ' on ' impSet];
    log.add('permutation importance on %s: top = %s', impSet, imp.layer(1));
end
if isfield(M, 'native_importance') && ~isempty(M.native_importance)
    nat = M.native_importance(mdl, F.names);
end

% ------------------------------------------------------------------ result struct
res = struct();
res.task = 'regression';
res.method = struct('key', M.key, 'name', M.name, 'description', M.description, 'model_text', modelText);
res.dataset = cfg.dataset_name;
res.created = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
res.cfg = cfg;
res.params = params;
res.target_label = cfg.target_label;
res.grid = struct('R', S.R, 'geokeys', S.geokeys, 'x', S.x, 'y', S.y);
res.features = struct('layers', {F.layers}, 'kinds', {S.kinds}, 'names', {F.names}, ...
    'source', F.source, 'mu', sc.mu, 'sigma', sc.sigma, 'impute', F.impute, 'opts', F.opts);
res.samples_table = T;
res.map = struct('grade', gradeMap, 'model', modelMap, 'uncertainty', uncMap, 'valid', F.valid);
res.eval = ev;
res.importance = imp;
res.native_importance = nat;
res.tuning = tuning;
res.model = mdl;
res.log = log.lines;

% ------------------------------------------------------------------ 7. save + GUI
res = mpm_save_result(res, log);
if cfg.show_gui && usejava('desktop')
    mpm_results_viewer(res);
elseif cfg.save_figures && ~isempty(res.output_dir)
    mpm_results_viewer(res, 'Visible', 'off', 'ExportDir', fullfile(res.output_dir, 'figures'), 'Close', true);
end
end

function p = predict_all(M, mdl, Z)
if isfield(M, 'has_uncertainty') && M.has_uncertainty, [p, ~] = M.predict(mdl, Z);
else, p = M.predict(mdl, Z); end
p = double(p(:));
end

function [fwd, back] = transform_fns(t)
switch lower(t)
    case 'log10', fwd = @log10; back = @(v) 10.^v;
    case 'log',   fwd = @log;   back = @exp;
    case 'none',  fwd = @(v) v; back = @(v) v;
    otherwise, error('mpm:run', 'Unknown transform "%s".', t);
end
end
