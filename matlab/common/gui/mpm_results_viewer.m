function fig = mpm_results_viewer(res, varargin)
%MPM_RESULTS_VIEWER GUI for training / test results of any MPM method.
%
%   mpm_results_viewer(res)                      result struct from mpm_*_run
%   mpm_results_viewer('...\result.mat')         saved result
%   mpm_results_viewer()                         pick a result.mat
%   fig = mpm_results_viewer(res, 'Visible', 'off', 'ExportDir', dir, 'Close', true)
%                                                headless: write one PNG per tab
%
%   Classification tabs: Summary | ROC / AUC | Success-rate & P-A | Map | Confusion | Importance | Tuning
%   Regression tabs:     Summary | Observed vs predicted | Residuals | Grade map | Cut-off ROC |
%                        Importance | Tuning

ip = inputParser;
ip.addParameter('Visible', 'on');
ip.addParameter('ExportDir', '');
ip.addParameter('Close', false);
ip.parse(varargin{:});
opt = ip.Results;

if nargin < 1 || isempty(res)
    start = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results');
    [f, p] = uigetfile({'result.mat', 'MPM result (result.mat)'; '*.mat', 'MAT-files'}, 'Open MPM result', start);
    if isequal(f, 0), fig = []; return; end
    res = fullfile(p, f);
end
if ischar(res) || isstring(res)
    file = char(res);
    s = load(file, 'res'); res = s.res;
    if ~isfield(res, 'output_dir') || isempty(res.output_dir) || ~isfolder(res.output_dir)
        res.output_dir = fileparts(file);
    end
end

C = palette();
fig = uifigure('Name', sprintf('MPM results - %s - %s (%s)', res.method.name, res.dataset, res.task), ...
    'Position', [80 60 1280 820], 'Visible', opt.Visible, 'Color', C.bg);
main = uigridlayout(fig, [2 1], 'RowHeight', {40, '1x'}, 'Padding', [8 8 8 8], 'RowSpacing', 6);

bar = uigridlayout(main, [1 5], 'ColumnWidth', {'1x', 110, 120, 110, 100}, 'Padding', [0 0 0 0]);
uilabel(bar, 'Text', sprintf('<b>%s</b> &nbsp;|&nbsp; %s &nbsp;|&nbsp; %s &nbsp;|&nbsp; %s', res.method.name, ...
    res.dataset, res.task, res.created), 'Interpreter', 'html', 'FontSize', 14);
uibutton(bar, 'Text', 'Open result...', 'ButtonPushedFcn', @(~, ~) mpm_results_viewer());
uibutton(bar, 'Text', 'Export figures...', 'ButtonPushedFcn', @(~, ~) export_dialog());
b = uibutton(bar, 'Text', 'Output folder', 'ButtonPushedFcn', @(~, ~) open_folder(res));
b.Enable = isfield(res, 'output_dir') && ~isempty(res.output_dir) && isfolder(res.output_dir);
uibutton(bar, 'Text', 'Close', 'ButtonPushedFcn', @(~, ~) delete(fig));

tg = uitabgroup(main);
switch res.task
    case 'classification'
        tab_clf_summary(tg, res);
        tab_clf_roc(tg, res, C);
        tab_clf_rates(tg, res, C);
        tab_clf_map(tg, res, C);
        tab_binary(tg, res, 'deposits', lower(score_label(res)));
        tab_clf_confusion(tg, res);
    case 'regression'
        tab_reg_summary(tg, res);
        tab_reg_scatter(tg, res, C);
        tab_reg_residuals(tg, res, C);
        tab_reg_map(tg, res, C);
        tab_reg_cutoff(tg, res, C);
end
tab_importance(tg, res, C);
tab_tuning(tg, res);

if ~isempty(opt.ExportDir)
    mpm_export_tabs(fig, tg, opt.ExportDir);
    if opt.Close, delete(fig); fig = []; end
end

    function export_dialog()
        d0 = ''; if isfield(res, 'output_dir'), d0 = res.output_dir; end
        d = uigetdir(d0, 'Folder for PNG figures');
        if isequal(d, 0), return; end
        mpm_export_tabs(fig, tg, d);
        uialert(fig, sprintf('Figures written to\n%s', d), 'Export', 'Icon', 'success');
    end
end

% =============================================================================== classification
function tab_clf_summary(tg, res)
t = uitab(tg, 'Title', 'Summary');
g = uigridlayout(t, [3 2], 'ColumnWidth', {'3x', '2x'}, 'RowHeight', {'1x', '1x', '1.3x'});
T = mpm_metrics_table(res);
isMap = contains(T.set, 'all cells');
p1 = uipanel(g, 'Title', sprintf('Sample validation: deposits vs. sampled background (threshold %.3g)', res.threshold));
p1.Layout.Row = 1; p1.Layout.Column = 1;
show_table(p1, round_table(T(~isMap, {'set', 'positives', 'negatives', 'roc_auc', 'accuracy', 'precision', ...
    'recall', 'specificity', 'f1', 'kappa', 'brier'})));
Tm = T(isMap, {'set', 'positives', 'negatives', 'roc_auc', 'success_rate_auc', 'pa_threshold', ...
    'pa_prediction_pct', 'pa_area_pct'});
Tm.Properties.VariableNames(2:3) = {'deposits', 'background_cells'};
p1 = uipanel(g, 'Title', 'Map validation: deposit cells vs. every other cell of the study area (P-A = prediction-area intersection)');
p1.Layout.Row = 2; p1.Layout.Column = 1;
show_table(p1, round_table(Tm));
p2 = uipanel(g, 'Title', 'Exploration targets (classes of the prospectivity map)');
p2.Layout.Row = 3; p2.Layout.Column = 1;
show_table(p2, round_table(res.targets));
p3 = uipanel(g, 'Title', 'Model, settings and log');
p3.Layout.Row = [1 3]; p3.Layout.Column = 2;
gl = uigridlayout(p3, [1 1], 'Padding', [2 2 2 2]);
uitextarea(gl, 'Value', info_lines(res), 'Editable', 'off', 'FontName', 'Consolas', 'FontSize', 11);
end

function tab_clf_roc(tg, res, C)
t = uitab(tg, 'Title', 'ROC / AUC');
g = uigridlayout(t, [1 2], 'ColumnWidth', {'1x', '1x'});
ax = uiaxes(g); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
plot(ax, [0 1], [0 1], ':', 'Color', C.grey, 'HandleVisibility', 'off');
E = res.eval.samples;
plot(ax, E.train.roc.fpr, E.train.roc.tpr, '-', 'Color', C.train, 'LineWidth', 2, ...
    'DisplayName', sprintf('Training (AUC = %.3f, n = %d+%d)', E.train.auc, E.train.npos, E.train.nneg));
if isfield(E, 'test')
    plot(ax, E.test.roc.fpr, E.test.roc.tpr, '-', 'Color', C.test, 'LineWidth', 2, ...
        'DisplayName', sprintf('Test (AUC = %.3f, n = %d+%d)', E.test.auc, E.test.npos, E.test.nneg));
end
finish_roc(ax, 'Samples: deposits vs. sampled background');
ax = uiaxes(g); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
plot(ax, [0 1], [0 1], ':', 'Color', C.grey, 'HandleVisibility', 'off');
Mp = res.eval.map;
plot(ax, Mp.train.roc.fpr, Mp.train.roc.tpr, '-', 'Color', C.train, 'LineWidth', 2, ...
    'DisplayName', sprintf('Training deposits (AUC = %.3f, n = %d)', Mp.train.auc, Mp.train.npos));
if isfield(Mp, 'test')
    plot(ax, Mp.test.roc.fpr, Mp.test.roc.tpr, '-', 'Color', C.test, 'LineWidth', 2, ...
        'DisplayName', sprintf('Test deposits (AUC = %.3f, n = %d)', Mp.test.auc, Mp.test.npos));
end
finish_roc(ax, sprintf('Map: deposit cells vs. all %s background cells', fmt_int(Mp.train.nneg)));
end

function finish_roc(ax, ttl)
axis(ax, [0 1 0 1]); axis(ax, 'square');
xlabel(ax, 'False positive rate (1 - specificity)'); ylabel(ax, 'True positive rate (sensitivity)');
title(ax, ttl);
legend(ax, 'Location', 'southeast');
end

function tab_clf_rates(tg, res, C)
t = uitab(tg, 'Title', 'Success-rate & P-A');
g = uigridlayout(t, [1 2], 'ColumnWidth', {'1x', '1x'});
Mp = res.eval.map;
ax = uiaxes(g); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
plot(ax, [0 100], [0 100], ':', 'Color', C.grey, 'HandleVisibility', 'off');
plot(ax, 100 * Mp.train.area_frac, 100 * Mp.train.capture_frac, '-', 'Color', C.train, 'LineWidth', 2, ...
    'DisplayName', sprintf('Success rate, training (AUC = %.3f)', Mp.train.sr_auc));
if isfield(Mp, 'test')
    plot(ax, 100 * Mp.test.area_frac, 100 * Mp.test.capture_frac, '-', 'Color', C.test, 'LineWidth', 2, ...
        'DisplayName', sprintf('Prediction rate, test (AUC = %.3f)', Mp.test.sr_auc));
end
axis(ax, [0 100 0 100]); axis(ax, 'square');
xlabel(ax, 'Study area classified as prospective (%, highest scores first)');
ylabel(ax, 'Deposits captured (%)');
title(ax, 'Success-rate / prediction-rate curves'); legend(ax, 'Location', 'southeast');

ax = uiaxes(g); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
yyaxis(ax, 'left');
plot(ax, Mp.train.pa_thr, Mp.train.pa_pred, '-', 'Color', C.train, 'LineWidth', 2, 'DisplayName', 'Prediction rate (training)');
if isfield(Mp, 'test')
    plot(ax, Mp.test.pa_thr, Mp.test.pa_pred, '--', 'Color', C.test, 'LineWidth', 1.8, 'DisplayName', 'Prediction rate (test)');
end
ylabel(ax, 'Deposits with score >= threshold (%)'); ylim(ax, [0 100]); ax.YColor = C.train;
yyaxis(ax, 'right');
plot(ax, Mp.train.pa_thr, Mp.train.pa_area, '-', 'Color', C.area, 'LineWidth', 2, 'DisplayName', 'Occupied area');
ylabel(ax, 'Area with score >= threshold (%)'); ax.YColor = C.area; ax.YDir = 'reverse'; ylim(ax, [0 100]);
xlim(ax, [min(Mp.train.pa_thr) max(Mp.train.pa_thr)]);
pa = Mp.train.pa_intersection;
if isfinite(pa.threshold)
    yyaxis(ax, 'left');
    plot(ax, pa.threshold, pa.prediction, 'o', 'MarkerSize', 9, 'MarkerFaceColor', C.mark, 'MarkerEdgeColor', 'k', ...
        'DisplayName', sprintf('Intersection: %.1f %% deposits in %.1f %% area (t = %.3g)', pa.prediction, pa.area, pa.threshold));
end
xlabel(ax, [score_label(res) ' threshold']);
title(ax, 'Prediction-area (P-A) plot'); legend(ax, 'Location', 'southwest');
end

function tab_clf_map(tg, res, C)
t = uitab(tg, 'Title', 'Prospectivity map');
g = uigridlayout(t, [2 1], 'RowHeight', {30, '1x'});
ctl = uigridlayout(g, [1 6], 'ColumnWidth', {60, 290, 150, 150, 170, '1x'}, 'Padding', [0 0 0 0]);
uilabel(ctl, 'Text', 'Show:');
sn = score_label(res); isProb = strcmp(sn, 'Prospectivity');
if isProb, items = {'Prospectivity (0-1)', 'Target classes'};
else
    isPU = isfield(res.samples, 'unlabelled') && res.samples.unlabelled;
    if isPU, how = 'higher = more prospective'; else, how = 'higher = more anomalous'; end
    items = {sprintf('%s (%s)', sn, how), 'Target classes'};
end
hasUnc = isfield(res.map, 'uncertainty') && ~isempty(res.map.uncertainty);
if hasUnc && isProb, items{end+1} = 'Uncertainty (std. of probability)';
elseif hasUnc, items{end+1} = 'Uncertainty (std. of ensemble members)'; end
dd = uidropdown(ctl, 'Items', items);
cbTr = uicheckbox(ctl, 'Text', 'Training deposits', 'Value', true);
cbTe = uicheckbox(ctl, 'Text', 'Test deposits', 'Value', true);
rel = []; if isfield(res.samples, 'train_relevant'), rel = res.samples.train_relevant; end
cbRv = uicheckbox(ctl, 'Text', 'Relevance vectors', 'Value', any(rel), 'Enable', any(rel));
ax = uiaxes(g);
redraw();
dd.ValueChangedFcn = @(~, ~) redraw(); cbTr.ValueChangedFcn = @(~, ~) redraw(); cbTe.ValueChangedFcn = @(~, ~) redraw();
cbRv.ValueChangedFcn = @(~, ~) redraw();
    function redraw()
        cla(ax, 'reset'); hold(ax, 'on');
        x = res.grid.x; y = res.grid.y;
        if strcmp(dd.Value, 'Target classes')
            A = double(res.map.targets); A(A == 0) = NaN;
            K = height(res.targets);
            imagesc(ax, x, y, A, 'AlphaData', ~isnan(A));
            colormap(ax, target_colors(K)); clim(ax, [0.5 K + 0.5]);
            cb = colorbar(ax); cb.Ticks = 1:K; cb.TickLabels = flipud(res.targets.label);
            title(ax, 'Exploration target classes');
        elseif startsWith(dd.Value, 'Uncertainty')
            A = res.map.uncertainty;
            imagesc(ax, x, y, A, 'AlphaData', ~isnan(A));
            colormap(ax, parula(256));
            cb = colorbar(ax); cb.Label.String = 'Std. of probability';
            title(ax, sprintf('Prediction uncertainty - %s', res.method.name));
        else
            A = res.map.score;
            imagesc(ax, x, y, A, 'AlphaData', ~isnan(A));
            colormap(ax, turbo(256)); clim(ax, [0 1]);
            if ~isProb || ~all(A(res.map.valid) >= 0 & A(res.map.valid) <= 1), clim(ax, 'auto'); end
            cb = colorbar(ax); cb.Label.String = sn;
            title(ax, sprintf('%s - %s', sn, res.method.name));
        end
        D = res.deposits;
        if cbTr.Value && any(~D.test)
            plot(ax, D.x(~D.test), D.y(~D.test), 'p', 'MarkerSize', 11, 'MarkerFaceColor', 'w', ...
                'MarkerEdgeColor', 'k', 'DisplayName', 'Training deposits');
        end
        if cbTe.Value && any(D.test)
            plot(ax, D.x(D.test), D.y(D.test), 'p', 'MarkerSize', 11, 'MarkerFaceColor', C.test, ...
                'MarkerEdgeColor', 'k', 'DisplayName', 'Test deposits');
        end
        if cbRv.Value && any(rel)
            [r, c] = ind2sub(size(res.map.valid), res.samples.train_cells(rel));
            yy = res.samples.train_y(rel);
            plot(ax, x(c(yy)), y(r(yy)), 'o', 'MarkerSize', 13, 'LineWidth', 1.8, 'Color', 'm', ...
                'DisplayName', sprintf('Relevance vectors: deposit (%d)', sum(yy)));
            plot(ax, x(c(~yy)), y(r(~yy)), 's', 'MarkerSize', 9, 'LineWidth', 1.5, 'Color', 'w', ...
                'MarkerFaceColor', 'k', 'DisplayName', sprintf('Relevance vectors: background (%d)', sum(~yy)));
        end
        if (cbTr.Value || cbTe.Value || cbRv.Value), legend(ax, 'Location', 'northeastoutside'); end
        map_axes(ax);
    end
end

function tab_clf_confusion(tg, res)
t = uitab(tg, 'Title', 'Confusion matrix');
E = res.eval.samples;
sets = {'train'}; if isfield(E, 'test'), sets{end+1} = 'test'; end
g = uigridlayout(t, [1 numel(sets)]);
for i = 1:numel(sets)
    e = E.(sets{i});
    p = uipanel(g, 'Title', sprintf('%s (threshold %.3g)  accuracy %.3f, kappa %.3f, F1 %.3f', ...
        e.name, e.threshold, e.accuracy, e.kappa, e.f1));
    cc = confusionchart(p, e.confusion, {'background', 'deposit'});
    cc.RowSummary = 'row-normalized'; cc.ColumnSummary = 'column-normalized';
    cc.XLabel = 'Predicted'; cc.YLabel = 'True';
end
end

% =============================================================================== regression
function tab_reg_summary(tg, res)
t = uitab(tg, 'Title', 'Summary');
g = uigridlayout(t, [2 2], 'ColumnWidth', {'3x', '2x'}, 'RowHeight', {'1x', '1x'});
p1 = uipanel(g, 'Title', sprintf('Validation metrics (model scale = %s, grade scale = original units)', res.cfg.transform));
p1.Layout.Row = 1; p1.Layout.Column = 1;
show_table(p1, round_table(mpm_metrics_table(res)));
p2 = uipanel(g, 'Title', 'Samples (first rows; full table in samples.csv)');
p2.Layout.Row = 2; p2.Layout.Column = 1;
S = res.samples_table; show_table(p2, round_table(S(1:min(height(S), 500), :)));
p3 = uipanel(g, 'Title', 'Model, settings and log');
p3.Layout.Row = [1 2]; p3.Layout.Column = 2;
gl = uigridlayout(p3, [1 1], 'Padding', [2 2 2 2]);
uitextarea(gl, 'Value', info_lines(res), 'Editable', 'off', 'FontName', 'Consolas', 'FontSize', 11);
end

function tab_reg_scatter(tg, res, C)
t = uitab(tg, 'Title', 'Observed vs predicted');
g = uigridlayout(t, [1 2]);
lbl = res.target_label;
scatter_panel(uiaxes(g), res, 'model', sprintf('%s (%s scale)', lbl, res.cfg.transform), C);
scatter_panel(uiaxes(g), res, 'grade', lbl, C);
end

function scatter_panel(ax, res, scale, lbl, C)
hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
E = res.eval; lo = inf; hi = -inf;
sets = {'train', 'test', 'cv'}; cols = {C.train, C.test, C.grey};
names = {'Training', 'Test', 'Training, out-of-fold CV'};
for i = [3 1 2]
    k = [sets{i} '_' scale];
    if ~isfield(E, k) || isempty(E.(k)), continue; end
    e = E.(k);
    mk = 'o'; sz = 22; if i == 3, sz = 10; end
    scatter(ax, e.obs, e.pred, sz, cols{i}, mk, 'filled', 'MarkerFaceAlpha', 0.6, ...
        'DisplayName', sprintf('%s: R^2 = %.3f, RMSE = %.3g (n = %d)', names{i}, e.r2, e.rmse, e.n));
    lo = min([lo; e.obs; e.pred]); hi = max([hi; e.obs; e.pred]);
end
if isfinite(lo)
    plot(ax, [lo hi], [lo hi], 'k-', 'DisplayName', '1:1');
    axis(ax, [lo hi lo hi]);
end
if strcmp(scale, 'grade') && ~strcmp(res.cfg.transform, 'none') && lo > 0
    ax.XScale = 'log'; ax.YScale = 'log';
end
axis(ax, 'square');
xlabel(ax, ['Observed ' lbl]); ylabel(ax, ['Predicted ' lbl]);
title(ax, sprintf('Observed vs predicted (%s scale)', scale));
legend(ax, 'Location', 'northwest');
end

function tab_reg_residuals(tg, res, C)
t = uitab(tg, 'Title', 'Residuals');
g = uigridlayout(t, [1 2]);
E = res.eval;
ax = uiaxes(g); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
histogram(ax, E.train_model.resid, 30, 'FaceColor', C.train, 'FaceAlpha', 0.6, 'Normalization', 'pdf', 'DisplayName', 'Training');
if isfield(E, 'test_model') && ~isempty(E.test_model)
    histogram(ax, E.test_model.resid, 30, 'FaceColor', C.test, 'FaceAlpha', 0.6, 'Normalization', 'pdf', 'DisplayName', 'Test');
end
xline(ax, 0, 'k-', 'HandleVisibility', 'off');
xlabel(ax, sprintf('Residual = observed - predicted (%s scale)', res.cfg.transform)); ylabel(ax, 'Density');
title(ax, 'Residual distribution'); legend(ax);
ax = uiaxes(g); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
scatter(ax, E.train_model.pred, E.train_model.resid, 20, C.train, 'filled', 'MarkerFaceAlpha', 0.6, 'DisplayName', 'Training');
if isfield(E, 'test_model') && ~isempty(E.test_model)
    scatter(ax, E.test_model.pred, E.test_model.resid, 20, C.test, 'filled', 'MarkerFaceAlpha', 0.7, 'DisplayName', 'Test');
end
yline(ax, 0, 'k-', 'HandleVisibility', 'off');
xlabel(ax, sprintf('Predicted (%s scale)', res.cfg.transform)); ylabel(ax, 'Residual');
title(ax, 'Residuals vs predicted'); legend(ax);
end

function tab_reg_map(tg, res, C)
t = uitab(tg, 'Title', 'Grade map');
g = uigridlayout(t, [2 1], 'RowHeight', {30, '1x'});
ctl = uigridlayout(g, [1 7], 'ColumnWidth', {60, 230, 110, 140, 130, 150, '1x'}, 'Padding', [0 0 0 0]);
uilabel(ctl, 'Text', 'Show:');
items = {'Predicted grade', 'Predicted (model scale)'};
if ~isempty(res.map.uncertainty), items{end+1} = 'Uncertainty (model scale)'; end
dd = uidropdown(ctl, 'Items', items);
cbLog = uicheckbox(ctl, 'Text', 'Log colours', 'Value', ~strcmp(res.cfg.transform, 'none'));
cbTr = uicheckbox(ctl, 'Text', 'Training samples', 'Value', true);
cbTe = uicheckbox(ctl, 'Text', 'Test samples', 'Value', true);
S0 = res.samples_table;
hasRv = ismember('relevant', S0.Properties.VariableNames) && any(S0.relevant);
cbRv = uicheckbox(ctl, 'Text', 'Relevance vectors', 'Value', hasRv, 'Enable', hasRv);
ax = uiaxes(g);
redraw();
dd.ValueChangedFcn = @(~, ~) redraw(); cbLog.ValueChangedFcn = @(~, ~) redraw();
cbTr.ValueChangedFcn = @(~, ~) redraw(); cbTe.ValueChangedFcn = @(~, ~) redraw();
cbRv.ValueChangedFcn = @(~, ~) redraw();
    function redraw()
        cla(ax, 'reset'); hold(ax, 'on');
        S = res.samples_table;
        switch dd.Value
            case 'Predicted grade', A = res.map.grade; v = S.observed; ttl = ['Predicted ' res.target_label];
            case 'Predicted (model scale)', A = res.map.model; v = S.observed_model; ttl = 'Prediction (model scale)';
            otherwise, A = res.map.uncertainty; v = []; ttl = 'Prediction uncertainty (std., model scale)';
        end
        A = double(A); useLog = cbLog.Value && strcmp(dd.Value, 'Predicted grade') && all(A(isfinite(A)) > 0);
        if useLog, A = log10(A); if ~isempty(v), v = log10(max(v, eps)); end, end
        imagesc(ax, res.grid.x, res.grid.y, A, 'AlphaData', isfinite(A));
        colormap(ax, turbo(256));
        lim = quantile(A(isfinite(A)), [0.01 0.99]); if ~isempty(v), lim = [min(lim(1), quantile(v, 0.02)), max(lim(2), quantile(v, 0.98))]; end
        if all(isfinite(lim)) && lim(2) > lim(1), clim(ax, lim); end
        cb = colorbar(ax); cb.Label.String = ttl; if useLog, cb.Label.String = ['log_{10} ' ttl]; end
        if ~isempty(v)
            if cbTr.Value
                scatter(ax, S.x(~S.test), S.y(~S.test), 28, v(~S.test), 'o', 'filled', 'MarkerEdgeColor', 'k', ...
                    'DisplayName', 'Training samples (observed)');
            end
            if cbTe.Value && any(S.test)
                scatter(ax, S.x(S.test), S.y(S.test), 40, v(S.test), 's', 'filled', 'MarkerEdgeColor', C.test, ...
                    'LineWidth', 1.5, 'DisplayName', 'Test samples (observed)');
            end
        end
        if hasRv && cbRv.Value
            plot(ax, S.x(S.relevant), S.y(S.relevant), 'o', 'MarkerSize', 14, 'LineWidth', 2, 'Color', 'm', ...
                'DisplayName', sprintf('Relevance vectors (%d)', sum(S.relevant)));
        end
        if (~isempty(v) && (cbTr.Value || cbTe.Value)) || (hasRv && cbRv.Value)
            legend(ax, 'Location', 'northeastoutside');
        end
        title(ax, sprintf('%s - %s', ttl, res.method.name));
        map_axes(ax);
    end
end

function tab_reg_cutoff(tg, res, C)
if ~isfield(res.eval, 'cutoff') || isempty(res.eval.cutoff), return; end
c = res.eval.cutoff;
t = uitab(tg, 'Title', 'Cut-off ROC');
g = uigridlayout(t, [1 1]);
ax = uiaxes(g); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
plot(ax, [0 1], [0 1], ':', 'Color', C.grey, 'HandleVisibility', 'off');
plot(ax, c.train.fpr, c.train.tpr, '-', 'Color', C.train, 'LineWidth', 2, ...
    'DisplayName', sprintf('Training (AUC = %.3f, %d >= cut-off of %d)', c.train.auc, c.train.npos, c.train.npos + c.train.nneg));
if isfield(c, 'test')
    plot(ax, c.test.fpr, c.test.tpr, '-', 'Color', C.test, 'LineWidth', 2, ...
        'DisplayName', sprintf('Test (AUC = %.3f, %d >= cut-off of %d)', c.test.auc, c.test.npos, c.test.npos + c.test.nneg));
end
finish_roc(ax, sprintf('Ranking of samples above the cut-off grade %.4g %s by predicted grade', c.value, res.target_label));
end

% =============================================================================== shared tabs
function tab_binary(tg, res, pointLabel, scoreName)
% binary target grid (classification): 1 = predicted deposit area, 0 = non-deposit (P-A / C-A)
if ~isfield(res, 'binary') || isempty(res.binary), return; end
t = uitab(tg, 'Title', 'Binary targets (P-A / C-A)');
mpm_binary_panel(t, res.binary, res.grid.x, res.grid.y, res.binary.points, pointLabel, scoreName);
end

function tab_importance(tg, res, C)
hasPerm = ~isempty(res.importance); hasNat = ~isempty(res.native_importance);
if ~hasPerm && ~hasNat, return; end
t = uitab(tg, 'Title', 'Importance');
g = uigridlayout(t, [1 hasPerm + hasNat]);
if hasPerm
    T = res.importance; T = T(end:-1:1, :);
    ax = uiaxes(g); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    barh(ax, T.importance, 'FaceColor', C.train);
    errorbar(ax, T.importance, 1:height(T), T.importance_sd, 'horizontal', 'k.', 'LineWidth', 1);
    ax.YTick = 1:height(T); ax.YTickLabel = T.layer; ax.TickLabelInterpreter = 'none';
    xlabel(ax, 'Loss of skill when the layer is permuted (AUC / R^2 drop or RMSE increase)');
    title(ax, T.Properties.Description, 'Interpreter', 'none', 'FontSize', 11);
end
if hasNat
    T = res.native_importance; T = T(min(height(T), 25):-1:1, :);
    ax = uiaxes(g); grid(ax, 'on'); box(ax, 'on');
    barh(ax, T.importance, 'FaceColor', C.test);
    ax.YTick = 1:height(T); ax.YTickLabel = T.feature; ax.TickLabelInterpreter = 'none';
    xlabel(ax, 'Importance');
    title(ax, [res.native_importance.Properties.Description ' (top 25 features)'], 'Interpreter', 'none', 'FontSize', 11);
end
end

function tab_tuning(tg, res)
if isempty(res.tuning), return; end
T = res.tuning;
pre = {'cv_', 'sel_'};                    % sel_ = selection score without CV (anomaly detectors)
mcol = T.Properties.VariableNames{find(startsWith(T.Properties.VariableNames, pre), 1)};
ttl = 'Tuning (CV)'; if startsWith(mcol, 'sel_'), ttl = 'Tuning'; end
t = uitab(tg, 'Title', ttl);
pn = setdiff(T.Properties.VariableNames, {'best'}, 'stable');
pn = pn(~startsWith(pn, pre));
two = numel(pn) == 2 && isnumeric(T.(pn{1})) && isnumeric(T.(pn{2}));
g = uigridlayout(t, [1 1 + two]);
show_table(g, round_table(T));
if two
    p = uipanel(g, 'Title', sprintf('%s by %s and %s (best marked in the table)', mcol, pn{1}, pn{2}));
    h = heatmap(p, T, pn{2}, pn{1}, 'ColorVariable', mcol, 'ColorMethod', 'mean');
    h.Colormap = parula(256); h.CellLabelFormat = '%.3f';
end
end

% =============================================================================== helpers
function L = info_lines(res)
cfg = res.cfg;
L = {sprintf('Method      : %s (%s)', res.method.name, res.method.key)
    sprintf('Model       : %s', res.method.model_text)
    sprintf('Dataset     : %s', res.dataset)
    sprintf('Data folder : %s', cfg.data_dir)
    sprintf('Points      : %s', cfg.points_csv)
    sprintf('Created     : %s', res.created)
    sprintf('Output      : %s', field_or(res, 'output_dir', '(not saved)'))
    ''
    'Evidence layers:'};
for i = 1:numel(res.features.layers)
    kind = res.features.kinds{i};
    if strcmp(kind, 'categorical') && (ismember(res.features.layers{i}, cellstr(res.features.opts.as_numeric)) ...
            || strcmp(res.features.opts.categorical, 'numeric'))
        kind = 'categorical, used as numeric';
    elseif strcmp(kind, 'categorical')
        kind = 'categorical, one-hot';
    end
    L{end+1, 1} = sprintf('  %-28s %s', res.features.layers{i}, kind); %#ok<AGROW>
end
L{end+1} = sprintf('Features    : %d (categorical = %s, missing = %s)', numel(res.features.names), ...
    res.features.opts.categorical, res.features.opts.missing);
L{end+1} = sprintf('Split       : %s, test fraction %.2f, seed %d', cfg.split.method, cfg.split.test_fraction, cfg.seed);
if strcmp(res.task, 'classification')
    L{end+1} = sprintf('Positives   : buffer %d cells', cfg.positives.buffer_cells);
    if isfield(res.samples, 'unlabelled') && res.samples.unlabelled
        L{end+1} = sprintf('Unlabelled  : %d random cells of the study area (PU learning, no distance buffer)', ...
            sum(~res.samples.train_y));
        L{end+1} = sprintf('Test bkgd   : %s, ratio %.3g, min distance %.0f m', cfg.negatives.strategy, ...
            cfg.negatives.ratio, cfg.negatives.min_distance);
    else
        L{end+1} = sprintf('Negatives   : %s, ratio %.3g, min distance %.0f m', cfg.negatives.strategy, ...
            cfg.negatives.ratio, cfg.negatives.min_distance);
    end
    L{end+1} = sprintf('Threshold   : %.4g', res.threshold);
else
    L{end+1} = sprintf('Target      : %s, transform %s, duplicates %s', cfg.target_field, cfg.transform, cfg.duplicates);
end
L{end+1} = sprintf('Tuning      : %s (%d-fold, %s)', mat2str(cfg.tune.enable), cfg.tune.kfold, cfg.tune.metric);
L{end+1} = '';
L{end+1} = 'Final parameters:';
f = fieldnames(res.params);
for i = 1:numel(f)
    v = res.params.(f{i}); if isnumeric(v) || islogical(v), v = mat2str(v); end
    L{end+1} = sprintf('  %-22s %s', f{i}, v); %#ok<AGROW>
end
L{end+1} = '';
L{end+1} = 'Log:';
L = [L; res.log(:)];
end

function v = field_or(s, f, d)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end

function show_table(parent, T)
if ~isa(parent, 'matlab.ui.container.GridLayout'), parent = uigridlayout(parent, [1 1], 'Padding', [2 2 2 2]); end
uitable(parent, 'Data', T, 'ColumnSortable', true);
end

function T = round_table(T)
for i = 1:width(T)
    v = T.(i);
    if isfloat(v), T.(i) = round(v, 4, 'significant'); end
end
end

function map_axes(ax)
axis(ax, 'image'); ax.YDir = 'normal'; box(ax, 'on');
xlabel(ax, 'Easting (m)'); ylabel(ax, 'Northing (m)');
ax.XAxis.Exponent = 0; ax.YAxis.Exponent = 0;
xtickformat(ax, '%.0f'); ytickformat(ax, '%.0f');
end

function cmap = target_colors(K)
base = [0.86 0.86 0.86; 1.00 0.89 0.45; 0.99 0.60 0.18; 0.84 0.19 0.15; 0.50 0.00 0.15];
if K <= size(base, 1), cmap = base(1:K, :);
else, cmap = [base(1, :); hot(K - 1)]; end
end

function s = fmt_int(n)
s = regexprep(sprintf('%d', round(n)), '(\d)(?=(\d{3})+$)', '$1,');
end

function C = palette()
C.bg = [0.97 0.97 0.97];
C.train = [0.00 0.45 0.74];
C.test = [0.85 0.33 0.10];
C.area = [0.47 0.67 0.19];
C.mark = [1.00 0.84 0.00];
C.grey = [0.55 0.55 0.55];
end

function open_folder(res)
if ispc, winopen(res.output_dir); else, disp(res.output_dir); end
end

function s = score_label(res)
% name of the classification map score: 'Prospectivity' or e.g. 'Anomaly score' (anomaly detectors)
s = 'Prospectivity';
if isfield(res.map, 'score_name') && ~isempty(res.map.score_name)
    s = char(res.map.score_name); s(1) = upper(s(1));
end
end
