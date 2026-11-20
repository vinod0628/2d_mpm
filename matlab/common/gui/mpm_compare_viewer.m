function fig = mpm_compare_viewer(results, varargin)
%MPM_COMPARE_VIEWER Compare several MPM results (different methods, same task) in one window.
%
%   mpm_compare_viewer({res1, res2, ...})                 result structs
%   mpm_compare_viewer({'...\result.mat', '...'})         saved results
%   mpm_compare_viewer()                                  pick several result.mat files
%   fig = mpm_compare_viewer(..., 'Visible', 'off', 'ExportDir', dir, 'Close', true)
%
%   Classification: metric table, test ROC (samples and map), success/prediction-rate curves.
%   Regression:     metric table, test observed-vs-predicted, cut-off ROC.
%   Runs are comparable when they share the dataset, split and seed (run_compare_*.m does that).

ip = inputParser;
ip.addParameter('Visible', 'on'); ip.addParameter('ExportDir', ''); ip.addParameter('Close', false);
ip.parse(varargin{:}); opt = ip.Results;

if nargin < 1 || isempty(results)
    start = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results');
    [f, p] = uigetfile({'result.mat', 'MPM result'}, 'Select results to compare', start, 'MultiSelect', 'on');
    if isequal(f, 0), fig = []; return; end
    results = fullfile(p, cellstr(f));
end
if ~iscell(results), results = num2cell(results); end
for i = 1:numel(results)
    if ischar(results{i}) || isstring(results{i}), s = load(results{i}, 'res'); results{i} = s.res; end
end
task = results{1}.task;
if ~all(cellfun(@(r) strcmp(r.task, task), results))
    error('mpm:compare', 'All results must be of the same task (classification or regression).');
end
n = numel(results);
cols = lines(max(n, 7));
names = cellfun(@(r) sprintf('%s (%s)', r.method.name, r.created(end-7:end)), results, 'UniformOutput', false);

fig = uifigure('Name', sprintf('MPM comparison - %s - %d runs', task, n), 'Position', [100 80 1280 800], ...
    'Visible', opt.Visible);
tg = uitabgroup(uigridlayout(fig, [1 1]));

t = uitab(tg, 'Title', 'Metrics');
rows = table();
for i = 1:n
    T = mpm_metrics_table(results{i});
    T = addvars(T, repmat(string(names{i}), height(T), 1), repmat(string(results{i}.dataset), height(T), 1), ...
        'Before', 1, 'NewVariableNames', {'method', 'dataset'});
    rows = [rows; T]; %#ok<AGROW>
end
for j = 1:width(rows), if isfloat(rows.(j)), rows.(j) = round(rows.(j), 4, 'significant'); end, end
uitable(uigridlayout(t, [1 1]), 'Data', rows, 'ColumnSortable', true);

switch task
    case 'classification'
        t = uitab(tg, 'Title', 'ROC (test)');
        g = uigridlayout(t, [1 2]);
        ax1 = new_axes(g); ax2 = new_axes(g);
        for i = 1:n
            r = results{i};
            [e, lbl] = pick(r.eval.samples);
            plot(ax1, e.roc.fpr, e.roc.tpr, 'LineWidth', 2, 'Color', cols(i, :), ...
                'DisplayName', sprintf('%s: AUC %.3f (%s)', names{i}, e.auc, lbl));
            [e, lbl] = pick(r.eval.map);
            plot(ax2, e.roc.fpr, e.roc.tpr, 'LineWidth', 2, 'Color', cols(i, :), ...
                'DisplayName', sprintf('%s: AUC %.3f (%s)', names{i}, e.auc, lbl));
        end
        roc_axes(ax1, 'Samples: deposits vs. sampled background');
        roc_axes(ax2, 'Map: deposit cells vs. all background cells');
        t = uitab(tg, 'Title', 'Success / prediction rate');
        ax = new_axes(uigridlayout(t, [1 1]));
        plot(ax, [0 100], [0 100], ':', 'Color', [.5 .5 .5], 'HandleVisibility', 'off');
        for i = 1:n
            [e, lbl] = pick(results{i}.eval.map);
            plot(ax, 100 * e.area_frac, 100 * e.capture_frac, 'LineWidth', 2, 'Color', cols(i, :), ...
                'DisplayName', sprintf('%s: AUC %.3f (%s)', names{i}, e.sr_auc, lbl));
        end
        axis(ax, [0 100 0 100]); axis(ax, 'square'); legend(ax, 'Location', 'southeast');
        xlabel(ax, 'Study area (%)'); ylabel(ax, 'Deposits captured (%)');
        title(ax, 'Prediction-rate (test deposits; training if no test set)');
    case 'regression'
        t = uitab(tg, 'Title', 'Observed vs predicted (test)');
        ax = new_axes(uigridlayout(t, [1 1]));
        lo = inf; hi = -inf;
        for i = 1:n
            e = results{i}.eval.test_model; lbl = 'test';
            if isempty(e), e = results{i}.eval.train_model; lbl = 'train'; end
            scatter(ax, e.obs, e.pred, 18, cols(i, :), 'filled', 'MarkerFaceAlpha', 0.5, ...
                'DisplayName', sprintf('%s: R^2 %.3f, RMSE %.3g (%s)', names{i}, e.r2, e.rmse, lbl));
            lo = min([lo; e.obs; e.pred]); hi = max([hi; e.obs; e.pred]);
        end
        plot(ax, [lo hi], [lo hi], 'k-', 'DisplayName', '1:1'); axis(ax, [lo hi lo hi]); axis(ax, 'square');
        xlabel(ax, 'Observed (model scale)'); ylabel(ax, 'Predicted (model scale)'); legend(ax, 'Location', 'northwest');
        if all(cellfun(@(r) ~isempty(r.eval.cutoff), results))
            t = uitab(tg, 'Title', 'Cut-off ROC');
            ax = new_axes(uigridlayout(t, [1 1]));
            for i = 1:n
                c = results{i}.eval.cutoff;
                if isfield(c, 'test'), e = c.test; lbl = 'test'; else, e = c.train; lbl = 'train'; end
                plot(ax, e.fpr, e.tpr, 'LineWidth', 2, 'Color', cols(i, :), ...
                    'DisplayName', sprintf('%s: AUC %.3f (%s, cut-off %.4g)', names{i}, e.auc, lbl, c.value));
            end
            roc_axes(ax, 'Samples above the cut-off grade ranked by predicted grade');
        end
end

if ~isempty(opt.ExportDir)
    mpm_export_tabs(fig, tg, opt.ExportDir, 'compare_');
    if opt.Close, delete(fig); fig = []; end
end
end

function ax = new_axes(parent)
ax = uiaxes(parent); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
end

function [e, lbl] = pick(s)
if isfield(s, 'test'), e = s.test; lbl = 'test'; else, e = s.train; lbl = 'train'; end
end

function roc_axes(ax, ttl)
plot(ax, [0 1], [0 1], ':', 'Color', [.5 .5 .5], 'HandleVisibility', 'off');
axis(ax, [0 1 0 1]); axis(ax, 'square');
xlabel(ax, 'False positive rate'); ylabel(ax, 'True positive rate'); title(ax, ttl);
legend(ax, 'Location', 'southeast');
end
