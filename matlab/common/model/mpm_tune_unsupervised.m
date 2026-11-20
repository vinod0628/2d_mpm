function [best, T] = mpm_tune_unsupervised(M, Zfit, Zlab, ylab, base, grid, log)
%MPM_TUNE_UNSUPERVISED Grid search for anomaly detectors (fitted without labels).
%
%   [best, T] = mpm_tune_unsupervised(M, Zfit, Zlab, ylab, base, grid, log)
%
%   Every combination is fitted on the unlabelled cells Zfit (labels are never seen by the fit),
%   then scored on the labelled TRAINING samples Zlab / ylab (training deposits vs. training
%   background); the combination with the highest AUC is kept. Because the fit does not use the
%   labels, no cross-validation folds are needed; the test deposits stay independent.
%   T has one row per combination: parameter columns, sel_auc, best.
if nargin < 7 || isempty(log), log = mpm_Logger(true); end
names = fieldnames(grid);
if isempty(names), best = base; T = table(); return; end
vals = cellfun(@(f) to_cell(grid.(f)), names, 'UniformOutput', false);
sz = cellfun(@numel, vals); nComb = prod(sz);
auc = nan(nComb, 1);
for c = 1:nComb
    p = base; sub = cell(1, numel(names)); [sub{:}] = ind2sub(sz(:)', c);
    for j = 1:numel(names), p.(names{j}) = vals{j}{sub{j}}; end
    try
        mdl = M.train(Zfit, [], p, struct('final', false, 'task', M.task, 'groups', []));
        R = mpm_roc(logical(ylab), M.predict(mdl, Zlab));
        auc(c) = R.auc;
    catch err
        log.add('  combination %d failed: %s', c, err.message);
    end
end
if all(isnan(auc)), error('mpm:tune', 'All tuning combinations failed.'); end
T = table();
for j = 1:numel(names)
    col = cell(nComb, 1);
    for c = 1:nComb
        sub = cell(1, numel(names)); [sub{:}] = ind2sub(sz(:)', c); col{c} = vals{j}{sub{j}};
    end
    if all(cellfun(@(v) isnumeric(v) && isscalar(v), col)), col = cell2mat(col);
    else, col = string(cellfun(@val_text, col, 'UniformOutput', false)); end
    T.(names{j}) = col;
end
T.sel_auc = auc;
[~, b] = max(auc);
T.best = false(nComb, 1); T.best(b) = true;
T.Properties.Description = 'AUC of training deposits vs. training background (model fitted without labels)';
best = base; sub = cell(1, numel(names)); [sub{:}] = ind2sub(sz(:)', b);
for j = 1:numel(names), best.(names{j}) = vals{j}{sub{j}}; end
end

function s = val_text(v)
if isnumeric(v) || islogical(v), s = mat2str(v); else, s = char(string(v)); end
end

function c = to_cell(v)
if iscell(v), c = v(:)'; elseif isstring(v), c = cellstr(v(:)'); else, c = num2cell(v(:)'); end
end
