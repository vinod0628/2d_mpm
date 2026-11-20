function [best, T, cvPred] = mpm_tune_cv(M, X, y, base, grid, k, metric, rs, groups, log)
%MPM_TUNE_CV Grid-search hyper-parameters by k-fold cross-validation (generic for all methods).
%
%   [best, T] = mpm_tune_cv(M, X, y, base, grid, k, metric, rs, groups, log)
%
%   M       method struct (see classification/methods/clf_template.m)
%   base    default parameter struct; grid = struct of candidate values per parameter,
%           e.g. struct('BoxConstraint', [0.1 1 10], 'KernelScale', [0.5 1 2])
%   metric  'auc' (maximise) | 'rmse' (minimise) | 'r2' (maximise) | 'mae' (minimise)
%   groups  optional group id per sample (grouped folds), [] = none
%   best    BASE with the best grid values filled in;  T = table of all combinations + CV score
%   cvPred  out-of-fold predictions of the best combination
if nargin < 9, groups = []; end
if nargin < 10 || isempty(log), log = mpm_Logger(true); end
names = fieldnames(grid);
if isempty(names)
    best = base; T = table(); cvPred = []; return;
end
vals = cellfun(@(f) to_cell(grid.(f)), names, 'UniformOutput', false);
sz = cellfun(@numel, vals);
nComb = prod(sz);
isClf = strcmp(metric, 'auc');
if isClf, fold = mpm_kfold(size(X, 1), k, rs, y, groups);
else, fold = mpm_kfold(size(X, 1), k, rs, [], groups); end

scores = nan(nComb, 1); sds = nan(nComb, 1);
preds = cell(nComb, 1);
for c = 1:nComb
    sub = cell(1, numel(names));
    [sub{:}] = ind2sub(sz(:)', c);
    p = base;
    for j = 1:numel(names), p.(names{j}) = vals{j}{sub{j}}; end
    oof = nan(size(X, 1), 1); per = nan(k, 1);
    try
        for f = 1:k
            tr = fold ~= f; te = fold == f;
            if ~any(te), continue; end
            if isClf && (all(y(tr)) || ~any(y(tr))), continue; end
            ctx = struct('final', false, 'task', M.task);
            mdl = M.train(X(tr, :), y(tr), p, ctx);
            oof(te) = M.predict(mdl, X(te, :));
            per(f) = score_of(metric, y(te), oof(te));
        end
        scores(c) = score_of(metric, y, oof);               % pooled out-of-fold score
        sds(c) = std(per, 'omitnan');
        preds{c} = oof;
        log.add('  [%d/%d] %s: CV %s = %.4f', c, nComb, mpm_param_text(p, names), metric, scores(c));
    catch err
        log.add('  CV combination %d failed: %s', c, err.message);
    end
end
T = table();
for j = 1:numel(names)
    col = cell(nComb, 1);
    for c = 1:nComb
        sub = cell(1, numel(names)); [sub{:}] = ind2sub(sz(:)', c);
        col{c} = vals{j}{sub{j}};
    end
    if all(cellfun(@(v) isnumeric(v) && isscalar(v), col)), col = cell2mat(col);
    else, col = string(cellfun(@val_text, col, 'UniformOutput', false)); end
    T.(names{j}) = col;
end
T.(['cv_' metric]) = scores;
T.(['cv_' metric '_fold_sd']) = sds;
if any(strcmp(metric, {'rmse', 'mae'})), [~, b] = min(scores); else, [~, b] = max(scores); end
if all(isnan(scores)), error('mpm:tune', 'All CV combinations failed.'); end
best = base;
sub = cell(1, numel(names)); [sub{:}] = ind2sub(sz(:)', b);
for j = 1:numel(names), best.(names{j}) = vals{j}{sub{j}}; end
T.best = false(nComb, 1); T.best(b) = true;
cvPred = preds{b};
end

function s = val_text(v)
if isnumeric(v) || islogical(v), s = mat2str(v); else, s = char(string(v)); end
end

function c = to_cell(v)
if iscell(v), c = v(:)'; elseif isstring(v), c = cellstr(v(:)'); else, c = num2cell(v(:)'); end
end

function s = score_of(metric, y, p)
ok = isfinite(p);
switch metric
    case 'auc',  C = mpm_roc(logical(y(ok)), p(ok)); s = C.auc;
    case 'rmse', s = sqrt(mean((y(ok) - p(ok)).^2));
    case 'mae',  s = mean(abs(y(ok) - p(ok)));
    case 'r2',   s = 1 - sum((y(ok) - p(ok)).^2) / sum((y(ok) - mean(y(ok))).^2);
    otherwise, error('mpm:tune', 'Unknown metric %s', metric);
end
end
