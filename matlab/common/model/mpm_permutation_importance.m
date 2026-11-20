function T = mpm_permutation_importance(M, mdl, X, y, source, layerNames, metric, nRep, rs)
%MPM_PERMUTATION_IMPORTANCE Model-agnostic importance of each evidence LAYER.
%
%   T = mpm_permutation_importance(M, mdl, X, y, source, layerNames, metric, nRep, rs)
%
%   All feature columns of one layer (e.g. every one-hot column of a geology map) are permuted
%   together, so the importance is reported per evidence layer. Importance = drop of the score
%   (AUC / R^2) or increase of the error (RMSE) relative to the unpermuted data.
base = score_of(metric, y, M.predict(mdl, X));
L = numel(layerNames);
imp = nan(L, nRep);
patch = isfield(M, 'input') && strcmpi(M.input, 'patch');   % X = cell indices; permute whole
if patch, src = mpm_patch_source.active(); end              % windows of a layer between samples
for k = 1:L
    cols = find(source == k);
    if isempty(cols), continue; end
    for r = 1:nRep
        if patch
            src.set_permutation(cols, X, X(randperm(rs, size(X, 1))));
            s = score_of(metric, y, M.predict(mdl, X));
            src.clear_permutation();
            if strcmp(metric, 'rmse'), imp(k, r) = s - base; else, imp(k, r) = base - s; end
            continue;
        end
        Xp = X;
        Xp(:, cols) = X(randperm(rs, size(X, 1)), cols);
        s = score_of(metric, y, M.predict(mdl, Xp));
        if strcmp(metric, 'rmse'), imp(k, r) = s - base; else, imp(k, r) = base - s; end
    end
end
T = table(string(layerNames(:)), mean(imp, 2), std(imp, 0, 2), ...
    'VariableNames', {'layer', 'importance', 'importance_sd'});
T = sortrows(T, 'importance', 'descend');
T.Properties.Description = sprintf('Permutation importance (%s, baseline %.4f, %d repeats)', metric, base, nRep);
end

function s = score_of(metric, y, p)
switch metric
    case 'auc',  C = mpm_roc(logical(y), p); s = C.auc;
    case 'rmse', s = sqrt(mean((y - p).^2, 'omitnan'));
    case 'r2',   s = 1 - sum((y - p).^2, 'omitnan') / sum((y - mean(y)).^2);
end
end
