function M = reg_xgboost()
%REG_XGBOOST XGBoost regression for grade estimation (own implementation of Chen & Guestrin 2016).
%
%   Squared-error loss (g = F - y, h = 1, so min_child_weight = minimum samples per leaf) on the
%   model scale (e.g. log10 grade). Other settings and hyper-parameters as clf_xgboost
%   (BalanceClasses not applicable). See common/model/mpm_xgboost_train.m.

M.key = 'xgboost';
M.name = 'XGBoost regression';
M.task = 'regression';
M.description = 'Regularised gradient tree boosting, squared error (own implementation of Chen & Guestrin 2016).';
M.defaults = struct('NumRounds', 150, 'Eta', 0.1, 'MaxDepth', 3, 'MinChildWeight', 1, 'Lambda', 1, ...
    'Alpha', 0, 'Gamma', 0, 'Subsample', 1, 'ColsampleByTree', 1, 'TreeMethod', 'exact', 'MaxBin', 256);
M.grid = struct('MaxDepth', [2 3 4], 'NumRounds', [25 50 100 200]);
M.has_uncertainty = false;
M.train = @fit_model;
M.predict = @predict_value;
M.native_importance = @native_importance;
M.describe = @describe;
end

function mdl = fit_model(X, y, p, ctx) %#ok<INUSD>
mdl = mpm_xgboost_train(X, y, struct('loss', 'squared', 'n_rounds', p.NumRounds, 'eta', p.Eta, ...
    'max_depth', p.MaxDepth, 'min_child_weight', p.MinChildWeight, 'lambda', p.Lambda, 'alpha', p.Alpha, ...
    'gamma', p.Gamma, 'subsample', p.Subsample, 'colsample_bytree', p.ColsampleByTree, ...
    'tree_method', p.TreeMethod, 'max_bin', p.MaxBin));
end

function y = predict_value(mdl, X)
y = mpm_gb_predict(mdl, X);
end

function T = native_importance(mdl, names)
T = table(string(names(:)), mdl.importance_gain(:), mdl.importance_split(:), ...
    'VariableNames', {'feature', 'importance', 'splits'});
T = sortrows(T, 'importance', 'descend');
T.Properties.Description = 'XGBoost total split gain';
end

function txt = describe(mdl, p)
txt = sprintf('%d trees, max depth %d, eta %.3g, lambda %.3g, gamma %.3g, %s splits, mean %.1f leaves/tree', ...
    numel(mdl.trees), p.MaxDepth, p.Eta, p.Lambda, p.Gamma, p.TreeMethod, mean(mdl.n_leaves));
end
