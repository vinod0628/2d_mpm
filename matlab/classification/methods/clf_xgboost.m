function M = clf_xgboost()
%CLF_XGBOOST XGBoost (regularised gradient tree boosting) for prospectivity mapping.
%
%   Own MATLAB implementation of Chen & Guestrin (2016) - see common/model/mpm_xgboost_train.m:
%   logistic loss, second-order split gain, exact greedy splits, depth-wise trees with gamma
%   pruning, L2/L1 leaf regularisation, shrinkage, row / column subsampling.
%   Output: probability sigmoid(F). Native importance: total split gain per feature.
%
%   Hyper-parameters (library name in brackets)
%     NumRounds (n_estimators), Eta (learning_rate), MaxDepth (max_depth), MinChildWeight
%     (min_child_weight), Lambda (reg_lambda), Alpha (reg_alpha), Gamma (gamma), Subsample
%     (subsample), ColsampleByTree (colsample_bytree), TreeMethod 'exact' | 'approx', MaxBin,
%     BalanceClasses (instance weights so deposits and background weigh equally)
%   Library defaults are eta 0.3 / max_depth 6; for MPM-size training sets (tens of deposits)
%   shallower trees with eta 0.1 are the default here, and depth x rounds are tuned by CV.
%
%   Reference: Chen, T., Guestrin, C., 2016. XGBoost: a scalable tree boosting system. KDD '16.

M.key = 'xgboost';
M.name = 'XGBoost';
M.task = 'classification';
M.description = 'Regularised gradient tree boosting (own implementation of Chen & Guestrin 2016).';
M.defaults = struct('NumRounds', 150, 'Eta', 0.1, 'MaxDepth', 3, 'MinChildWeight', 1, 'Lambda', 1, ...
    'Alpha', 0, 'Gamma', 0, 'Subsample', 1, 'ColsampleByTree', 1, 'TreeMethod', 'exact', 'MaxBin', 256, ...
    'BalanceClasses', true);
M.grid = struct('MaxDepth', [2 3 4], 'NumRounds', [25 50 100 200]);
M.score_is_probability = true;
M.has_uncertainty = false;
M.train = @fit_model;
M.predict = @predict_score;
M.native_importance = @native_importance;
M.describe = @describe;
end

function mdl = fit_model(X, y, p, ctx) %#ok<INUSD>
y = double(logical(y)); w = ones(size(y));
if p.BalanceClasses
    w(y == 1) = numel(y) / (2 * max(sum(y == 1), 1)); w(y == 0) = numel(y) / (2 * max(sum(y == 0), 1));
end
mdl = mpm_xgboost_train(X, y, struct('loss', 'logistic', 'n_rounds', p.NumRounds, 'eta', p.Eta, ...
    'max_depth', p.MaxDepth, 'min_child_weight', p.MinChildWeight, 'lambda', p.Lambda, 'alpha', p.Alpha, ...
    'gamma', p.Gamma, 'subsample', p.Subsample, 'colsample_bytree', p.ColsampleByTree, ...
    'tree_method', p.TreeMethod, 'max_bin', p.MaxBin, 'weights', w));
end

function s = predict_score(mdl, X)
s = mpm_gb_loss('link', 'logistic', [], mpm_gb_predict(mdl, X));
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
