function M = clf_lightgbm()
%CLF_LIGHTGBM LightGBM (histogram, leaf-wise gradient boosting with GOSS / EFB) for prospectivity.
%
%   Own MATLAB implementation of Ke et al. (2017) - see common/model/mpm_lightgbm_train.m:
%   histogram split finding with histogram subtraction, leaf-wise (best-first) trees, Exclusive
%   Feature Bundling (one-hot geology classes are bundled), optional Gradient-based One-Side
%   Sampling (Boosting = 'goss'), logistic loss with second-order leaf weights.
%   With Boosting = 'gbdt' and EFB = false this is also the algorithm of scikit-learn's
%   HistGradientBoosting (which follows LightGBM), so no separate HGB method is needed.
%
%   Hyper-parameters (library name in brackets)
%     NumRounds (num_iterations), LearningRate, NumLeaves (num_leaves), MaxDepth (max_depth, -1 =
%     none), MinDataInLeaf (min_data_in_leaf), MinSumHessian (min_sum_hessian_in_leaf), LambdaL1,
%     LambdaL2, MinGainToSplit, MaxBin, FeatureFraction, BaggingFraction / BaggingFreq,
%     Boosting 'gbdt' | 'goss', TopRate / OtherRate (GOSS a, b), EFB, MaxConflictRate,
%     BalanceClasses
%   Library default min_data_in_leaf = 20 and num_leaves = 31 assume thousands of samples; with
%   tens of deposits the defaults here are 5 and 8, and leaves x rounds are tuned by CV.
%   GOSS is meant for large training sets (it trains each tree on ~30 % of the samples).
%
%   Reference: Ke, G. et al., 2017. LightGBM: a highly efficient gradient boosting decision tree.
%   NeurIPS 30.

M.key = 'lightgbm';
M.name = 'LightGBM';
M.task = 'classification';
M.description = 'Histogram leaf-wise gradient boosting with EFB/GOSS (own implementation of Ke et al. 2017).';
M.defaults = struct('NumRounds', 150, 'LearningRate', 0.1, 'NumLeaves', 8, 'MaxDepth', -1, ...
    'MinDataInLeaf', 5, 'MinSumHessian', 1e-3, 'LambdaL1', 0, 'LambdaL2', 0, 'MinGainToSplit', 0, ...
    'MaxBin', 255, 'FeatureFraction', 1, 'BaggingFraction', 1, 'BaggingFreq', 0, 'Boosting', 'gbdt', ...
    'TopRate', 0.2, 'OtherRate', 0.1, 'EFB', true, 'MaxConflictRate', 0, 'BalanceClasses', true);
M.grid = struct('NumLeaves', [4 8 16], 'NumRounds', [25 50 100 200]);
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
mdl = mpm_lightgbm_train(X, y, lgb_opts(p, 'logistic', w));
end

function o = lgb_opts(p, loss, w)
o = struct('loss', loss, 'n_rounds', p.NumRounds, 'learning_rate', p.LearningRate, ...
    'num_leaves', p.NumLeaves, 'max_depth', p.MaxDepth, 'min_data_in_leaf', p.MinDataInLeaf, ...
    'min_sum_hessian', p.MinSumHessian, 'lambda_l1', p.LambdaL1, 'lambda_l2', p.LambdaL2, ...
    'min_gain_to_split', p.MinGainToSplit, 'max_bin', p.MaxBin, 'feature_fraction', p.FeatureFraction, ...
    'bagging_fraction', p.BaggingFraction, 'bagging_freq', p.BaggingFreq, 'boosting', p.Boosting, ...
    'top_rate', p.TopRate, 'other_rate', p.OtherRate, 'efb', p.EFB, 'max_conflict_rate', p.MaxConflictRate, ...
    'weights', w);
end

function s = predict_score(mdl, X)
s = mpm_gb_loss('link', 'logistic', [], mpm_gb_predict(mdl, X));
end

function T = native_importance(mdl, names)
T = table(string(names(:)), mdl.importance_gain(:), mdl.importance_split(:), ...
    'VariableNames', {'feature', 'importance', 'splits'});
T = sortrows(T, 'importance', 'descend');
T.Properties.Description = 'LightGBM total split gain';
end

function txt = describe(mdl, p)
txt = sprintf('%d trees, %d leaves max, lr %.3g, %s, %d feature bundles (EFB %d)', numel(mdl.trees), ...
    p.NumLeaves, p.LearningRate, p.Boosting, mdl.n_bundles, p.EFB);
end
