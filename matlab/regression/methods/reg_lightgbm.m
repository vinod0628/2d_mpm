function M = reg_lightgbm()
%REG_LIGHTGBM LightGBM regression for grade estimation (own implementation of Ke et al. 2017).
%
%   Squared-error loss on the model scale (e.g. log10 grade); histogram leaf-wise trees with EFB
%   and optional GOSS. Hyper-parameters as clf_lightgbm (BalanceClasses not applicable).
%   See common/model/mpm_lightgbm_train.m.

M.key = 'lightgbm';
M.name = 'LightGBM regression';
M.task = 'regression';
M.description = 'Histogram leaf-wise gradient boosting with EFB/GOSS, squared error (own implementation of Ke et al. 2017).';
M.defaults = struct('NumRounds', 150, 'LearningRate', 0.1, 'NumLeaves', 8, 'MaxDepth', -1, ...
    'MinDataInLeaf', 5, 'MinSumHessian', 1e-3, 'LambdaL1', 0, 'LambdaL2', 0, 'MinGainToSplit', 0, ...
    'MaxBin', 255, 'FeatureFraction', 1, 'BaggingFraction', 1, 'BaggingFreq', 0, 'Boosting', 'gbdt', ...
    'TopRate', 0.2, 'OtherRate', 0.1, 'EFB', true, 'MaxConflictRate', 0);
M.grid = struct('NumLeaves', [4 8 16], 'NumRounds', [25 50 100 200]);
M.has_uncertainty = false;
M.train = @fit_model;
M.predict = @predict_value;
M.native_importance = @native_importance;
M.describe = @describe;
end

function mdl = fit_model(X, y, p, ctx) %#ok<INUSD>
mdl = mpm_lightgbm_train(X, y, struct('loss', 'squared', 'n_rounds', p.NumRounds, ...
    'learning_rate', p.LearningRate, 'num_leaves', p.NumLeaves, 'max_depth', p.MaxDepth, ...
    'min_data_in_leaf', p.MinDataInLeaf, 'min_sum_hessian', p.MinSumHessian, 'lambda_l1', p.LambdaL1, ...
    'lambda_l2', p.LambdaL2, 'min_gain_to_split', p.MinGainToSplit, 'max_bin', p.MaxBin, ...
    'feature_fraction', p.FeatureFraction, 'bagging_fraction', p.BaggingFraction, ...
    'bagging_freq', p.BaggingFreq, 'boosting', p.Boosting, 'top_rate', p.TopRate, ...
    'other_rate', p.OtherRate, 'efb', p.EFB, 'max_conflict_rate', p.MaxConflictRate));
end

function y = predict_value(mdl, X)
y = mpm_gb_predict(mdl, X);
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
