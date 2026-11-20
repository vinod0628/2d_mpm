function M = reg_template()
%REG_TEMPLATE Template for a new regression (grade estimation) method.
%
%   HOW TO ADD A METHOD (e.g. Gaussian process, boosted trees, MLP, kriging with external drift):
%     1. Copy this file to  regression/methods/reg_<key>.m  and rename the function.
%     2. Fill in key/name/defaults/grid and the fit_model / predict_value local functions.
%     3. Copy regression/run_svm_regression.m to run_<key>_regression.m and change only
%        M = reg_svm();  ->  M = reg_<key>();
%   Data loading, sample-to-cell matching, log transform, split, CV tuning, grade map,
%   metrics (R2, RMSE, MAE, cut-off ROC/AUC), saving and the results GUI are shared.
%
%   CONTRACT (fields read by MPM_REGRESSION_RUN)
%     key, name, task='regression', description
%     defaults, grid        as for classification (grid = struct() disables tuning)
%     has_uncertainty       true if predict_value returns [y, unc] (unc on the model scale)
%     train(X, y, p, ctx)   X: n x P z-scored single, y: n x 1 double on the MODEL scale (e.g.
%                           log10 grade), ctx.final false inside CV. Returns the model.
%     predict(mdl, X)       n x 1 prediction on the model scale [, n x 1 uncertainty]
%     native_importance     [] or @(mdl, featureNames) -> table(feature, importance)
%     describe              [] or @(mdl, p) -> char
%
%   NOTE: local functions must not be called "predict" / "train" - they would shadow the
%   toolbox functions inside this file.
%
%   The example body is a ridge linear regression (fitrlinear), so a copy runs as-is.

M.key = 'template';
M.name = 'Template (ridge regression example)';
M.task = 'regression';
M.description = 'Skeleton for new regression methods; example body = ridge regression.';
M.defaults = struct('Lambda', 1e-3);
M.grid = struct('Lambda', [1e-4 1e-3 1e-2 1e-1 1]);
M.has_uncertainty = false;
M.train = @fit_model;
M.predict = @predict_value;
M.native_importance = @native_importance;
M.describe = [];
end

function mdl = fit_model(X, y, p, ctx) %#ok<INUSD>
mdl = fitrlinear(double(X), double(y), 'Learner', 'leastsquares', 'Regularization', 'ridge', ...
    'Lambda', p.Lambda);
end

function y = predict_value(mdl, X)
y = predict(mdl, double(X));
end

function T = native_importance(mdl, names)
T = table(string(names(:)), abs(mdl.Beta(:)), 'VariableNames', {'feature', 'importance'});
T = sortrows(T, 'importance', 'descend');
T.Properties.Description = '|coefficient| on z-scored features';
end
