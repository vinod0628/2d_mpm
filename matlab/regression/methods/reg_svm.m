function M = reg_svm()
%REG_SVM Support vector regression (SVR) for grade estimation.
%
%   Model : fitrsvm (Statistics and Machine Learning Toolbox), Gaussian kernel by default.
%   The response (already on the model scale, e.g. log10 grade) is z-scored inside the method,
%   so BoxConstraint and Epsilon do not depend on the grade units.
%
%   Hyper-parameters:
%     KernelFunction  'gaussian' | 'linear' | 'polynomial'
%     BoxConstraint   C on the z-scored response
%     KernelScale     MULTIPLIER of sqrt(P) (features are z-scored; 1 = gamma 1/P heuristic)
%     Epsilon         half-width of the insensitive tube, in response standard deviations
%     PolynomialOrder used with 'polynomial'

M.key = 'svm';
M.name = 'Support vector regression';
M.task = 'regression';
M.description = 'Epsilon-SVR (fitrsvm) on z-scored features and response.';
M.defaults = struct('KernelFunction', 'gaussian', 'BoxConstraint', 1, 'KernelScale', 1, ...
    'Epsilon', 0.1, 'PolynomialOrder', 3);
M.grid = struct('BoxConstraint', [0.3 1 3 10], 'KernelScale', [0.5 1 2 4], 'Epsilon', [0.05 0.2]);
M.has_uncertainty = false;
M.train = @fit_model;
M.predict = @predict_value;
M.native_importance = [];
M.describe = @describe;
end

function mdl = fit_model(X, y, p, ctx) %#ok<INUSD>
X = double(X); y = double(y);
mdl.mu = mean(y); mdl.sd = std(y); if mdl.sd < eps, mdl.sd = 1; end
args = {'KernelFunction', p.KernelFunction, 'BoxConstraint', p.BoxConstraint, ...
    'Epsilon', p.Epsilon, 'Standardize', false};
switch lower(p.KernelFunction)
    case {'gaussian', 'rbf'}, args = [args, {'KernelScale', p.KernelScale * sqrt(size(X, 2))}];
    case 'polynomial', args = [args, {'PolynomialOrder', p.PolynomialOrder}];
end
mdl.svr = compact(fitrsvm(X, (y - mdl.mu) / mdl.sd, args{:}));
end

function y = predict_value(mdl, X)
y = predict(mdl.svr, double(X)) * mdl.sd + mdl.mu;
end

function txt = describe(mdl, p)
txt = sprintf('%s kernel, %d support vectors, KernelScale = %.4g (x%.3g sqrt(P)), C = %.4g, epsilon = %.3g sd', ...
    mdl.svr.KernelParameters.Function, size(mdl.svr.SupportVectors, 1), mdl.svr.KernelParameters.Scale, ...
    p.KernelScale, p.BoxConstraint, p.Epsilon);
end
