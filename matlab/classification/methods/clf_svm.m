function M = clf_svm()
%CLF_SVM Support vector machine classifier for prospectivity mapping (Zuo & Carranza, 2011).
%
%   M = clf_svm()  returns the method description used by MPM_CLASSIFICATION_RUN.
%
%   Model : fitcsvm (Statistics and Machine Learning Toolbox), RBF kernel by default.
%   Output: posterior probability of the deposit class (Platt sigmoid, fitPosterior) for the
%           final model; raw SVM scores during CV tuning (AUC only needs a ranking).
%
%   Hyper-parameters (cfg.params overrides, cfg.tune.grid replaces M.grid):
%     KernelFunction  'rbf' | 'linear' | 'polynomial'
%     BoxConstraint   C, soft-margin penalty
%     KernelScale     MULTIPLIER of sqrt(P) (P = number of features). Features are z-scored, so
%                     1 = the usual gamma = 1/P heuristic; the absolute scale is stored in the model.
%     PolynomialOrder used with 'polynomial'
%     BalanceClasses  true -> uniform class prior (useful when negatives outnumber deposits)
%     Calibrate       true -> fitPosterior for probabilities in [0, 1]
%
%   Reference: Zuo, R., Carranza, E.J.M., 2011. Support vector machine: a tool for mapping mineral
%   prospectivity. Computers & Geosciences 37, 1967-1975.

M.key = 'svm';
M.name = 'Support vector machine';
M.task = 'classification';
M.description = 'Kernel SVM (fitcsvm) with Platt-scaled probabilities.';
M.defaults = struct('KernelFunction', 'rbf', 'BoxConstraint', 1, 'KernelScale', 1, ...
    'PolynomialOrder', 3, 'BalanceClasses', true, 'Calibrate', true);
M.grid = struct('BoxConstraint', [0.1 1 10 100], 'KernelScale', [0.25 0.5 1 2 4]);
M.score_is_probability = true;
M.has_uncertainty = false;
M.train = @fit_model;
M.predict = @predict_score;
M.native_importance = [];          % none for kernel SVM -> permutation importance only
M.describe = @describe;
end

function mdl = fit_model(X, y, p, ctx)
X = double(X);
args = {'KernelFunction', p.KernelFunction, 'BoxConstraint', p.BoxConstraint, ...
    'ClassNames', [false; true], 'Standardize', false, 'CacheSize', 'maximal'};
switch lower(p.KernelFunction)
    case {'rbf', 'gaussian'}, args = [args, {'KernelScale', p.KernelScale * sqrt(size(X, 2))}];
    case 'polynomial', args = [args, {'PolynomialOrder', p.PolynomialOrder}];
end
if p.BalanceClasses, args = [args, {'Prior', 'uniform'}]; end
mdl = fitcsvm(X, logical(y), args{:});
if ctx.final && p.Calibrate
    mdl = fitPosterior(mdl);
end
mdl = compact(mdl);
end

function s = predict_score(mdl, X)
[~, sc] = predict(mdl, double(X));
s = sc(:, 2);
end

function txt = describe(mdl, p)
txt = sprintf('%s kernel, %d support vectors, KernelScale = %.4g (x%.3g sqrt(P)), BoxConstraint = %.4g', ...
    mdl.KernelParameters.Function, size(mdl.SupportVectors, 1), mdl.KernelParameters.Scale, ...
    p.KernelScale, p.BoxConstraint);
end
