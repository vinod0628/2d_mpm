function M = clf_rvm()
%CLF_RVM Relevance vector machine classifier for prospectivity mapping (Tipping, 2001).
%
%   Sparse Bayesian kernel classifier: Bernoulli likelihood, Laplace approximation, one ARD
%   precision per kernel basis function. Most basis functions are pruned during training; the
%   few training samples that survive are the relevance vectors (shown on the GUI map).
%   Implementation: common/model/mpm_rvm_train.m (no toolbox RVM exists).
%
%   Output   : moderated posterior probability  p = sigmoid(m / sqrt(1 + pi*s2/8))  (MacKay 1992),
%              m, s2 = mean and variance of the latent function -> probabilities are pulled
%              towards 0.5 where the model is uncertain (far from training data).
%   Uncertainty map: approximate standard deviation of the probability, p(1-p)*sqrt(s2).
%
%   Hyper-parameters (no C / epsilon to tune - the ARD prior sets the sparsity):
%     Kernel          'gaussian' | 'linear'
%     KernelScale     MULTIPLIER of sqrt(P) (z-scored features; same convention as clf_svm)
%     Bias            include a constant basis function
%     BalanceClasses  weight the likelihood so both classes count equally
%     Moderated       true: MacKay-moderated probabilities; false: sigmoid(m)
%     MaxIter, Tol, AlphaMax, MaxBasis   see mpm_rvm_train
%
%   References: Tipping, M.E., 2001. Sparse Bayesian learning and the relevance vector machine.
%   JMLR 1, 211-244.  MacKay, D.J.C., 1992. The evidence framework applied to classification
%   networks. Neural Computation 4, 720-736.

M.key = 'rvm';
M.name = 'Relevance vector machine';
M.task = 'classification';
M.description = 'Sparse Bayesian kernel classifier (own implementation, Laplace approximation).';
M.defaults = struct('Kernel', 'gaussian', 'KernelScale', 1, 'Bias', true, 'BalanceClasses', true, ...
    'Moderated', true, 'MaxIter', 1000, 'Tol', 1e-3, 'AlphaMax', 1e9, 'MaxBasis', 2000);
M.grid = struct('KernelScale', [0.25 0.5 1 2 4]);
M.score_is_probability = true;
M.has_uncertainty = true;
M.train = @fit_model;
M.predict = @predict_score;
M.native_importance = [];
M.relevant_samples = @(mdl) mdl.rv_index;    % training rows kept as relevance vectors
M.describe = @describe;
end

function mdl = fit_model(X, y, p, ctx) %#ok<INUSD>
y = double(logical(y));
w = ones(size(y));
if p.BalanceClasses
    n1 = sum(y == 1); n0 = sum(y == 0);
    w(y == 1) = numel(y) / (2 * max(n1, 1)); w(y == 0) = numel(y) / (2 * max(n0, 1));
end
o = struct('likelihood', 'bernoulli', 'kernel', p.Kernel, 'kernel_scale', p.KernelScale * sqrt(size(X, 2)), ...
    'bias', p.Bias, 'max_iter', p.MaxIter, 'tol', p.Tol, 'alpha_max', p.AlphaMax, ...
    'max_basis', p.MaxBasis, 'weights', w);
mdl = mpm_rvm_train(X, y, o);
mdl.moderated = p.Moderated;
end

function [s, u] = predict_score(mdl, X)
if nargout > 1 || mdl.moderated
    [m, v] = mpm_rvm_predict(mdl, X);
else
    m = mpm_rvm_predict(mdl, X); v = zeros(size(m));
end
if mdl.moderated, k = 1 ./ sqrt(1 + pi * v / 8); else, k = 1; end
s = 1 ./ (1 + exp(-k .* m));
if nargout > 1
    q = 1 ./ (1 + exp(-m));
    u = q .* (1 - q) .* sqrt(v);
end
end

function txt = describe(mdl, p)
txt = sprintf('%s kernel, %d relevance vectors of %d training samples (bias %d), KernelScale = %.4g (x%.3g sqrt(P)), %d iterations%s', ...
    mdl.opts.kernel, numel(mdl.rv_index), mdl.n_train, mdl.bias, mdl.opts.kernel_scale, p.KernelScale, ...
    mdl.iterations, conv_text(mdl.converged));
end

function s = conv_text(c)
if c, s = ''; else, s = ' (not converged - raise MaxIter)'; end
end
