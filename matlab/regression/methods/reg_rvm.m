function M = reg_rvm()
%REG_RVM Relevance vector regression for grade estimation (Tipping, 2001).
%
%   Sparse Bayesian kernel regression with a Gaussian likelihood; the noise level is estimated
%   from the data (beta), so there is no epsilon / C to tune. The response (model scale, e.g.
%   log10 grade) is z-scored inside the method.
%   Implementation: common/model/mpm_rvm_train.m.
%
%   Output     : predictive mean (model scale)
%   Uncertainty: predictive standard deviation sqrt(1/beta + phi' Sigma phi) (model scale) -
%                noise + weight uncertainty; grows away from the training samples.
%
%   Hyper-parameters:
%     Kernel       'gaussian' | 'linear'
%     KernelScale  MULTIPLIER of sqrt(P) (z-scored features)
%     Bias         include a constant basis function
%     MaxIter, Tol, AlphaMax, MaxBasis   see mpm_rvm_train (MaxBasis bounds the cost when there
%                  are thousands of samples)

M.key = 'rvm';
M.name = 'Relevance vector regression';
M.task = 'regression';
M.description = 'Sparse Bayesian kernel regression (own implementation) with predictive variance.';
M.defaults = struct('Kernel', 'gaussian', 'KernelScale', 1, 'Bias', true, 'MaxIter', 1000, ...
    'Tol', 1e-3, 'AlphaMax', 1e9, 'MaxBasis', 2000);
M.grid = struct('KernelScale', [0.25 0.5 1 2 4]);
M.has_uncertainty = true;
M.train = @fit_model;
M.predict = @predict_value;
M.native_importance = [];
M.relevant_samples = @(mdl) mdl.rv_index;
M.describe = @describe;
end

function mdl = fit_model(X, y, p, ctx) %#ok<INUSD>
y = double(y);
ym = mean(y); ys = std(y); if ys < eps, ys = 1; end
o = struct('likelihood', 'gaussian', 'kernel', p.Kernel, 'kernel_scale', p.KernelScale * sqrt(size(X, 2)), ...
    'bias', p.Bias, 'max_iter', p.MaxIter, 'tol', p.Tol, 'alpha_max', p.AlphaMax, 'max_basis', p.MaxBasis);
mdl = mpm_rvm_train(X, (y - ym) / ys, o);
mdl.y_mean = ym; mdl.y_sd = ys;
end

function [y, sd] = predict_value(mdl, X)
if nargout > 1
    [m, v] = mpm_rvm_predict(mdl, X);
    sd = sqrt(v + 1 / mdl.beta) * mdl.y_sd;
else
    m = mpm_rvm_predict(mdl, X);
end
y = m * mdl.y_sd + mdl.y_mean;
end

function txt = describe(mdl, p)
txt = sprintf(['%s kernel, %d relevance vectors of %d training samples (bias %d), KernelScale = %.4g ', ...
    '(x%.3g sqrt(P)), noise sd = %.3g (model scale), %d iterations%s'], mdl.opts.kernel, numel(mdl.rv_index), ...
    mdl.n_train, mdl.bias, mdl.opts.kernel_scale, p.KernelScale, mdl.y_sd / sqrt(mdl.beta), mdl.iterations, ...
    conv_text(mdl.converged));
end

function s = conv_text(c)
if c, s = ''; else, s = ' (not converged - raise MaxIter)'; end
end
