function mdl = mpm_rvm_train(X, t, opts)
%MPM_RVM_TRAIN Relevance vector machine = sparse Bayesian learning (Tipping, 2001).
%
%   mdl = mpm_rvm_train(X, t, opts)
%
%   X     N x P features (z-scored), t: N x 1 targets
%   opts.likelihood   'gaussian'  (regression, t real)  |  'bernoulli'  (classification, t in {0,1})
%   opts.kernel       'gaussian' (default) | 'linear'
%   opts.kernel_scale  s of the Gaussian kernel k(x,z) = exp(-||x - z||^2 / s^2)  (same as fitcsvm)
%   opts.bias          true: add a constant basis function
%   opts.max_iter      outer (hyper-parameter) iterations, default 1000
%   opts.tol           convergence: max |delta log(alpha)| of the active set, default 1e-3
%   opts.fit_tol       ... or the fitted values change by less than fit_tol (relative) for 5
%                      iterations in a row (alphas creeping to infinity no longer matter), 1e-6
%   opts.alpha_max     prune a basis function when its precision alpha exceeds this, default 1e9
%   opts.alpha_init    initial alpha, default 1
%   opts.max_basis     use at most this many training samples as kernel centres (random subset),
%                      bounds the O(M^3) cost for large training sets, default 2000
%   opts.weights       N x 1 likelihood weights (bernoulli; class balancing), default ones
%
%   Model: t = Phi(x) w (+ noise),  w_i ~ N(0, 1/alpha_i) with one precision per basis function
%   (ARD prior). Type-II maximum likelihood re-estimation (Tipping 2001, eqs. 16, 18):
%       gamma_i = 1 - alpha_i * Sigma_ii,   alpha_i <- gamma_i / mu_i^2,
%       beta <- (N - sum gamma) / ||t - Phi mu||^2                      (gaussian)
%   Bernoulli likelihood: for fixed alpha the posterior mode and covariance come from a Laplace
%   approximation (Newton / IRLS with step halving). Most alpha_i diverge -> those basis
%   functions are pruned; the surviving training samples are the "relevance vectors".
%
%   mdl fields: centres (RV feature vectors), rv_index (row in X of each RV), bias (true if the
%   bias basis survived), scales (column normalisation), mu, Sigma, alpha, beta (gaussian noise
%   precision), opts, iterations, converged, n_train.
%   Prediction: MPM_RVM_PREDICT.

d = struct('likelihood', 'gaussian', 'kernel', 'gaussian', 'kernel_scale', 1, 'bias', true, ...
    'max_iter', 1000, 'tol', 1e-3, 'alpha_max', 1e9, 'alpha_init', 1, 'max_basis', 2000, ...
    'weights', [], 'irls_iter', 25, 'fit_tol', 1e-6);
opts = mpm_merge(d, opts);
X = double(X); t = double(t(:)); N = size(X, 1);
isBern = strcmpi(opts.likelihood, 'bernoulli');
c = ones(N, 1); if ~isempty(opts.weights), c = double(opts.weights(:)); end

% ---------------------------------------------------------------- basis functions
cidx = (1:N)';
if N > opts.max_basis, cidx = sort(randperm(N, opts.max_basis))'; end
Phi = mpm_rvm_basis(X, X(cidx, :), opts.kernel, opts.kernel_scale, opts.bias);
rvIndex = cidx; if opts.bias, rvIndex = [NaN; cidx]; end     % NaN = bias column
scales = sqrt(sum(Phi.^2, 1))'; scales(scales < eps) = 1;    % unit-norm columns (stability)
Phi = Phi ./ scales';
M = size(Phi, 2);

alpha = opts.alpha_init * ones(M, 1);
active = true(M, 1);
w = zeros(M, 1);
if isBern, beta = NaN; else, beta = 1 / max(0.1 * var(t), 1e-6); end
converged = false; prevFit = []; nStable = 0;
for it = 1:opts.max_iter
    P = Phi(:, active); a = alpha(active);
    if isBern
        [wa, Sigma] = laplace_mode(P, t, a, w(active), c, opts.irls_iter);
        w(:) = 0; w(active) = wa;
        mu = wa;
    else
        [mu, Sigma] = gauss_post(P, t, a, beta);
    end
    fit = P * mu;
    if ~isempty(prevFit) && norm(fit - prevFit) <= opts.fit_tol * max(norm(fit), eps)
        nStable = nStable + 1;
    else
        nStable = 0;
    end
    prevFit = fit;
    gamma = max(1 - a .* diag(Sigma), 0);
    anew = gamma ./ max(mu.^2, realmin);
    anew(~isfinite(anew)) = Inf;
    if ~isBern
        e = t - P * mu;
        beta = max(N - sum(gamma), 1e-6) / max(e' * e, eps);
    end
    dlog = max(abs(log(min(anew, opts.alpha_max)) - log(min(a, opts.alpha_max))));
    alpha(active) = anew;
    prune = active & alpha > opts.alpha_max;
    if all(prune(active))                                  % keep at least one basis function
        ia = find(active); [~, k] = min(alpha(ia)); prune(ia(k)) = false;
    end
    active(prune) = false;
    if (dlog < opts.tol && ~any(prune)) || nStable >= 5, converged = true; break; end
end

% ---------------------------------------------------------------- final posterior
P = Phi(:, active); a = alpha(active);
if isBern
    [mu, Sigma] = laplace_mode(P, t, a, w(active), c, opts.irls_iter);
else
    [mu, Sigma] = gauss_post(P, t, a, beta);
end
ia = find(active);
isBias = isnan(rvIndex(ia));
mdl.bias = any(isBias);
mdl.rv_index = rvIndex(ia(~isBias));
mdl.centres = X(mdl.rv_index, :);
% reorder so the bias (if kept) is the first weight, matching MPM_RVM_BASIS
order = [find(isBias); find(~isBias)];
mdl.mu = mu(order);
mdl.Sigma = Sigma(order, order);
mdl.alpha = a(order);
mdl.scales = scales(ia(order));
mdl.beta = beta;
mdl.opts = rmfield(opts, 'weights');
mdl.iterations = it;
mdl.converged = converged;
mdl.n_train = N;
mdl.n_candidates = M;
end

% =====================================================================================
function [mu, Sigma] = gauss_post(P, t, a, beta)
H = beta * (P' * P) + diag(a);
U = chol_safe(H);
Ui = U \ eye(size(U));
Sigma = Ui * Ui';
mu = beta * (Sigma * (P' * t));
end

function [w, Sigma] = laplace_mode(P, t, a, w, c, maxIt)
% posterior mode of w for a Bernoulli likelihood (weighted), Newton steps with step halving
lp = @(w) logpost(P, t, a, w, c);
L = lp(w);
for k = 1:maxIt
    z = P * w; y = sigm(z);
    g = P' * (c .* (t - y)) - a .* w;
    H = P' * (P .* (c .* y .* (1 - y))) + diag(a);
    U = chol_safe(H);
    dw = U \ (U' \ g);
    step = 1;
    while true
        wn = w + step * dw; Ln = lp(wn);
        if Ln >= L - 1e-12 || step < 1e-8, break; end
        step = step / 2;
    end
    w = wn; Lold = L; L = Ln;
    if norm(step * dw) < 1e-8 * max(1, norm(w)) || abs(L - Lold) < 1e-10 * max(1, abs(L)), break; end
end
y = sigm(P * w);
H = P' * (P .* (c .* y .* (1 - y))) + diag(a);
U = chol_safe(H);
Ui = U \ eye(size(U));
Sigma = Ui * Ui';
end

function L = logpost(P, t, a, w, c)
z = P * w;
L = sum(c .* (t .* logsig(z) + (1 - t) .* logsig(-z))) - 0.5 * sum(a .* w.^2);
end

function v = logsig(z)
v = min(z, 0) - log1p(exp(-abs(z)));
end

function y = sigm(z)
y = 1 ./ (1 + exp(-z));
end

function U = chol_safe(H)
H = (H + H') / 2;
[U, p] = chol(H);
j = 1e-10 * max(mean(diag(H)), eps);
while p > 0
    [U, p] = chol(H + j * eye(size(H)));
    j = j * 10;
end
end
