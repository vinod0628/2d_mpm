function M = clf_ocsvm()
%CLF_OCSVM One-class support vector machine anomaly detection for prospectivity mapping.
%
%   Idea (Chen & Wu 2017): treat the multivariate evidence of mineralised ground as outliers of
%   the study area and find them with an outlier detector - no known deposits are needed to fit.
%
%   Model (Schoelkopf et al. 2001): map the data with a Gaussian kernel and separate it from the
%   origin with maximum margin:
%       min  1/2 ||w||^2 + 1/(nu*l) sum xi_i - rho   s.t.  w.Phi(x_i) >= rho - xi_i,  xi_i >= 0
%   dual:  f(x) = sum_i alpha_i k(x_i, x) - rho,   0 <= alpha_i <= 1/(nu*l),  sum alpha_i = 1
%   f >= 0 inside the estimated support of the data, f < 0 outside. nu-property: nu is an upper
%   bound on the fraction of outliers and a lower bound on the fraction of support vectors.
%   rho = sum_j alpha_j k(x_j, x_i) for any x_i with 0 < alpha_i < 1/(nu*l)  (KKT).
%
%   Implementation: the dual is solved by fitcsvm (one-class SMO). fitcsvm scales the alphas to
%   sum nu*l (bound 1); they are rescaled here to sum 1. Its Bias is NOT the rho of the paper
%   (checked: with it the nu-property fails), so rho is recomputed from the free support vectors
%   (KKT, mean over 0 < alpha < bound).
%   Score: anomaly = 1 - sum_i alpha_i k(x_i, x)   (sum alpha = 1, k <= 1 -> score in [0, 1));
%   a monotone function of -f; the support boundary f = 0 is at score 1 - rho (in the log).
%
%   Fit      : on min(cfg.anomaly.n_fit, MaxSamples) random cells of the study area, without
%              labels. Known deposits only select Nu x KernelScale (AUC of training deposits vs.
%              training background), the threshold (Youden index, as in Chen & Wu) and validate.
%   Hyper-parameters:
%     Nu           expected outlier fraction (upper bound), 0 < Nu <= 1
%     KernelScale  MULTIPLIER of sqrt(P) (z-scored features; same convention as clf_svm)
%     MaxSamples   cap on the fit cells (the QP is O(l^2))
%
%   References: Schoelkopf, B., Platt, J.C., Shawe-Taylor, J., Smola, A.J., Williamson, R.C.,
%   2001. Estimating the support of a high-dimensional distribution. Neural Comput. 13,
%   1443-1471.  Chen, Y., Wu, W., 2017. Mapping mineral prospectivity by using one-class support
%   vector machine to identify multivariate geological anomalies from digital geological survey
%   data. Aust. J. Earth Sci. 64, 639-651.

M.key = 'ocsvm';
M.name = 'One-class SVM';
M.task = 'classification';
M.learning = 'unsupervised';
M.description = 'Unsupervised anomaly detection: one-class SVM (Schoelkopf et al. 2001), Gaussian kernel.';
M.defaults = struct('Nu', 0.1, 'KernelScale', 1, 'MaxSamples', 3000);
M.grid = struct('Nu', [0.02 0.05 0.1 0.2], 'KernelScale', [0.25 0.5 1 2]);
M.score_is_probability = false;
M.score_name = 'anomaly score';
M.has_uncertainty = false;
M.train = @fit_model;
M.predict = @predict_score;
M.native_importance = [];
M.describe = @describe;
end

function mdl = fit_model(X, ~, p, ~)
X = double(X); n = size(X, 1);
if n > p.MaxSamples, X = X(sort(randperm(n, p.MaxSamples)), :); n = p.MaxSamples; end
ks = p.KernelScale * sqrt(size(X, 2));
svm = fitcsvm(X, ones(n, 1), 'KernelFunction', 'gaussian', 'KernelScale', ks, 'Nu', p.Nu, ...
    'Standardize', false, 'CacheSize', 'maximal');
a = svm.Alpha; sv = svm.SupportVectors;
bound = max(a);                                   % = 1 in fitcsvm's scaling (1/(nu*l) after /sum)
a = a / sum(a); bound = bound / sum(svm.Alpha);
mdl = struct('sv', sv, 'alpha', a, 'kernel_scale', ks, 'nu', p.Nu, 'n_fit', n);
g = kernel_sum(mdl, sv);
free = a > 1e-8 * bound & a < bound * (1 - 1e-6);
if any(free)
    mdl.rho = mean(g(free));
else                                              % all SVs at the bound: rho between the classes
    gAll = kernel_sum(mdl, X);
    inb = gAll >= max(g);  mdl.rho = (max(g) + min(gAll(inb))) / 2;
end
mdl.n_free = sum(free);
mdl.outlier_fraction = mean(kernel_sum(mdl, X) < mdl.rho * (1 - 1e-9));   % nu-property check
mdl.converged = svm.ConvergenceInfo.Converged;
end

function g = kernel_sum(mdl, X)
g = zeros(size(X, 1), 1);
for s = 1:20000:size(X, 1)
    i = s:min(s + 19999, size(X, 1));
    g(i) = exp(-pdist2(X(i, :), mdl.sv, 'squaredeuclidean') / mdl.kernel_scale ^ 2) * mdl.alpha;
end
end

function s = predict_score(mdl, X)
s = 1 - kernel_sum(mdl, double(X));
end

function txt = describe(mdl, p)
txt = sprintf(['Gaussian kernel, Nu = %.3g, KernelScale = %.4g (x%.3g sqrt(P)), %d support vectors ' ...
    '(%d free) of %d fit cells; outside the support: %.1f %% of fit cells (nu-property: <= Nu), ' ...
    'boundary at score %.4f%s'], mdl.nu, mdl.kernel_scale, p.KernelScale, numel(mdl.alpha), mdl.n_free, ...
    mdl.n_fit, 100 * mdl.outlier_fraction, 1 - mdl.rho, conv_text(mdl.converged));
end

function s = conv_text(c)
if c, s = ''; else, s = ' (SMO not converged)'; end
end
