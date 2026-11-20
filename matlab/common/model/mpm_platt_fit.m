function [A, B] = mpm_platt_fit(f, y, w)
%MPM_PLATT_FIT Platt sigmoid  P(y = 1 | f) = 1 / (1 + exp(A f + B))  for raw classifier outputs.
%
%   [A, B] = mpm_platt_fit(f, y)       f: N x 1 raw outputs (ideally out-of-fold), y: logical
%   [A, B] = mpm_platt_fit(f, y, w)    w: N x 1 sample weights in the likelihood (default 1)
%
%   Platt (1999): A, B maximise the likelihood of the regularised targets
%     t = (N+ + 1)/(N+ + 2) for positives,  t = 1/(N- + 2) for negatives
%   (avoids over-fitting the sigmoid on separable outputs). Minimisation by Newton's method with
%   backtracking line search and the numerically stable loss of Lin, Lin & Weng (2007, Alg. 1).
%
%   References: Platt, J.C., 1999. Probabilistic outputs for support vector machines and
%   comparisons to regularized likelihood methods. In: Advances in Large Margin Classifiers,
%   MIT Press, 61-74.  Lin, H.-T., Lin, C.-J., Weng, R.C., 2007. A note on Platt's probabilistic
%   outputs for support vector machines. Machine Learning 68, 267-276.
f = double(f(:)); y = logical(y(:));
if nargin < 3 || isempty(w), w = ones(size(f)); else, w = double(w(:)); end
np = sum(y); nn = sum(~y);
t = zeros(size(f)); t(y) = (np + 1) / (np + 2); t(~y) = 1 / (nn + 2);
A = 0; B = log((nn + 1) / (np + 1));
maxIter = 100; minStep = 1e-10; sigma = 1e-12;
F = loss(A, B);
for it = 1:maxIter
    z = f * A + B;
    [p, q] = pq(z);                              % p = 1/(1+exp(z)), q = 1 - p (stable)
    d2 = w .* p .* q;
    h11 = sigma + sum(d2 .* f .^ 2); h22 = sigma + sum(d2); h21 = sum(d2 .* f);
    d1 = w .* (t - p);
    g1 = sum(d1 .* f); g2 = sum(d1);
    if abs(g1) < 1e-5 && abs(g2) < 1e-5, break; end
    det = h11 * h22 - h21 ^ 2;
    dA = -(h22 * g1 - h21 * g2) / det; dB = -(-h21 * g1 + h11 * g2) / det;
    gd = g1 * dA + g2 * dB;
    step = 1;
    while step >= minStep
        An = A + step * dA; Bn = B + step * dB; Fn = loss(An, Bn);
        if Fn < F + 1e-4 * step * gd, A = An; B = Bn; F = Fn; break; end
        step = step / 2;
    end
    if step < minStep, break; end                % line search failed: keep the current A, B
end

    function v = loss(a, b)
        zz = f * a + b;
        e = zeros(size(zz)); k = zz >= 0;          % stable  t*z + log(1 + exp(-z))
        e(k) = t(k) .* zz(k) + log1p(exp(-zz(k)));
        e(~k) = (t(~k) - 1) .* zz(~k) + log1p(exp(zz(~k)));
        v = sum(w .* e);
    end
end

function [p, q] = pq(z)
p = zeros(size(z)); k = z >= 0;
p(k) = exp(-z(k)) ./ (1 + exp(-z(k)));
p(~k) = 1 ./ (1 + exp(z(~k)));
q = 1 - p;
end
