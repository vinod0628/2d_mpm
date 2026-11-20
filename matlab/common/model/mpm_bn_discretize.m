function cuts = mpm_bn_discretize(x, y, method, nbins)
%MPM_BN_DISCRETIZE Cut points that discretise a continuous evidence variable for a Bayesian network.
%
%   cuts = mpm_bn_discretize(x, y, 'mdl')            supervised, Fayyad & Irani (1993)
%   cuts = mpm_bn_discretize(x, [], 'equalfreq', k)  k equal-frequency classes
%   States: x <= cuts(1) -> 1, cuts(1) < x <= cuts(2) -> 2, ..., x > cuts(end) -> numel(cuts) + 1.
%
%   MDL (Fayyad & Irani 1993, "Multi-interval discretization of continuous-valued attributes for
%   classification learning", IJCAI): recursively split at the boundary point T that minimises
%   the class-information entropy  E(T; S) = |S1|/|S| Ent(S1) + |S2|/|S| Ent(S2)  and accept the
%   split only if
%       Gain(T; S) > log2(N - 1) / N + Delta(T; S) / N,
%       Delta = log2(3^k - 2) - [k Ent(S) - k1 Ent(S1) - k2 Ent(S2)]
%   (N = |S|; k, k1, k2 = number of classes present in S, S1, S2). Candidate cuts are midpoints
%   between consecutive distinct values (boundary points). With no accepted split the variable has
%   one state (it carries no class information for this training set).
x = double(x(:));
switch lower(method)
    case 'mdl'
        y = double(y(:));
        ok = isfinite(x); x = x(ok); y = y(ok);
        [x, o] = sort(x); y = y(o);
        cuts = split(x, y);
        cuts = sort(cuts(:))';
    case 'equalfreq'
        x = x(isfinite(x));
        q = quantile(x, (1:nbins - 1) / nbins);
        u = unique(x);
        cuts = zeros(1, 0);
        for k = 1:numel(q)                     % move each cut to a midpoint between data values
            lo = u(find(u <= q(k), 1, 'last')); hi = u(find(u > q(k), 1, 'first'));
            if ~isempty(lo) && ~isempty(hi), cuts(end+1) = (lo + hi) / 2; end %#ok<AGROW>
        end
        cuts = unique(cuts);
    otherwise
        error('mpm:bn', 'Discretization must be ''mdl'' or ''equalfreq''.');
end
end

function cuts = split(x, y)
% x sorted ascending
N = numel(x); cuts = [];
if N < 2, return; end
cls = unique(y);
if numel(cls) < 2, return; end
Y = double(y == cls(:)');                          % N x k indicators
C = cumsum(Y, 1);                                  % class counts of the left part
tot = C(end, :);
cand = find(diff(x) > 0);                          % split after position i (x(i) < x(i+1))
if isempty(cand), return; end
nl = cand; nr = N - nl;
L = C(cand, :); Rr = tot - L;
E = (nl .* ent(L ./ nl) + nr .* ent(Rr ./ nr)) / N;
[Emin, j] = min(E);
i = cand(j);
entS = ent(tot / N);
gain = entS - Emin;
k = numel(cls); k1 = sum(L(j, :) > 0); k2 = sum(Rr(j, :) > 0);
e1 = ent(L(j, :) / nl(j)); e2 = ent(Rr(j, :) / nr(j));
delta = log2(3 ^ k - 2) - (k * entS - k1 * e1 - k2 * e2);
if gain <= (log2(N - 1) + delta) / N, return; end
cut = (x(i) + x(i + 1)) / 2;
cuts = [split(x(1:i), y(1:i)), cut, split(x(i+1:end), y(i+1:end))];
end

function h = ent(p)
% row-wise Shannon entropy (bits) of probability rows p
t = -p .* log2(p); t(p <= 0) = 0;
h = sum(t, 2);
end
