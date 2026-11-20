function mdl = mpm_xgboost_train(X, y, opts)
%MPM_XGBOOST_TRAIN XGBoost: regularised gradient tree boosting, written from Chen & Guestrin (2016).
%
%   mdl = mpm_xgboost_train(X, y, opts)        predict with MPM_GB_PREDICT (+ MPM_GB_LOSS link)
%
%   Algorithm (paper section in brackets)
%     objective   L = sum_i l(y_i, F_i) + sum_k [ gamma*T_k + 1/2*lambda*||w_k||^2 (+ alpha*|w_k|) ]  [2.1]
%     boosting    second-order Taylor expansion of l: g_i, h_i; optimal leaf weight
%                 w* = -G / (H + lambda) (Eq. 5); split gain Eq. 7                              [2.2]
%     shrinkage   new tree weights scaled by eta; column subsampling per tree; row subsampling   [2.3]
%     splits      exact greedy over all distinct values (Alg. 1, tree_method 'exact') or
%                 approximate over quantile candidate thresholds (Alg. 2, tree_method 'approx')  [3.1-3.2]
%     growth      depth-wise (level by level) to max_depth, then splits whose gain does not exceed
%                 gamma are pruned bottom-up (gain - gamma < 0 -> merge into a leaf)
%     min_child_weight  minimum hessian sum of a child (stops splitting near-pure nodes)
%   Missing values are imputed upstream, so the sparsity-aware default direction (Alg. 3) is not
%   needed here.
%
%   opts (defaults): loss 'logistic' | 'squared', n_rounds 100, eta 0.3, max_depth 6,
%     min_child_weight 1, lambda 1, alpha 0, gamma 0, subsample 1, colsample_bytree 1,
%     tree_method 'exact' | 'approx', max_bin 256 (approx), weights [] (instance weights)
%
%   Reference: Chen, T., Guestrin, C., 2016. XGBoost: a scalable tree boosting system. Proc. 22nd
%   ACM SIGKDD, 785-794.
d = struct('loss', 'logistic', 'n_rounds', 100, 'eta', 0.3, 'max_depth', 6, 'min_child_weight', 1, ...
    'lambda', 1, 'alpha', 0, 'gamma', 0, 'subsample', 1, 'colsample_bytree', 1, ...
    'tree_method', 'exact', 'max_bin', 256, 'weights', []);
o = mpm_merge(d, opts);
X = double(X); y = double(y(:)); [n, P] = size(X);
w = ones(n, 1); if ~isempty(o.weights), w = double(o.weights(:)); end
if strcmpi(o.tree_method, 'exact'), mb = Inf; else, mb = o.max_bin; end
[Bin, cuts, nbins] = mpm_gb_binning(X, mb);
nbMax = max(nbins);
so = struct('lambda', o.lambda, 'alpha', o.alpha, 'min_hess', o.min_child_weight, 'min_count', 1);

mdl.base = mpm_gb_loss('base', o.loss, y, [], w);
mdl.trees = cell(o.n_rounds, 1);
gainImp = zeros(1, P); splitCnt = zeros(1, P);
F = mdl.base * ones(n, 1);
nLeaves = zeros(o.n_rounds, 1);
for t = 1:o.n_rounds
    [g, h] = mpm_gb_loss('grad', o.loss, y, F, w);
    rows = (1:n)';
    if o.subsample < 1, rows = sort(randperm(n, max(2, round(o.subsample * n))))'; end
    allowed = true(1, P);
    if o.colsample_bytree < 1
        allowed(:) = false; allowed(randperm(P, max(1, round(o.colsample_bytree * P)))) = true;
    end
    tr = grow_depthwise(Bin, g, h, rows, nbins, nbMax, allowed, so, o.max_depth);
    tr = prune(tr, o.gamma);
    tr.value = -soft(tr.G, o.alpha) ./ (tr.H + o.lambda) * o.eta;     % Eq. 5 with shrinkage
    tr.thr = nan(size(tr.feat));
    in = tr.feat > 0;
    tr.thr(in) = arrayfun(@(f, b) cuts{f}(b), tr.feat(in), tr.bin(in));
    for k = find(in)'
        gainImp(tr.feat(k)) = gainImp(tr.feat(k)) + tr.gain(k);
        splitCnt(tr.feat(k)) = splitCnt(tr.feat(k)) + 1;
    end
    nLeaves(t) = sum(reachable_leaves(tr));
    mdl.trees{t} = struct('feat', tr.feat, 'thr', tr.thr, 'left', tr.left, 'right', tr.right, 'value', tr.value);
    F = F + mpm_gb_predict(struct('base', 0, 'trees', {mdl.trees(t)}), X);
end
mdl.loss = o.loss;
mdl.importance_gain = gainImp;
mdl.importance_split = splitCnt;
mdl.n_leaves = nLeaves;
mdl.opts = rmfield(o, 'weights');
mdl.algorithm = 'xgboost';
end

% =====================================================================================
function tr = grow_depthwise(Bin, g, h, rows, nbins, nbMax, allowed, so, maxDepth)
cap = 2^(maxDepth + 1);
tr.feat = zeros(cap, 1); tr.bin = zeros(cap, 1); tr.left = zeros(cap, 1); tr.right = zeros(cap, 1);
tr.G = zeros(cap, 1); tr.H = zeros(cap, 1); tr.gain = zeros(cap, 1);
nodeRows = cell(cap, 1);
nodeRows{1} = rows; tr.G(1) = sum(g(rows)); tr.H(1) = sum(h(rows));
nNodes = 1; level = 1;
for depth = 0:maxDepth-1
    next = [];
    for nd = level
        r = nodeRows{nd};
        if numel(r) < 2, continue; end
        [Gh, Hh, Ch] = node_hist(Bin, g, h, r, nbMax);
        s = mpm_gb_best_split(Gh, Hh, Ch, nbins, so, allowed);
        if ~(s.gain > 0), continue; end
        goL = Bin(r, s.feature) <= s.bin;
        L = nNodes + 1; Rn = nNodes + 2; nNodes = nNodes + 2;
        tr.feat(nd) = s.feature; tr.bin(nd) = s.bin; tr.gain(nd) = s.gain;
        tr.left(nd) = L; tr.right(nd) = Rn;
        nodeRows{L} = r(goL); nodeRows{Rn} = r(~goL);
        tr.G(L) = s.GL; tr.H(L) = s.HL; tr.G(Rn) = s.GR; tr.H(Rn) = s.HR;
        next = [next, L, Rn]; %#ok<AGROW>
    end
    if isempty(next), break; end
    level = next;
end
k = 1:nNodes;
tr.feat = tr.feat(k); tr.bin = tr.bin(k); tr.left = tr.left(k); tr.right = tr.right(k);
tr.G = tr.G(k); tr.H = tr.H(k); tr.gain = tr.gain(k);
end

function tr = prune(tr, gamma)
% bottom-up: a split whose children are both leaves is removed when gain - gamma <= 0
for nd = numel(tr.feat):-1:1
    if tr.feat(nd) > 0 && tr.feat(tr.left(nd)) == 0 && tr.feat(tr.right(nd)) == 0 && tr.gain(nd) - gamma <= 0
        tr.feat(nd) = 0; tr.left(nd) = 0; tr.right(nd) = 0;
    end
end
end

function leaf = reachable_leaves(tr)
leaf = false(numel(tr.feat), 1); stack = 1;
while ~isempty(stack)
    nd = stack(end); stack(end) = [];
    if tr.feat(nd) > 0, stack = [stack, tr.left(nd), tr.right(nd)]; else, leaf(nd) = true; end %#ok<AGROW>
end
end

function [G, H, C] = node_hist(Bin, g, h, r, nbMax)
b = double(Bin(r, :)); [m, P] = size(b);
sub = [b(:), reshape(repmat(1:P, m, 1), [], 1)];
G = accumarray(sub, repmat(g(r), P, 1), [nbMax P]);
H = accumarray(sub, repmat(h(r), P, 1), [nbMax P]);
C = accumarray(sub, 1, [nbMax P]);
end

function v = soft(g, a)
if a > 0, v = sign(g) .* max(abs(g) - a, 0); else, v = g; end
end
