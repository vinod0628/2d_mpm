function mdl = mpm_lightgbm_train(X, y, opts)
%MPM_LIGHTGBM_TRAIN LightGBM: histogram-based leaf-wise gradient boosting with GOSS and EFB,
%   written from Ke et al. (2017).
%
%   mdl = mpm_lightgbm_train(X, y, opts)       predict with MPM_GB_PREDICT (+ MPM_GB_LOSS link)
%
%   Algorithm (paper section in brackets)
%     histogram    features are bucketed into <= max_bin bins once; split finding scans bin
%                  histograms of gradients/hessians (Alg. 1); a child's histogram is obtained by
%                  subtracting its sibling's from the parent's ("histogram subtraction")        [2.1]
%     leaf-wise    the tree grows best-first: the leaf with the largest gain is split next, until
%                  num_leaves leaves (max_depth limits depth if > 0)                             [2.1]
%     GOSS         gradient-based one-side sampling (boosting = 'goss'): keep the top_rate share
%                  of instances with the largest |g|, sample other_rate of the rest at random and
%                  up-weight them by (1 - top_rate) / other_rate (Alg. 2)                         [3]
%     EFB          exclusive feature bundling: features that are rarely non-default at the same
%                  time (one-hot classes) are bundled greedily (Alg. 3) and merged into one
%                  histogram with bin offsets (Alg. 4); splits are still found per original
%                  feature (its default bin = node total - its other bins)                        [4]
%     objective    second-order leaf weights w = -G / (H + lambda_l2) (L1 soft threshold if
%                  lambda_l1 > 0), shrinkage learning_rate; splits need gain > min_gain_to_split,
%                  >= min_data_in_leaf samples and >= min_sum_hessian_in_leaf per child
%     bagging      boosting 'gbdt' with bagging_fraction < 1 every bagging_freq iterations;
%                  feature_fraction = columns sampled per tree
%
%   opts (defaults = LightGBM library defaults): loss 'logistic' | 'squared', n_rounds 100,
%     learning_rate 0.1, num_leaves 31, max_depth -1, min_data_in_leaf 20, min_sum_hessian 1e-3,
%     lambda_l1 0, lambda_l2 0, min_gain_to_split 0, max_bin 255, feature_fraction 1,
%     bagging_fraction 1, bagging_freq 0, boosting 'gbdt' | 'goss', top_rate 0.2, other_rate 0.1,
%     efb true, max_conflict_rate 0, weights []
%
%   Reference: Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y., 2017.
%   LightGBM: a highly efficient gradient boosting decision tree. NeurIPS 30, 3146-3154.
d = struct('loss', 'logistic', 'n_rounds', 100, 'learning_rate', 0.1, 'num_leaves', 31, 'max_depth', -1, ...
    'min_data_in_leaf', 20, 'min_sum_hessian', 1e-3, 'lambda_l1', 0, 'lambda_l2', 0, 'min_gain_to_split', 0, ...
    'max_bin', 255, 'feature_fraction', 1, 'bagging_fraction', 1, 'bagging_freq', 0, 'boosting', 'gbdt', ...
    'top_rate', 0.2, 'other_rate', 0.1, 'efb', true, 'max_conflict_rate', 0, 'weights', []);
o = mpm_merge(d, opts);
X = double(X); y = double(y(:)); [n, P] = size(X);
w = ones(n, 1); if ~isempty(o.weights), w = double(o.weights(:)); end
[Bin, cuts, nbins] = mpm_gb_binning(X, o.max_bin);
E = build_bundles(Bin, nbins, o.efb, o.max_conflict_rate);
so = struct('lambda', o.lambda_l2, 'alpha', o.lambda_l1, 'min_hess', o.min_sum_hessian, ...
    'min_count', o.min_data_in_leaf);

mdl.base = mpm_gb_loss('base', o.loss, y, [], w);
mdl.trees = cell(o.n_rounds, 1);
gainImp = zeros(1, P); splitCnt = zeros(1, P);
F = mdl.base * ones(n, 1);
bagRows = (1:n)';
for t = 1:o.n_rounds
    [g, h] = mpm_gb_loss('grad', o.loss, y, F, w);
    switch lower(o.boosting)
        case 'goss'                                           % Alg. 2
            topN = max(1, round(o.top_rate * n)); othN = max(1, round(o.other_rate * n));
            [~, ord] = sort(abs(g), 'descend');
            rest = ord(topN+1:end);
            pick = rest(randperm(numel(rest), min(othN, numel(rest))));
            amp = (1 - o.top_rate) / o.other_rate;
            g(pick) = g(pick) * amp; h(pick) = h(pick) * amp;
            rows = sort([ord(1:topN); pick]);
        otherwise                                             % gbdt (+ bagging)
            if o.bagging_fraction < 1 && o.bagging_freq > 0 && mod(t - 1, o.bagging_freq) == 0
                bagRows = sort(randperm(n, max(2, round(o.bagging_fraction * n))))';
            end
            rows = bagRows;
    end
    allowed = true(1, P);
    if o.feature_fraction < 1
        allowed(:) = false; allowed(randperm(P, max(1, round(o.feature_fraction * P)))) = true;
    end
    tr = grow_leafwise(Bin, E, g, h, rows, nbins, allowed, so, o);
    tr.value = -soft(tr.G, o.lambda_l1) ./ (tr.H + o.lambda_l2) * o.learning_rate;
    tr.thr = nan(size(tr.feat));
    in = tr.feat > 0;
    tr.thr(in) = arrayfun(@(f, b) cuts{f}(b), tr.feat(in), tr.bin(in));
    for k = find(in)'
        gainImp(tr.feat(k)) = gainImp(tr.feat(k)) + tr.gain(k);
        splitCnt(tr.feat(k)) = splitCnt(tr.feat(k)) + 1;
    end
    mdl.trees{t} = struct('feat', tr.feat, 'thr', tr.thr, 'left', tr.left, 'right', tr.right, 'value', tr.value);
    F = F + mpm_gb_predict(struct('base', 0, 'trees', {mdl.trees(t)}), X);
end
mdl.loss = o.loss;
mdl.importance_gain = gainImp;
mdl.importance_split = splitCnt;
mdl.n_bundles = numel(E.bundles);
mdl.bundles = E.bundles;
mdl.opts = rmfield(o, 'weights');
mdl.algorithm = 'lightgbm';
end

% ===================================================================================== EFB
function E = build_bundles(Bin, nbins, useEfb, maxConflictRate)
% Greedy bundling (Alg. 3) + merge exclusive features (Alg. 4). Each feature's default bin is its
% most frequent bin; bundle bin 1 = every feature of the bundle at its default bin.
[n, P] = size(Bin);
E.default = zeros(1, P);
for j = 1:P, E.default(j) = mode(double(Bin(:, j))); end
nz = Bin ~= uint16(E.default);                             % non-default entries
if useEfb
    conf = double(nz)' * double(nz);                       % pairwise conflict counts
    conf(1:P+1:end) = 0;
    [~, order] = sort(sum(conf, 2), 'descend');
    K = floor(maxConflictRate * n);
    bundles = {}; bConf = []; bBins = [];
    for f = order'
        placed = false;
        for b = 1:numel(bundles)
            c = sum(conf(f, bundles{b}));
            if bConf(b) + c <= K && bBins(b) + nbins(f) - 1 < intmax('uint16')
                bundles{b}(end+1) = f; bConf(b) = bConf(b) + c; bBins(b) = bBins(b) + nbins(f) - 1;
                placed = true; break
            end
        end
        if ~placed, bundles{end+1} = f; bConf(end+1) = 0; bBins(end+1) = 1 + nbins(f) - 1; end %#ok<AGROW>
    end
else
    bundles = num2cell(1:P);
end
E.bundles = bundles;
E.of = zeros(1, P);                                        % bundle of each feature
E.map = cell(1, P);                                        % feature bin -> bundle bin (0 = default)
E.size = zeros(1, numel(bundles));
E.BB = ones(n, numel(bundles), 'uint16');                  % bundled bin matrix
for b = 1:numel(bundles)
    off = 1;
    for f = bundles{b}
        m = zeros(nbins(f), 1);
        nd = setdiff(1:nbins(f), E.default(f));
        m(nd) = off + (1:numel(nd));
        off = off + numel(nd);
        E.map{f} = m; E.of(f) = b;
        r = nz(:, f);
        E.BB(r, b) = uint16(m(Bin(r, f)));                 % conflicts: later feature wins (Alg. 4)
    end
    E.size(b) = off;
end
% vectorised unpacking: per-feature histogram cell (bin, f) <- bundle histogram cell (bundle bin, bundle)
nbMax = max(nbins); szB = max(E.size);
src = []; dst = [];
for f = 1:P
    mp = E.map{f}(:); k = find(mp > 0);                  % column vectors (constant feature -> empty)
    if isempty(k), continue; end
    src = [src; sub2ind([nbMax P], k, repmat(f, numel(k), 1))]; %#ok<AGROW>
    dst = [dst; sub2ind([szB numel(bundles)], mp(k), repmat(E.of(f), numel(k), 1))]; %#ok<AGROW>
end
E.src = src; E.dst = dst;
E.def = sub2ind([nbMax P], E.default, 1:P);
E.nbMax = nbMax; E.szB = szB;
end

function [G, H, C] = leaf_hist(E, g, h, r, ~, P)
% bundle histograms of rows r, unpacked to per-feature histograms (nbMax x P); each feature's
% default bin = leaf total - its other bins (exact when bundled features never conflict)
b = double(E.BB(r, :)); [m, nB] = size(b);
sub = [b(:), reshape(repmat(1:nB, m, 1), [], 1)];
GB = accumarray(sub, repmat(g(r), nB, 1), [E.szB nB]);
HB = accumarray(sub, repmat(h(r), nB, 1), [E.szB nB]);
CB = accumarray(sub, 1, [E.szB nB]);
G = zeros(E.nbMax, P); H = G; C = G;
G(E.src) = GB(E.dst); H(E.src) = HB(E.dst); C(E.src) = CB(E.dst);
G(E.def) = sum(g(r)) - sum(G, 1); H(E.def) = sum(h(r)) - sum(H, 1); C(E.def) = numel(r) - sum(C, 1);
end

% ===================================================================================== leaf-wise tree
function tr = grow_leafwise(Bin, E, g, h, rows, nbins, allowed, so, o)
P = size(Bin, 2); L = max(2, o.num_leaves); cap = 2 * L;
tr.feat = zeros(cap, 1); tr.bin = zeros(cap, 1); tr.left = zeros(cap, 1); tr.right = zeros(cap, 1);
tr.G = zeros(cap, 1); tr.H = zeros(cap, 1); tr.gain = zeros(cap, 1);
depth = zeros(cap, 1); nodeRows = cell(cap, 1); hist = cell(cap, 1); best = cell(cap, 1);
nodeRows{1} = rows; tr.G(1) = sum(g(rows)); tr.H(1) = sum(h(rows));
[Gh, Hh, Ch] = leaf_hist(E, g, h, rows, nbins, P);
hist{1} = {Gh, Hh, Ch};
best{1} = find_split(hist{1}, nbins, so, allowed, depth(1), o);
leaves = 1; nNodes = 1;
while numel(leaves) < L
    gains = cellfun(@(s) s.gain, best(leaves));
    [gmax, k] = max(gains);
    if ~(gmax > o.min_gain_to_split), break; end
    nd = leaves(k); s = best{nd}; r = nodeRows{nd};
    goL = Bin(r, s.feature) <= s.bin;
    a = nNodes + 1; c = nNodes + 2; nNodes = nNodes + 2;
    tr.feat(nd) = s.feature; tr.bin(nd) = s.bin; tr.gain(nd) = s.gain; tr.left(nd) = a; tr.right(nd) = c;
    nodeRows{a} = r(goL); nodeRows{c} = r(~goL);
    tr.G(a) = s.GL; tr.H(a) = s.HL; tr.G(c) = s.GR; tr.H(c) = s.HR;
    depth([a c]) = depth(nd) + 1;
    % histogram subtraction: build the smaller child, derive the larger one
    if numel(nodeRows{a}) <= numel(nodeRows{c}), sm = a; lg = c; else, sm = c; lg = a; end
    [Gs, Hs, Cs] = leaf_hist(E, g, h, nodeRows{sm}, nbins, P);
    hist{sm} = {Gs, Hs, Cs};
    hp = hist{nd};
    hist{lg} = {hp{1} - Gs, hp{2} - Hs, hp{3} - Cs};
    hist{nd} = []; nodeRows{nd} = [];
    best{a} = find_split(hist{a}, nbins, so, allowed, depth(a), o);
    best{c} = find_split(hist{c}, nbins, so, allowed, depth(c), o);
    leaves = [leaves(1:k-1), leaves(k+1:end), a, c];
end
k = 1:nNodes;
tr.feat = tr.feat(k); tr.bin = tr.bin(k); tr.left = tr.left(k); tr.right = tr.right(k);
tr.G = tr.G(k); tr.H = tr.H(k); tr.gain = tr.gain(k);
end

function s = find_split(hc, nbins, so, allowed, dep, o)
if o.max_depth > 0 && dep >= o.max_depth
    s = struct('gain', -Inf); return
end
s = mpm_gb_best_split(hc{1}, hc{2}, hc{3}, nbins, so, allowed);
end

function v = soft(g, a)
if a > 0, v = sign(g) .* max(abs(g) - a, 0); else, v = g; end
end
