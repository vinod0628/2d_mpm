function mdl = mpm_pubag_train(X, s, o)
%MPM_PUBAG_TRAIN Positive-unlabelled (PU) bagging, written from Mordelet & Vert (2014).
%
%   mdl = mpm_pubag_train(X, s, opts)
%   X  N x P inputs (z-scored);  s  N x 1 logical: true = labelled positive, false = UNLABELLED
%   opts (all optional):
%     n_bags        T, number of bootstrap classifiers                                  (100)
%     subsample     K, size of each random subsample of the unlabelled set; default = number of
%                   positives (the paper's "safe choice" K = N_P)                          ([])
%     replace       true: U_t drawn with replacement (bootstrap, Alg. 2), false: without   (true)
%     base          'svm'  biased SVM, Gaussian kernel (the paper's base classifier)
%                   'tree' classification tree (CART; MPM practice, e.g. Farahbakhsh et al. 2023)
%     box, kernel_scale   SVM: C and Gaussian kernel width s (k = exp(-||x - z||^2 / s^2))  (1, sqrt(P))
%     min_leaf      tree: minimum leaf size                                                 (1)
%     oob           true: also compute the out-of-bag score of every unlabelled sample (Alg. 2)
%
%   ALGORITHM 1 (inductive bagging PU; Mordelet & Vert 2014)
%     for t = 1..T:  draw a subsample U_t of size K from U;
%                    train f_t to discriminate P (label 1) from U_t (label 0)
%     return f = 1/T sum_t f_t
%   ALGORITHM 2 (transductive): every x in U is scored by the average of the f_t whose U_t did
%     not contain x (out-of-bag); returned here as mdl.oob_score (aligned with find(~s)).
%   Base classifier: biased SVM, penalties with C+ n+ = C- n- (equal total penalty per class),
%     a single C tuned. Implemented with fitcsvm and a uniform class prior, which rescales the
%     per-sample box constraints exactly that way. The tree uses the same uniform prior.
%   Why it works (paper, Sect. 3): with U = gamma*P + (1 - gamma)*N, h_U/h_P = gamma +
%     (1 - gamma) h_N/h_P, so a P-vs-U score is monotone in the P-vs-N score; small K makes the
%     contamination of each U_t vary (Binomial(K, gamma)) and the aggregated classifiers diverse.
%
%   mdl (svm): Z unique support vectors (rows), Coef (nZ x T, alpha_i * y_i per member), bias (1 x T)
%       (tree): trees (cell of compact trees)
%   Prediction: MPM_PUBAG_PREDICT.
%
%   Reference: Mordelet, F., Vert, J.-P., 2014. A bagging SVM to learn from positive and unlabeled
%   examples. Pattern Recognition Letters 37, 201-209.
if nargin < 3, o = struct(); end
X = double(X); s = logical(s(:)); P = size(X, 2);
o = mpm_merge(struct('n_bags', 100, 'subsample', [], 'replace', true, 'base', 'svm', 'box', 1, ...
    'kernel_scale', sqrt(P), 'min_leaf', 1, 'oob', true), o);
ip = find(s); iu = find(~s);
nP = numel(ip); nU = numel(iu);
if nP < 1 || nU < 1, error('mpm:pubag', 'Need positive and unlabelled samples.'); end
K = o.subsample; if isempty(K), K = nP; end
K = max(1, round(K));
if ~o.replace, K = min(K, nU); end
T = o.n_bags;
XP = X(ip, :);
yt = [true(nP, 1); false(K, 1)];
mdl = struct('base', lower(o.base), 'n_bags', T, 'K', K, 'n_pos', nP, 'n_unl', nU, ...
    'replace', o.replace, 'kernel_scale', o.kernel_scale);

oobSum = zeros(nU, 1); oobN = zeros(nU, 1);
switch mdl.base
    case 'svm'
        SV = cell(T, 1); CF = cell(T, 1); bias = zeros(1, T); nsv = zeros(T, 1);
        for t = 1:T
            u = draw(nU, K, o.replace);
            m = fitcsvm([XP; X(iu(u), :)], yt, 'KernelFunction', 'gaussian', 'KernelScale', o.kernel_scale, ...
                'BoxConstraint', o.box, 'ClassNames', [false; true], 'Prior', 'uniform', 'Standardize', false);
            SV{t} = m.SupportVectors; CF{t} = m.Alpha .* m.SupportVectorLabels; bias(t) = m.Bias;
            nsv(t) = size(SV{t}, 1);
            if o.oob
                out = true(nU, 1); out(u) = false;
                oobSum(out) = oobSum(out) + svm_score(X(iu(out), :), SV{t}, CF{t}, bias(t), o.kernel_scale);
                oobN(out) = oobN(out) + 1;
            end
        end
        % merge the members into one kernel expansion over the distinct support vectors
        allSV = vertcat(SV{:});
        [mdl.Z, ~, map] = unique(allSV, 'rows');
        col = repelem((1:T)', nsv);
        mdl.Coef = accumarray([map, col], vertcat(CF{:}), [size(mdl.Z, 1), T]);
        mdl.bias = bias;
        mdl.n_sv = nsv;
    case 'tree'
        mdl.trees = cell(T, 1); nl = zeros(T, 1);
        for t = 1:T
            u = draw(nU, K, o.replace);
            m = fitctree([XP; X(iu(u), :)], yt, 'MinLeafSize', o.min_leaf, 'Prior', 'uniform', ...
                'ClassNames', [false; true]);
            mdl.trees{t} = compact(m);
            nl(t) = sum(~m.IsBranchNode);
            if o.oob
                out = true(nU, 1); out(u) = false;
                [~, sc] = predict(mdl.trees{t}, X(iu(out), :));
                oobSum(out) = oobSum(out) + sc(:, 2);
                oobN(out) = oobN(out) + 1;
            end
        end
        mdl.n_leaves = nl;
    otherwise
        error('mpm:pubag', 'base must be ''svm'' or ''tree''.');
end
if o.oob
    mdl.oob_score = oobSum ./ oobN;                % NaN if a sample was in every subsample
    mdl.oob_count = oobN;
end
end

function u = draw(nU, K, replace)
if replace, u = randi(nU, K, 1); else, u = randperm(nU, K)'; end
end

function f = svm_score(X, SV, cf, b, ks)
f = zeros(size(X, 1), 1);
for i = 1:20000:size(X, 1)
    j = i:min(i + 19999, size(X, 1));
    f(j) = exp(-pdist2(X(j, :), SV, 'squaredeuclidean') / ks ^ 2) * cf + b;
end
end
