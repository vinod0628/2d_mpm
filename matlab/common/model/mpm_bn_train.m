function mdl = mpm_bn_train(D, r, y, o)
%MPM_BN_TRAIN Bayesian network classifier on discrete evidence: naive Bayes, tree-augmented
%   naive Bayes (TAN) or selective naive Bayes.
%
%   mdl = mpm_bn_train(D, r, y, opts)
%   D   N x n integer states (1..r(i)) of the n evidence nodes;  r  1 x n numbers of states
%   y   N x 1 logical class (true = deposit)
%   opts.structure   'naive' | 'tan' | 'selective'
%   opts.N0          Dirichlet smoothing strength (Friedman et al. 1997: 5)
%   opts.alpha       significance of the conditional-independence G-test ('selective')
%   opts.prior       'uniform' (both classes 1/2) | 'empirical'
%   opts.draws       number of Dirichlet-posterior parameter draws for the uncertainty (0 = none)
%
%   Model: P(C | x) ~ P(C) prod_i P(x_i | C, x_pi(i)) with pi(i) the attribute parent (TAN) or none.
%   NAIVE   every node has the class as its only parent (complete conditional independence).
%   TAN     (Friedman, Geiger & Goldszmidt 1997, Construct-TAN): conditional mutual information
%           I(A_i; A_j | C) for every pair, maximum weighted spanning tree, edges directed away from
%           a root; each attribute has the class and at most one other attribute as parents. This
%           is the maximum-likelihood TAN structure (their Theorem 2).
%   SELECTIVE  only conditionally independent predictors (Porwal et al. 2006 definition): while
%           some pair is conditionally dependent given C (G = 2 N I(A_i; A_j | C) [nats],
%           chi-square with (r_i - 1)(r_j - 1) |C| dof, p < alpha), drop the member of the most
%           dependent pair that carries less information about the class, I(A; C).
%   Parameters: theta(x | pa) = (N(x, pa) + N0 theta0(x)) / (N(pa) + N0) with theta0 the (Laplace-
%           smoothed) marginal of x - the Dirichlet smoothing of Friedman et al. (1997), applied
%           after the structure is chosen.
%   Uncertainty: S draws theta ~ Dirichlet(N(., pa) + N0 theta0) of every conditional probability
%           table; the spread of the posterior probability over the draws is the parameter
%           uncertainty of the network.
o = mpm_merge(struct('structure', 'tan', 'N0', 5, 'alpha', 0.05, 'prior', 'uniform', 'draws', 0), o);
D = double(D); y = logical(y(:)); [N, n] = size(D); c = y + 1;          % class 1 = background, 2 = deposit
% ---------------------------------------------------------------- class prior
if strcmpi(o.prior, 'uniform'), pc = [0.5 0.5]; else, pc = [mean(~y) mean(y)]; end
mdl.logprior = log(pc(2)) - log(pc(1));
% ---------------------------------------------------------------- information measures
MI = zeros(1, n);
for i = 1:n, MI(i) = mutual_info(D(:, i), r(i), c); end
mdl.mi_class = MI;                                                      % I(A_i; C), bits
parent = zeros(1, n); use = true(1, n);
switch lower(o.structure)
    case 'naive'
    case 'tan'
        if n > 1
            W = cmi_matrix(D, r, c);                                    % I(A_i; A_j | C), nats
            T = minspantree(graph(-W, 'upper', 'omitselfloops'), 'Type', 'forest');  % maximum weighted spanning tree
            [~, root] = max(MI);                                        % any root gives the same likelihood
            parent = tree_parents(T, root, n);
        end
    case 'selective'
        W = cmi_matrix(D, r, c);
        while true
            act = find(use); if numel(act) < 2, break; end
            Wa = W(act, act); best = 0; bi = 0; bj = 0;
            for a = 1:numel(act)
                for b = a+1:numel(act)
                    i = act(a); j = act(b);
                    dof = max((r(i) - 1) * (r(j) - 1) * 2, 1);
                    p = 1 - chi2cdf(2 * N * Wa(a, b), dof);
                    if p < o.alpha && Wa(a, b) > best, best = Wa(a, b); bi = i; bj = j; end
                end
            end
            if bi == 0, break; end
            if MI(bi) < MI(bj), use(bi) = false; else, use(bj) = false; end
        end
    otherwise
        error('mpm:bn', 'structure must be naive, tan or selective.');
end
% ---------------------------------------------------------------- parameters
mdl.parent = parent; mdl.use = use; mdl.r = r;
mdl.L = cell(1, n); mdl.Ld = cell(1, n); mdl.counts = cell(1, n);
for i = find(use)
    p = parent(i); rp = 1; xp = ones(N, 1);
    if p > 0, rp = r(p); xp = D(:, p); end
    Nc = accumarray([D(:, i), xp, c], 1, [r(i), rp, 2]);               % counts N(x_i, x_p, c)
    th0 = (accumarray(D(:, i), 1, [r(i), 1]) + 1) / (N + r(i));        % smoothed marginal of x_i
    A = Nc + o.N0 * th0;                                                % Dirichlet posterior parameters
    TH = A ./ sum(A, 1);
    mdl.L{i} = log(TH(:, :, 2)) - log(TH(:, :, 1));                     % log-likelihood ratio table
    mdl.counts{i} = Nc;
    if o.draws > 0                                                      % posterior draws (gamma trick)
        G = randg(repmat(A, 1, 1, 1, o.draws));
        TD = G ./ sum(G, 1);
        mdl.Ld{i} = reshape(log(TD(:, :, 2, :)) - log(TD(:, :, 1, :)), r(i), rp, o.draws);
    end
end
mdl.structure = lower(o.structure); mdl.N0 = o.N0; mdl.draws = o.draws; mdl.n_train = N;
end

% =====================================================================================
function I = mutual_info(x, rx, c)
P = accumarray([x, c], 1, [rx, 2]) / numel(x);
Px = sum(P, 2); Pc = sum(P, 1);
t = P .* log2(P ./ (Px * Pc)); t(P == 0) = 0;
I = sum(t(:));
end

function W = cmi_matrix(D, r, c)
% conditional mutual information I(A_i; A_j | C) (nats) for all pairs, empirical (unsmoothed)
[N, n] = size(D); W = zeros(n);
for i = 1:n
    for j = i+1:n
        P = accumarray([D(:, i), D(:, j), c], 1, [r(i), r(j), 2]) / N;
        Pz = sum(sum(P, 1), 2); Pxz = sum(P, 2); Pyz = sum(P, 1);
        t = P .* log(P .* Pz ./ (Pxz .* Pyz)); t(P == 0) = 0;
        W(i, j) = sum(t(:)); W(j, i) = W(i, j);
    end
end
end

function parent = tree_parents(T, root, n)
% direct the spanning forest away from root (and from the first node of every other component)
parent = zeros(1, n); seen = false(1, n);
order = [root, setdiff(1:n, root)];
for s = order
    if seen(s), continue; end
    v = bfsearch(T, s, 'edgetonew');
    seen(s) = true;
    for e = 1:size(v, 1)
        parent(v(e, 2)) = v(e, 1); seen(v(e, 2)) = true;
    end
end
end
