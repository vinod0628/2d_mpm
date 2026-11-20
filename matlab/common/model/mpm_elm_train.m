function mdl = mpm_elm_train(X, T, o)
%MPM_ELM_TRAIN Extreme learning machine: single-hidden-layer feed-forward network (SLFN) whose
%hidden nodes are random and whose output weights are solved analytically (no iteration).
%
%   mdl = mpm_elm_train(X, T, opts)
%   X  N x P inputs (z-scored),  T  N x m targets,  opts (all optional):
%     n_hidden     number of hidden nodes L                                     (100)
%     activation   'sigmoid' | 'tanh' | 'sine' | 'hardlim' | 'tribas' | 'radbas' ('sigmoid')
%     C            regularisation parameter; Inf = original ELM (Moore-Penrose)  (1)
%     input_scale  hidden weights w ~ U(-a, a), a = input_scale * sqrt(3/P)     (1)
%     weights      N x 1 sample weights = diagonal of W (weighted ELM), [] = all 1
%
%   ALGORITHM (Huang, Zhu & Siew 2006, "Algorithm ELM"):
%     Step 1  randomly assign input weights w_i and biases b_i, i = 1..L
%     Step 2  hidden layer output matrix  H(j, i) = g(w_i . x_j + b_i)          (N x L, Eq. 4)
%     Step 3  output weights  beta = H^+ T  (minimum-norm least squares, Eq. 16)
%   Regularised ELM (Huang, Zhou, Ding & Zhang 2012) minimises 1/2||beta||^2 + C/2 sum ||e_j||^2:
%     N >= L:  beta = (I/C + H'H)^-1 H'T            N < L:  beta = H'(I/C + HH')^-1 T
%   Weighted ELM (Zong, Huang & Chen 2013) weights each training error by W = diag(w):
%     N >= L:  beta = (I/C + H'WH)^-1 H'WT          N < L:  beta = H'(I/C + WHH')^-1 WT
%   Both cases are solved here in the equivalent symmetric form with Hd = W^1/2 H, Td = W^1/2 T:
%     beta = (I/C + Hd'Hd) \ Hd'Td   or   beta = Hd' ((I/C + Hd Hd') \ Td)   (Cholesky).
%   C = Inf gives the weighted minimum-norm least-squares solution beta = pinv(Hd) Td.
%
%   Random hidden nodes: Theorem 2.1 of Huang et al. (2006) holds for w_i, b_i drawn from any
%   interval by any continuous distribution. The inputs here are z-scored (the paper scaled them
%   to [0, 1]), so w ~ U(-a, a) with a = input_scale*sqrt(3/P) gives the input part of the
%   pre-activation a standard deviation of input_scale; b ~ U(-1, 1).
%   Activation names follow the authors' reference code (sig, sin, hardlim, tribas, radbas).
%
%   mdl fields: W (P x L), b (1 x L), beta (L x m), activation, form ('primal' | 'dual' | 'pinv'),
%   n_hidden, C. Prediction: MPM_ELM_PREDICT.
%
%   References: Huang, G.-B., Zhu, Q.-Y., Siew, C.-K., 2006. Extreme learning machine: theory and
%   applications. Neurocomputing 70, 489-501.  Huang, G.-B., Zhou, H., Ding, X., Zhang, R., 2012.
%   Extreme learning machine for regression and multiclass classification. IEEE Trans. Syst. Man
%   Cybern. B 42, 513-529.  Zong, W., Huang, G.-B., Chen, Y., 2013. Weighted extreme learning
%   machine for imbalance learning. Neurocomputing 101, 229-242.
if nargin < 3, o = struct(); end
o = mpm_merge(struct('n_hidden', 100, 'activation', 'sigmoid', 'C', 1, 'input_scale', 1, ...
    'weights', []), o);
X = double(X); T = double(T);
[N, P] = size(X); L = round(o.n_hidden);
if size(T, 1) ~= N, error('mpm:elm', 'X and T must have the same number of rows.'); end
if L < 1, error('mpm:elm', 'n_hidden must be >= 1.'); end
if ~(o.C > 0), error('mpm:elm', 'C must be > 0 (Inf = no regularisation).'); end

% Step 1: random hidden nodes (global random stream -> reproducible with rng(seed))
a = o.input_scale * sqrt(3 / max(P, 1));
mdl.W = (2 * rand(P, L) - 1) * a;
mdl.b = 2 * rand(1, L) - 1;
mdl.activation = lower(o.activation);
mdl.n_hidden = L; mdl.C = o.C;

% Step 2: hidden layer output matrix
H = mpm_elm_hidden(mdl, X);

% Step 3: output weights
if isempty(o.weights), w = ones(N, 1); else, w = double(o.weights(:)); end
if numel(w) ~= N || any(w < 0), error('mpm:elm', 'weights must be N non-negative values.'); end
d = sqrt(w);
Hd = H .* d; Td = T .* d;
if isinf(o.C)
    mdl.beta = pinv(Hd) * Td; mdl.form = 'pinv';
elseif N >= L
    A = Hd' * Hd; A(1:L+1:end) = A(1:L+1:end) + 1 / o.C;
    mdl.beta = A \ (Hd' * Td); mdl.form = 'primal';
else
    A = Hd * Hd'; A(1:N+1:end) = A(1:N+1:end) + 1 / o.C;
    mdl.beta = Hd' * (A \ Td); mdl.form = 'dual';
end
end
