function mdl = mpm_autoencoder_train(X, o)
%MPM_AUTOENCODER_TRAIN Fully connected autoencoder trained by back-propagation + Adam.
%
%   mdl = mpm_autoencoder_train(X, opts)      X: N x P (z-scored)
%   opts (all optional):
%     hidden      encoder widths before the code layer, e.g. 16 or [32 16]       (16)
%     code        code (bottleneck) size                                         (4)
%     activation  hidden layers: 'tanh' | 'relu' | 'sigmoid'; code + output linear ('tanh')
%     l2          weight decay lambda (weights only, not biases)                 (1e-4)
%     epochs, batch, lr, beta1, beta2, epsilon     Adam settings (50, 256, 1e-3, 0.9, 0.999, 1e-8)
%
%   Architecture  P -> hidden -> code (linear) -> fliplr(hidden) -> P (linear), the symmetric
%                 deep autoencoder of Hinton & Salakhutdinov (2006) without RBM pre-training.
%   Loss          J = 1/(B P) sum_b ||x_hat_b - x_b||^2 + lambda/2 sum_l ||W_l||_F^2
%   Gradients     exact back-propagation of J (checked against finite differences).
%   Optimiser     Adam (Kingma & Ba 2015, Algorithm 1) with bias-corrected moments, mini-batches
%                 reshuffled every epoch. Initialisation: Glorot uniform weights, zero biases
%                 (Glorot & Bengio 2010; the trainnet default).
%   Written here instead of trainnet because trainnet costs ~30 ms per iteration on these tiny
%   networks (~50 x slower); results are equivalent (see README).
%
%   mdl: W, b (cell arrays), act (per layer), loss (per epoch). Reconstruction: MPM_AUTOENCODER_PREDICT.
if nargin < 2, o = struct(); end
o = mpm_merge(struct('hidden', 16, 'code', 4, 'activation', 'tanh', 'l2', 1e-4, 'epochs', 50, ...
    'batch', 256, 'lr', 1e-3, 'beta1', 0.9, 'beta2', 0.999, 'epsilon', 1e-8), o);
X = double(X); [N, P] = size(X);
h = o.hidden(:)';
sz = [P h o.code fliplr(h) P];
nl = numel(sz) - 1;
act = repmat({lower(o.activation)}, 1, nl);
act{numel(h) + 1} = 'linear'; act{nl} = 'linear';             % code and output layers linear
W = cell(1, nl); b = cell(1, nl);
for l = 1:nl
    r = sqrt(6 / (sz(l) + sz(l + 1)));                          % Glorot uniform
    W{l} = (2 * rand(sz(l), sz(l + 1)) - 1) * r; b{l} = zeros(1, sz(l + 1));
end
mW = cellfun(@(w) zeros(size(w)), W, 'UniformOutput', false); vW = mW;
mb = cellfun(@(w) zeros(size(w)), b, 'UniformOutput', false); vb = mb;
t = 0; B = min(o.batch, N);
loss = zeros(o.epochs, 1);
for ep = 1:o.epochs
    perm = randperm(N); acc = 0;
    for s = 1:B:N
        i = perm(s:min(s + B - 1, N)); Xb = X(i, :);
        [J, gW, gb] = mpm_autoencoder_grad(W, b, act, Xb, o.l2);
        acc = acc + J * numel(i);
        t = t + 1;
        c1 = 1 - o.beta1 ^ t; c2 = 1 - o.beta2 ^ t;
        for l = 1:nl
            mW{l} = o.beta1 * mW{l} + (1 - o.beta1) * gW{l};
            vW{l} = o.beta2 * vW{l} + (1 - o.beta2) * gW{l} .^ 2;
            W{l} = W{l} - o.lr * (mW{l} / c1) ./ (sqrt(vW{l} / c2) + o.epsilon);
            mb{l} = o.beta1 * mb{l} + (1 - o.beta1) * gb{l};
            vb{l} = o.beta2 * vb{l} + (1 - o.beta2) * gb{l} .^ 2;
            b{l} = b{l} - o.lr * (mb{l} / c1) ./ (sqrt(vb{l} / c2) + o.epsilon);
        end
    end
    loss(ep) = acc / N;
end
mdl = struct('W', {W}, 'b', {b}, 'act', {act}, 'sizes', sz, 'loss', loss, 'opts', o);
end
