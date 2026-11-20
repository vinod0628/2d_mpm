function [th, hist] = mpm_cnn_train(X, y, c, o)
%MPM_CNN_TRAIN Train one patch CNN (MPM_CNN_NET) by mini-batch Adam on CPU or GPU.
%
%   [theta, hist] = mpm_cnn_train(X, y, c, opts)
%   X  C x N x w x w single patches (MPM_CNN_NET layout);  y  1 x N (0/1);  c  1 x N sample weights (class balance)
%   opts: filters [F1 F2] ([16 32]), epochs (30), batch (128), lr (1e-3), l2 (1e-4), dropout (0.5),
%         drop_rate (0.05)  random-drop augmentation: each input entry of a training patch is set
%                           to 0 (the training mean) with this probability, drawn anew for every
%                           mini-batch (Li et al. 2021, "random-drop"; random zero noise)
%         rotate (false)    random rot90 / flips of each mini-batch (off by default: geological
%                           structures have a preferred orientation)
%         gpu (false)       run on the GPU (gpuArray); results are gathered back to the CPU
%   Adam (Kingma & Ba 2015): beta1 0.9, beta2 0.999, epsilon 1e-8, bias-corrected moments.
%   hist: mean training loss per epoch.
o = mpm_merge(struct('filters', [16 32], 'epochs', 30, 'batch', 128, 'lr', 1e-3, 'l2', 1e-4, ...
    'dropout', 0.5, 'drop_rate', 0.05, 'rotate', false, 'gpu', false, ...
    'beta1', 0.9, 'beta2', 0.999, 'epsilon', 1e-8), o);
N = size(X, 2); C = size(X, 1);
like = zeros(1, 'single'); if o.gpu, like = gpuArray(like); end
th = mpm_cnn_net('init', C, o.filters(1), o.filters(2), like);
fn = fieldnames(th);
for i = 1:numel(fn), M.(fn{i}) = zeros(size(th.(fn{i})), 'like', like); V.(fn{i}) = M.(fn{i}); end
y = single(y(:)'); c = single(c(:)');
F2 = o.filters(2); t = 0;
hist = zeros(o.epochs, 1);
for ep = 1:o.epochs
    ord = randperm(N); Ls = 0; nb = 0;
    for s = 1:o.batch:N
        i = ord(s:min(s + o.batch - 1, N)); B = numel(i);
        Xb = X(:, i, :, :);
        if o.gpu, Xb = gpuArray(Xb); end
        if o.drop_rate > 0, Xb = Xb .* (rand(size(Xb), 'like', Xb) >= o.drop_rate); end
        if o.rotate, Xb = rot_flip(Xb); end
        m = [];
        if o.dropout > 0, m = (rand(F2, B, 'like', like) >= o.dropout) / (1 - o.dropout); end
        yb = y(i); cb = c(i);
        if o.gpu, yb = gpuArray(yb); cb = gpuArray(cb); end
        [L, g] = mpm_cnn_net('loss', th, Xb, yb, cb, o.l2, m);
        t = t + 1;
        for k = 1:numel(fn)                              % Adam update
            f = fn{k};
            M.(f) = o.beta1 * M.(f) + (1 - o.beta1) * g.(f);
            V.(f) = o.beta2 * V.(f) + (1 - o.beta2) * g.(f) .^ 2;
            th.(f) = th.(f) - o.lr * (M.(f) / (1 - o.beta1 ^ t)) ./ (sqrt(V.(f) / (1 - o.beta2 ^ t)) + o.epsilon);
        end
        Ls = Ls + double(gather(L)); nb = nb + 1;
    end
    hist(ep) = Ls / nb;
end
for k = 1:numel(fn), th.(fn{k}) = gather(th.(fn{k})); end
end

function X = rot_flip(X)
% random multiple of 90 degrees and random mirror of C x B x w x w patches (spatial dims 3, 4)
for k = 1:randi(4) - 1
    X = flip(permute(X, [1 2 4 3]), 3);
end
if rand < 0.5, X = flip(X, 4); end
end
