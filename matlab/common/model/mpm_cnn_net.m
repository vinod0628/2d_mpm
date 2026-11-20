function varargout = mpm_cnn_net(what, varargin)
%MPM_CNN_NET Small patch CNN (LeNet type) written with im2col matrix products: forward, exact
%   back-propagation, initialisation. Plain array operations only, so the same code runs on the
%   CPU (single) and on an NVIDIA GPU (gpuArray single).
%
%   Architecture (input patch of w x w cells and C evidence channels, w >= 5):
%     conv 3x3 valid, F1 filters, ReLU -> conv 3x3 valid, F2 filters, ReLU
%     -> global average pooling over the (w-4) x (w-4) map -> dropout -> linear -> logit z
%   P(deposit | patch) = sigmoid(z). Valid convolutions + global average pooling make the network
%   equal to a fully convolutional network: the whole map is scored by convolving the full
%   evidence cube once (MPM_CNN_MAP), identical to cutting a patch around every cell.
%
%   Data layout: X is C x B x w x w (channels, batch, rows = dy, columns = dx): every 3x3
%   neighbourhood slice is a set of contiguous C x B blocks (fast im2col on CPU and GPU).
%   theta = mpm_cnn_net('init', C, F1, F2, like)       Glorot-uniform weights, zero biases
%   z = mpm_cnn_net('forward', theta, X, dropMask)     1 x B logits; dropMask F2 x B or []
%   [L, grad] = mpm_cnn_net('loss', theta, X, y, c, lambda, dropMask)
%       L = sum_b c_b [softplus(z_b) - y_b z_b] / sum_b c_b + lambda/2 (|W1|^2 + |W2|^2 + |v|^2)
%   Weight layout: W1 (F1 x 9C), W2 (F2 x 9F1); column index c + C (k - 1) for channel c and
%   offset k = 1..9 ordered dy fastest, then dx (cross-correlation, as in CNN libraries).
switch what
    case 'init'
        [C, F1, F2, like] = varargin{:};
        g = @(sz, fin, fout) (2 * rand(sz, 'like', like) - 1) * sqrt(6 / (fin + fout));
        th.W1 = g([F1, 9 * C], 9 * C, 9 * F1);   th.b1 = zeros(F1, 1, 'like', like);
        th.W2 = g([F2, 9 * F1], 9 * F1, 9 * F2); th.b2 = zeros(F2, 1, 'like', like);
        th.v = g([1, F2], F2, 1);                th.c = zeros(1, 1, 'like', like);
        varargout = {th};
    case 'forward'
        [th, X, m] = varargin{:};
        varargout = {forward(th, X, m)};
    case 'loss'
        [th, X, y, c, lambda, m] = varargin{:};
        [z, k] = forward(th, X, m);
        sc = sum(c);
        L = sum(c .* (max(z, 0) + log(1 + exp(-abs(z))) - y .* z)) / sc + ...
            lambda / 2 * (sum(th.W1(:) .^ 2) + sum(th.W2(:) .^ 2) + sum(th.v .^ 2));
        dz = c .* (1 ./ (1 + exp(-z)) - y) / sc;                       % 1 x B
        gr.v = dz * k.gd' + lambda * th.v;  gr.c = sum(dz);
        dg = th.v' * dz;                                               % F2 x B
        if ~isempty(m), dg = dg .* m; end
        B = size(dz, 2); F2 = size(th.W2, 1); F1 = size(th.W1, 1);
        dZ2 = reshape(repmat(dg / k.n2, 1, 1, k.n2), F2, []) .* (k.Z2 > 0);
        gr.W2 = dZ2 * k.Col2' + lambda * th.W2;  gr.b2 = sum(dZ2, 2);
        dA1 = col2im3(th.W2' * dZ2, F1, k.s1, B);                     % F1 x B x s1 x s1
        dZ1 = reshape(dA1, F1, []) .* (k.Z1 > 0);
        gr.W1 = dZ1 * k.Col1' + lambda * th.W1;  gr.b1 = sum(dZ1, 2);
        varargout = {L, gr};
end
end

function [z, k] = forward(th, X, m)
[~, B, w, ~] = size(X);
k.s1 = w - 2; s2 = w - 4; k.n2 = s2 ^ 2;
F1 = size(th.W1, 1); F2 = size(th.W2, 1);
k.Col1 = im2col3(X);
k.Z1 = th.W1 * k.Col1 + th.b1;                                         % F1 x (s1 s1 B)
k.Col2 = im2col3(reshape(max(k.Z1, 0), F1, B, k.s1, k.s1));
k.Z2 = th.W2 * k.Col2 + th.b2;                                         % F2 x (s2 s2 B)
g = sum(reshape(max(k.Z2, 0), F2, B, k.n2), 3) / k.n2;              % global average pool
if ~isempty(m), g = g .* m; end
k.gd = g;
z = th.v * g + th.c;                                                   % 1 x B
end

function Col = im2col3(X)
% (9C) x (B oy ox) matrix of all 3x3 neighbourhoods of a valid convolution, X: C x B x Sy x Sx
[C, ~, Sy, Sx] = size(X); oy = Sy - 2; ox = Sx - 2;
blk = cell(9, 1); k = 0;
for dx = 0:2
    for dy = 0:2
        k = k + 1;
        blk{k} = reshape(X(:, :, 1+dy:oy+dy, 1+dx:ox+dx), C, []);   % contiguous C x B blocks
    end
end
Col = vertcat(blk{:});                                  % row c + C (k - 1)
end

function dX = col2im3(dCol, C, s, B)
% adjoint of im2col3 for a C x B x s x s input
o = s - 2;
dX = zeros(C, B, s, s, 'like', dCol);
k = 0;
for dx = 0:2
    for dy = 0:2
        k = k + 1;
        dX(:, :, 1+dy:o+dy, 1+dx:o+dx) = dX(:, :, 1+dy:o+dy, 1+dx:o+dx) + reshape(dCol((k - 1) * C + (1:C), :), C, B, o, o);
    end
end
end
