function Z = mpm_cnn_map(nets, src, w, gpu, tileRows)
%MPM_CNN_MAP Score every grid cell with patch CNNs, fully convolutionally (Long et al. 2015).
%
%   Z = mpm_cnn_map(nets, src, w, gpu)     nets: cell array of MPM_CNN_NET parameter structs
%   Z  H x W x K logits (one page per network) for a w x w window centred on every cell.
%   Because the network has only valid 3x3 convolutions followed by global average pooling, the
%   result equals cutting a w x w patch around each cell and running MPM_CNN_NET('forward') on it
%   (checked to ~1e-7), but costs one convolution of the whole cube instead of H*W patches.
%   The cube is processed in tiles of tileRows output rows (default 64) to bound memory; with
%   gpu = true each tile is computed on the GPU.
if nargin < 4, gpu = false; end
if nargin < 5 || isempty(tileRows), tileRows = 64; end
h = (w - 1) / 2; K = numel(nets);
H = src.H; W = src.W;
pad = zeros(H + 2 * h, W + 2 * h, src.nFeat, 'single');
pad(h+1:h+H, h+1:h+W, :) = src.cube;
Z = zeros(H, W, K, 'single');
q = w - 4;                                            % pooling window on the conv-2 map
for r0 = 1:tileRows:H
    r1 = min(r0 + tileRows - 1, H); T = r1 - r0 + 1;
    X = pad(r0:r1 + 2 * h, :, :);                     % (T + w - 1) x (W + w - 1) x C
    if gpu, X = gpuArray(X); end
    for k = 1:K
        th = nets{k};
        if gpu, th = structfun(@gpuArray, th, 'UniformOutput', false); end
        A1 = conv_valid(X, th.W1, th.b1);
        A2 = conv_valid(A1, th.W2, th.b2);            % (T + q - 1) x (W + q - 1) x F2
        F2 = size(A2, 3); G = zeros(T, W, F2, 'like', A2);
        bc = ones(q, 1, 'like', A2) / q; br = ones(1, q, 'like', A2) / q;
        for f = 1:F2                                  % separable q x q box mean = global avg pool
            G(:, :, f) = conv2(br, bc, A2(:, :, f), 'valid');
        end
        z = reshape(reshape(G, [], F2) * th.v' + th.c, T, W);
        Z(r0:r1, :, k) = gather(z);
    end
end
end

function A = conv_valid(X, Wt, b)
% valid 3x3 convolution + ReLU of one image X (Sy x Sx x C) with MPM_CNN_NET weights
[Sy, Sx, C] = size(X); oy = Sy - 2; ox = Sx - 2; F = size(Wt, 1);
A = zeros(oy * ox, F, 'like', X) + b';
k = 0;
for dx = 0:2
    for dy = 0:2
        k = k + 1;
        A = A + reshape(X(1+dy:oy+dy, 1+dx:ox+dx, :), [], C) * Wt(:, (k - 1) * C + (1:C))';
    end
end
A = reshape(max(A, 0), oy, ox, F);
end
