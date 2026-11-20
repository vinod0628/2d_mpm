function M = clf_cnn()
%CLF_CNN Convolutional neural network on image patches for prospectivity mapping.
%
%   Input    : a w x w window (PatchSize, odd) of all z-scored evidence layers centred on a cell
%              (M.input = 'patch': the pipeline passes cell indices; the windows are cut from
%              the evidence cube, mpm_patch_source). The CNN sees the spatial context of a cell -
%              contacts, gradients, structures near it - not only its own values (2-D patch CNNs
%              for MPM: Li, Chen & Xiang 2020; Li et al. 2021; Yang et al. 2022; Ding et al. 2023).
%   Network  : LeNet-type: conv 3x3 (F1) + ReLU -> conv 3x3 (F2) + ReLU (valid) -> global average
%              pooling -> dropout -> linear -> sigmoid; weighted cross-entropy + L2; Adam
%              (common/model/mpm_cnn_net.m, mpm_cnn_train.m - own im2col implementation, runs on
%              CPU and NVIDIA GPU).
%   Samples  : sliding-window augmentation (Ding et al. 2023): every window that contains a
%              training deposit is a positive, i.e. centres shifted by up to Shift cells; the
%              background cells are shifted the same way, dropping windows that contain a training
%              deposit. Random-drop augmentation (Li et al. 2021): input entries set to 0 with
%              probability DropRate, anew in every mini-batch. Rotation / flips optional (off).
%   Map      : fully convolutional evaluation (mpm_cnn_map) = identical to a window per cell.
%   Ensemble : NumNets networks (random initialisation, mini-batch order, drop masks); score =
%              mean probability, uncertainty map = standard deviation (Lakshminarayanan et al. 2017).
%   CV       : M.spatial_reach -> the pipeline keeps samples whose windows can overlap in the same
%              CV fold (avoids the spatial leakage of overlapping patches).
%
%   Hyper-parameters (CV grid: PatchSize):
%     PatchSize        w, odd >= 5 (cells)
%     Shift            max centre shift of the sliding-window augmentation ('auto' = (w-1)/2: all
%                      windows containing the cell; 0 = centred windows only)
%     Filters          [F1 F2];  Dropout;  L2;  Epochs;  MiniBatchSize;  LearnRate
%     DropRate         random-drop probability per input entry (0.05; review: 5-10 %)
%     Rotate           random rot90 / flip augmentation
%     NumNets, NumNetsTuning   ensemble size (final / inside CV)
%     MaxWindows       cap on augmented windows per class
%     ExecutionEnvironment     'auto' (GPU if an NVIDIA GPU is usable) | 'gpu' | 'cpu'
%
%   References: Li, S., Chen, J., Xiang, J., 2020. Neural Comput. Appl. 32, 2037-2053.  Li, T.,
%   Zuo, R., Xiong, Y., Peng, Y., 2021. Nat. Resour. Res. 30, 27-38.  Yang, N., Zhang, Z., Yang, J.,
%   Hong, Z., 2022. Comput. Geosci. 161, 105075.  Ding, K., Xue, L., Ran, X., Wang, J., Yan, Q.,
%   2023. Minerals 13, 730.  LeCun, Y., Bottou, L., Bengio, Y., Haffner, P., 1998. Proc. IEEE 86,
%   2278-2324.  Long, J., Shelhamer, E., Darrell, T., 2015. CVPR, 3431-3440.

M.key = 'cnn';
M.name = 'CNN (image patches)';
M.task = 'classification';
M.input = 'patch';
M.description = 'LeNet-type CNN on w x w evidence windows, sliding-window + random-drop augmentation, ensemble.';
M.defaults = struct('PatchSize', 9, 'Shift', 'auto', 'Filters', [16 32], 'Dropout', 0.5, 'L2', 1e-4, ...
    'Epochs', 15, 'MiniBatchSize', 128, 'LearnRate', 1e-3, 'DropRate', 0.05, 'Rotate', false, ...
    'NumNets', 5, 'NumNetsTuning', 1, 'MaxWindows', 4000, 'ExecutionEnvironment', 'auto');
M.grid = struct('PatchSize', [5 9 13]);
M.score_is_probability = true;
M.has_uncertainty = true;
M.train = @fit_model;
M.predict = @predict_score;
M.predict_map = @predict_map;
M.spatial_reach = @spatial_reach;
M.native_importance = [];
M.describe = @describe;
end

% =====================================================================================
function mdl = fit_model(lin, y, p, ctx)
src = mpm_patch_source.active();
if isempty(src), error('mpm:cnn', 'No patch source: run clf_cnn through mpm_classification_run.'); end
w = round(p.PatchSize);
if w < 5 || mod(w, 2) == 0, error('mpm:cnn', 'PatchSize must be odd and >= 5.'); end
h = (w - 1) / 2; s = shift_of(p, h);
lin = lin(:); y = logical(y(:));
% ---- sliding-window augmentation: centres shifted by up to s cells
[dr, dc] = ndgrid(-s:s, -s:s); dr = dr(:); dc = dc(:);
[r, c] = ind2sub([src.H src.W], lin);
R = r + dr'; Cc = c + dc';                                   % N x nShift centres
ok = R >= 1 & R <= src.H & Cc >= 1 & Cc <= src.W;
cen = zeros(size(R)); cen(ok) = R(ok) + (Cc(ok) - 1) * src.H;
ok(ok) = src.valid(cen(ok));                                 % centre inside the study area
dPos = src.chess_distance(lin(y));                           % distance to training deposits
neg = ~y & ok; neg(neg) = dPos(cen(neg)) > h;                % background windows without a deposit
ok(~y, :) = neg(~y, :);
[iS, iO] = find(ok);                                         % sample, shift of every kept window
yy = y(iS);
% ---- cap per class
sel = [cap(find(yy), p.MaxWindows); cap(find(~yy), p.MaxWindows)]; %#ok<FNDSB>
iS = iS(sel); iO = iO(sel); yy = yy(sel);
X = permute(src.extract(lin(iS), w, dr(iO), dc(iO)), [3 4 1 2]);   % C x N x w x w
n1 = sum(yy); n0 = sum(~yy);
cw = ones(numel(yy), 1); cw(yy) = numel(yy) / (2 * n1); cw(~yy) = numel(yy) / (2 * n0);
K = p.NumNets; if ~ctx.final, K = p.NumNetsTuning; end
gpu = use_gpu(p.ExecutionEnvironment);
o = struct('filters', p.Filters, 'epochs', p.Epochs, 'batch', p.MiniBatchSize, 'lr', p.LearnRate, ...
    'l2', p.L2, 'dropout', p.Dropout, 'drop_rate', p.DropRate, 'rotate', p.Rotate, 'gpu', gpu);
mdl.nets = cell(K, 1); mdl.loss = zeros(K, 1);
for k = 1:K
    [mdl.nets{k}, hist] = mpm_cnn_train(X, yy', cw', o);
    mdl.loss(k) = hist(end);
end
mdl.w = w; mdl.shift = s; mdl.n_windows = [n1 n0]; mdl.n_samples = [sum(y) sum(~y)];
mdl.gpu = gpu;
end

function [s, u] = predict_score(mdl, lin)
src = mpm_patch_source.active();
lin = lin(:); n = numel(lin); K = numel(mdl.nets);
Pr = zeros(n, K);
for i = 1:4096:n
    j = i:min(i + 4095, n);
    X = permute(src.extract(lin(j), mdl.w), [3 4 1 2]);
    for k = 1:K
        Pr(j, k) = 1 ./ (1 + exp(-double(mpm_cnn_net('forward', mdl.nets{k}, X, []))));
    end
end
s = mean(Pr, 2);
if nargout > 1, u = std(Pr, 0, 2); end
end

function [s, u] = predict_map(mdl, F)
% whole map at once, fully convolutional; returns values for the valid cells F.idx
src = mpm_patch_source.active();
Z = mpm_cnn_map(mdl.nets, src, mdl.w, mdl.gpu);
Z = reshape(double(Z), [], size(Z, 3));
Pr = 1 ./ (1 + exp(-Z(F.idx, :)));
s = mean(Pr, 2); u = std(Pr, 0, 2);
end

function r = spatial_reach(p, grid)
% Chebyshev distance (cells) within which two samples' training / prediction windows can overlap
w = p.PatchSize;
if isstruct(grid) && isfield(grid, 'PatchSize'), w = max([w, grid.PatchSize(:)']); end
h = (w - 1) / 2; s = shift_of(p, h);
r = 2 * h + s;
end

function s = shift_of(p, h)
if ischar(p.Shift) || isstring(p.Shift), s = h; else, s = min(round(p.Shift), h); end
end

function i = cap(i, m)
if numel(i) > m, i = i(sort(randperm(numel(i), m))); end
i = i(:);
end

function g = use_gpu(env)
switch lower(env)
    case 'cpu', g = false;
    case 'gpu'
        if ~canUseGPU, error('mpm:cnn', 'ExecutionEnvironment = gpu, but no usable NVIDIA GPU (Parallel Computing Toolbox + CUDA driver).'); end
        g = true;
    otherwise, g = canUseGPU;
end
end

function txt = describe(mdl, p)
dev = 'CPU'; if mdl.gpu, dev = 'GPU'; end
txt = sprintf(['%d CNNs on %dx%d windows (conv3x3 %d + conv3x3 %d, GAP, dropout %.2g), %d epochs Adam, ' ...
    'L2 %.2g; windows: %d deposit + %d background (sliding shift %d, random-drop %.2g) from %d + %d samples; %s'], ...
    numel(mdl.nets), mdl.w, mdl.w, p.Filters(1), p.Filters(2), p.Dropout, p.Epochs, p.L2, mdl.n_windows(1), ...
    mdl.n_windows(2), mdl.shift, p.DropRate, mdl.n_samples(1), mdl.n_samples(2), dev);
end
