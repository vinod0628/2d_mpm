function M = reg_mlp()
%REG_MLP Multilayer perceptron (feed-forward neural network) ensemble for grade estimation.
%
%   Architecture : fully connected feed-forward network with LayerSizes hidden layers and a
%                  linear output unit.
%   Training     : backpropagated gradients of the mean-squared error + L2 weight decay
%                  (Lambda), minimised with L-BFGS - fitrnet. The response (model scale, e.g.
%                  log10 grade) is z-scored inside the method.
%   Ensemble     : NumNets networks (different initialisations, optional bootstrap); prediction
%                  = mean, uncertainty map = standard deviation across networks (model scale).
%                  This spread is model (epistemic) uncertainty only - it does not include the
%                  sample noise, unlike the RVR predictive SD.
%
%   Hyper-parameters (CV-tuned: LayerSizes x Lambda): LayerSizes, Activation ('tanh' | 'relu' |
%   'sigmoid'), Lambda, NumNets / NumNetsTuning, Bootstrap, IterationLimit - see clf_mlp.

M.key = 'mlp';
M.name = 'Neural network regression (MLP)';
M.task = 'regression';
M.description = 'Feed-forward MLP ensemble, MSE + L2, L-BFGS (fitrnet).';
M.defaults = struct('LayerSizes', 10, 'Activation', 'tanh', 'Lambda', 1e-2, 'NumNets', 10, ...
    'NumNetsTuning', 2, 'Bootstrap', false, 'IterationLimit', 1000);
% small Lambda needs many L-BFGS iterations and over-fits noisy grades -> grid starts at 0.01
M.grid = struct('LayerSizes', {{5, 10, 20}}, 'Lambda', [1e-2 1e-1 1]);
M.has_uncertainty = true;
M.train = @fit_model;
M.predict = @predict_value;
M.native_importance = [];
M.describe = @describe;
end

function mdl = fit_model(X, y, p, ctx)
X = double(X); y = double(y); n = numel(y);
mdl.y_mean = mean(y); mdl.y_sd = std(y); if mdl.y_sd < eps, mdl.y_sd = 1; end
z = (y - mdl.y_mean) / mdl.y_sd;
K = p.NumNets; if ~ctx.final, K = p.NumNetsTuning; end
mdl.nets = cell(K, 1); iters = zeros(K, 1);
for k = 1:K
    idx = (1:n)'; if p.Bootstrap, idx = randi(n, n, 1); end
    net = fitrnet(X(idx, :), z(idx), 'LayerSizes', p.LayerSizes, 'Activations', p.Activation, ...
        'Lambda', p.Lambda, 'Standardize', false, 'IterationLimit', p.IterationLimit);
    iters(k) = numel(net.TrainingHistory.Iteration);
    mdl.nets{k} = compact(net);
end
mdl.iterations = iters;
end

function [y, u] = predict_value(mdl, X)
X = double(X);
K = numel(mdl.nets);
P = zeros(size(X, 1), K);
for k = 1:K, P(:, k) = predict(mdl.nets{k}, X); end
y = mean(P, 2) * mdl.y_sd + mdl.y_mean;
if nargout > 1, u = std(P, 0, 2) * mdl.y_sd; end
end

function txt = describe(mdl, p)
txt = sprintf('%d x MLP [%s] %s, Lambda = %.3g, L-BFGS iterations %d-%d', numel(mdl.nets), ...
    num2str(p.LayerSizes), p.Activation, p.Lambda, min(mdl.iterations), max(mdl.iterations));
if p.Bootstrap, txt = [txt ', bootstrap']; end
end
