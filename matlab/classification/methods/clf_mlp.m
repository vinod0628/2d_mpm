function M = clf_mlp()
%CLF_MLP Multilayer perceptron (feed-forward neural network) ensemble for prospectivity mapping.
%
%   Architecture : fully connected feed-forward network, LayerSizes hidden layers (e.g. 10 or
%                  [20 10]), Activation in the hidden layers, softmax output.
%   Training     : backpropagated gradients of the cross-entropy loss + L2 weight decay (Lambda),
%                  minimised with L-BFGS (quasi-Newton, full batch) - fitcnet, Statistics and
%                  Machine Learning Toolbox. Preferred over classic gradient-descent backprop
%                  (Singer & Kouda 1996) for MPM-size data: no learning rate / momentum to tune,
%                  converges reliably on a few hundred samples, and weight decay replaces early
%                  stopping (which would sacrifice scarce deposits for a validation set).
%   Ensemble     : NumNets networks from different random initialisations (optionally bootstrap
%                  samples). The score is the mean probability; the uncertainty map is the
%                  standard deviation across the networks (initialisation/bagging spread).
%
%   Hyper-parameters (tuned by CV: LayerSizes x Lambda):
%     LayerSizes      hidden-layer sizes: scalar (one layer) or vector (deep)
%     Activation      'tanh' | 'relu' | 'sigmoid'
%     Lambda          L2 regularisation strength (fitcnet scale ~ 1/number of samples)
%     NumNets         ensemble size of the final model; NumNetsTuning inside CV
%     Bootstrap       true -> each network sees a bootstrap sample (bagging)
%     BalanceClasses  weight samples so deposits and background count equally
%     IterationLimit  L-BFGS iterations
%
%   References: Singer, D.A., Kouda, R., 1996. Application of a feedforward neural network in
%   the search for Kuroko deposits. Math. Geol. 28, 1017-1023.  Porwal, A., Carranza, E.J.M.,
%   Hale, M., 2003. Artificial neural networks for mineral-potential mapping. Nat. Resour. Res.
%   12, 155-171.  Lakshminarayanan, B. et al., 2017. Simple and scalable predictive uncertainty
%   estimation using deep ensembles. NeurIPS.

M.key = 'mlp';
M.name = 'Neural network (MLP)';
M.task = 'classification';
M.description = 'Feed-forward MLP ensemble, cross-entropy + L2, L-BFGS (fitcnet).';
M.defaults = struct('LayerSizes', 10, 'Activation', 'tanh', 'Lambda', 1e-2, 'NumNets', 10, ...
    'NumNetsTuning', 3, 'Bootstrap', false, 'BalanceClasses', true, 'IterationLimit', 1000);
M.grid = struct('LayerSizes', {{5, 10, 20}}, 'Lambda', [1e-4 1e-3 1e-2 1e-1 1]);
M.score_is_probability = true;
M.has_uncertainty = true;
M.train = @fit_model;
M.predict = @predict_score;
M.native_importance = [];
M.describe = @describe;
end

function mdl = fit_model(X, y, p, ctx)
X = double(X); y = logical(y); n = numel(y);
w = ones(n, 1);
if p.BalanceClasses
    w(y) = n / (2 * max(sum(y), 1)); w(~y) = n / (2 * max(sum(~y), 1));
end
K = p.NumNets; if ~ctx.final, K = p.NumNetsTuning; end
mdl.nets = cell(K, 1); iters = zeros(K, 1);
for k = 1:K
    idx = (1:n)';
    if p.Bootstrap
        idx = randi(n, n, 1);
        while all(y(idx)) || ~any(y(idx)), idx = randi(n, n, 1); end   % keep both classes
    end
    net = fitcnet(X(idx, :), y(idx), 'LayerSizes', p.LayerSizes, 'Activations', p.Activation, ...
        'Lambda', p.Lambda, 'Weights', w(idx), 'Standardize', false, 'ClassNames', [false; true], ...
        'IterationLimit', p.IterationLimit);
    iters(k) = numel(net.TrainingHistory.Iteration);
    mdl.nets{k} = compact(net);
end
mdl.iterations = iters;
end

function [s, u] = predict_score(mdl, X)
X = double(X);
K = numel(mdl.nets);
P = zeros(size(X, 1), K);
for k = 1:K
    [~, sc] = predict(mdl.nets{k}, X);
    P(:, k) = sc(:, 2);
end
s = mean(P, 2);
if nargout > 1, u = std(P, 0, 2); end
end

function txt = describe(mdl, p)
txt = sprintf('%d x MLP [%s] %s, Lambda = %.3g, L-BFGS iterations %d-%d%s', numel(mdl.nets), ...
    num2str(p.LayerSizes), p.Activation, p.Lambda, min(mdl.iterations), max(mdl.iterations), ...
    bag_text(p.Bootstrap));
end

function s = bag_text(b)
if b, s = ', bootstrap'; else, s = ''; end
end
