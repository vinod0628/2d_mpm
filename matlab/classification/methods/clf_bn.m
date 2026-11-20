function M = clf_bn()
%CLF_BN Bayesian network classifiers for prospectivity mapping (Porwal, Carranza & Hale 2006).
%
%   Nodes    : one node per EVIDENCE LAYER (as the predictor maps of Porwal et al. 2006): a
%              categorical layer (one-hot columns 'layer=class') is ONE multi-state node (classes +
%              'missing'); a continuous layer is discretised on the training data of every fit
%              (Fayyad & Irani 1993 MDL, or equal-frequency classes). Layer membership comes from
%              mpm_feature_info (set by the pipeline); without it every column is a continuous node.
%   Structure: 'naive'     naive Bayes - complete conditional independence given the class
%                          (with binary evidence = weights of evidence; Bonham-Carter et al. 1989)
%              'tan'       augmented naive Bayes as tree-augmented naive Bayes (Friedman, Geiger &
%                          Goldszmidt 1997): max spanning tree of I(A_i; A_j | C), each node gets
%                          the class and at most one other node as parents
%              'selective' selective naive Bayes: only conditionally independent predictors
%                          (G-test on I(A_i; A_j | C), Porwal et al. 2006 definition)
%   Parameters: Dirichlet-smoothed conditional probability tables (N0, Friedman et al. 1997).
%   Output   : posterior P(deposit | evidence) with a uniform class prior (BalanceClasses).
%   Uncertainty map: SD of the posterior over Draws samples of the Dirichlet parameter posteriors.
%   Native importance: mutual information I(node; class) on the training data (bits).
%   Engine   : common/model/mpm_bn_discretize.m, mpm_bn_train.m, mpm_bn_predict.m (own code;
%              MATLAB has naive Bayes (fitcnb) but no TAN / selective classifiers or MDL binning).
%
%   Hyper-parameters (CV grid: Structure x Discretization):
%     Structure        'naive' | 'tan' | 'selective'
%     Discretization   'mdl' | 'equalfreq';  NumBins (equalfreq)
%     N0               Dirichlet smoothing strength (5);  Alpha  G-test level ('selective', 0.05)
%     BalanceClasses   uniform class prior;  Draws  posterior draws for the uncertainty map (50)
%
%   References: Porwal, A., Carranza, E.J.M., Hale, M., 2006. Bayesian network classifiers for
%   mineral potential mapping. Comput. Geosci. 32, 1-16.  Friedman, N., Geiger, D., Goldszmidt, M.,
%   1997. Bayesian network classifiers. Mach. Learn. 29, 131-163.  Fayyad, U.M., Irani, K.B., 1993.
%   Multi-interval discretization of continuous-valued attributes for classification learning.
%   IJCAI-93, 1022-1027.

M.key = 'bn';
M.name = 'Bayesian network';
M.task = 'classification';
M.description = 'Bayesian network classifiers (naive / TAN / selective) on discretised evidence layers.';
M.defaults = struct('Structure', 'tan', 'Discretization', 'mdl', 'NumBins', 5, 'N0', 5, 'Alpha', 0.05, ...
    'BalanceClasses', true, 'Draws', 50);
M.grid = struct('Structure', {{'naive', 'tan', 'selective'}}, 'Discretization', {{'mdl', 'equalfreq'}});
M.score_is_probability = true;
M.has_uncertainty = true;
M.train = @fit_model;
M.predict = @predict_score;
M.native_importance = @native_importance;
M.describe = @describe;
end

% =====================================================================================
function mdl = fit_model(X, y, p, ctx)
X = double(X); y = logical(y);
nodes = make_nodes(size(X, 2));
for k = 1:numel(nodes)                                     % discretise continuous nodes on training data
    if strcmp(nodes(k).type, 'continuous')
        nodes(k).cuts = mpm_bn_discretize(X(:, nodes(k).cols), y, p.Discretization, p.NumBins);
        nodes(k).r = numel(nodes(k).cuts) + 1;
    end
end
mdl.nodes = nodes;
D = encode(mdl, X);
prior = 'empirical'; if p.BalanceClasses, prior = 'uniform'; end
draws = 0; if ctx.final, draws = p.Draws; end
mdl.bn = mpm_bn_train(D, [nodes.r], y, struct('structure', p.Structure, 'N0', p.N0, 'alpha', p.Alpha, ...
    'prior', prior, 'draws', draws));
end

function [s, u] = predict_score(mdl, X)
D = encode(mdl, double(X));
if nargout > 1
    [s, u] = mpm_bn_predict(mdl.bn, D);
else
    s = mpm_bn_predict(mdl.bn, D);
end
end

function nodes = make_nodes(P)
info = mpm_feature_info();
nodes = struct('name', {}, 'type', {}, 'cols', {}, 'r', {}, 'cuts', {}, 'mu', {}, 'sigma', {}, 'classes', {});
if isempty(info) || numel(info.names) ~= P                 % outside the pipeline: one node per column
    for j = 1:P
        nodes(end+1) = struct('name', sprintf('x%d', j), 'type', 'continuous', 'cols', j, 'r', 1, ...
            'cuts', [], 'mu', 0, 'sigma', 1, 'classes', {{}}); %#ok<AGROW>
    end
    return
end
for L = unique(info.source(:))'
    cols = find(info.source == L);
    if all(info.onehot(cols))                              % categorical layer -> one multi-state node
        cls = regexprep(string(info.names(cols)), '^[^=]*=', '');
        nodes(end+1) = struct('name', info.layers{L}, 'type', 'categorical', 'cols', cols, ...
            'r', numel(cols) + 1, 'cuts', [], 'mu', double(info.mu(cols)), 'sigma', double(info.sigma(cols)), ...
            'classes', {cellstr(cls)}); %#ok<AGROW>
    else
        for j = cols
            nm = info.layers{L}; if numel(cols) > 1, nm = info.names{j}; end
            nodes(end+1) = struct('name', nm, 'type', 'continuous', 'cols', j, 'r', 1, 'cuts', [], ...
                'mu', 0, 'sigma', 1, 'classes', {{}}); %#ok<AGROW>
        end
    end
end
end

function D = encode(mdl, X)
% discrete state of every node: categorical = class index (last state = no class / missing);
% continuous = interval of the training cut points
n = numel(mdl.nodes); D = ones(size(X, 1), n);
for k = 1:n
    nd = mdl.nodes(k);
    switch nd.type
        case 'categorical'
            raw = X(:, nd.cols) .* nd.sigma + nd.mu > 0.5;  % recover the 0/1 one-hot values
            [has, idx] = max(raw, [], 2);
            idx(~has) = nd.r;
            D(:, k) = idx;
        case 'continuous'
            if ~isempty(nd.cuts)
                D(:, k) = discretize(X(:, nd.cols), [-Inf, nd.cuts, Inf], 'IncludedEdge', 'right');
            end
    end
end
D(isnan(D)) = 1;
end

function T = native_importance(mdl, ~)
T = table(string({mdl.nodes.name})', mdl.bn.mi_class(:), [mdl.nodes.r]', mdl.bn.use(:), ...
    'VariableNames', {'feature', 'importance', 'states', 'used'});
T = sortrows(T, 'importance', 'descend');
T.Properties.Description = 'Bayesian network: mutual information I(node; class) on the training data (bits)';
end

function txt = describe(mdl, p)
nd = mdl.nodes; b = mdl.bn;
st = strjoin(arrayfun(@(k) sprintf('%s:%d', nd(k).name, nd(k).r), 1:numel(nd), 'UniformOutput', false), ', ');
txt = sprintf('%s, %s discretisation, N0 = %g, %d nodes (states: %s)', upper(p.Structure), p.Discretization, ...
    p.N0, numel(nd), st);
switch b.structure
    case 'tan'
        e = find(b.parent > 0);
        if ~isempty(e)
            txt = sprintf('%s; TAN arcs: %s', txt, strjoin(arrayfun(@(k) sprintf('%s->%s', nd(b.parent(k)).name, ...
                nd(k).name), e, 'UniformOutput', false), ', '));
        end
    case 'selective'
        txt = sprintf('%s; dropped (conditionally dependent): %s', txt, strjoin({nd(~b.use).name}, ', '));
end
end
