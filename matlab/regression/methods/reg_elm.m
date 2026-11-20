function M = reg_elm()
%REG_ELM Extreme learning machine (ELM) ensemble for grade estimation.
%
%   Network  : single hidden layer of NumHidden random nodes g(w.x + b), linear output node
%              (Huang, Zhu & Siew 2006); only the output weights are learned, analytically.
%   Solution : regularised ELM (Huang et al. 2012), squared error on the model scale
%              (e.g. log10 grade, z-scored inside the method):
%                N >= L:  beta = (I/C + H'H)^-1 H'T      N < L:  beta = H'(I/C + HH')^-1 T
%   Ensemble : NumNets ELMs with independent random hidden layers; prediction = mean, uncertainty
%              map = SD across the members (model scale). As for the MLP this is the spread due
%              to the random hidden layers (model uncertainty), not the sample noise.
%
%   Hyper-parameters (CV-tuned: NumHidden x C): NumHidden, C (Inf = pseudo-inverse ELM),
%   Activation ('sigmoid' | 'tanh' | 'sine' | 'hardlim' | 'tribas' | 'radbas'), InputScale,
%   NumNets / NumNetsTuning - see clf_elm and common/model/mpm_elm_train.m.
%
%   References: Huang, G.-B., Zhu, Q.-Y., Siew, C.-K., 2006. Neurocomputing 70, 489-501.
%   Huang, G.-B., Zhou, H., Ding, X., Zhang, R., 2012. IEEE Trans. Syst. Man Cybern. B 42, 513-529.

M.key = 'elm';
M.name = 'Extreme learning machine regression';
M.task = 'regression';
M.description = 'Regularised ELM ensemble (random hidden layer, analytic output weights), squared error.';
M.defaults = struct('NumHidden', 100, 'C', 1, 'Activation', 'sigmoid', 'InputScale', 1, ...
    'NumNets', 20, 'NumNetsTuning', 5);
M.grid = struct('NumHidden', [25 50 100 200 400], 'C', [1e-2 1e-1 1 10 100 1000]);
M.has_uncertainty = true;
M.train = @fit_model;
M.predict = @predict_value;
M.native_importance = [];
M.describe = @describe;
end

function mdl = fit_model(X, y, p, ctx)
X = double(X); y = double(y);
mdl.y_mean = mean(y); mdl.y_sd = std(y); if mdl.y_sd < eps, mdl.y_sd = 1; end
z = (y - mdl.y_mean) / mdl.y_sd;
K = p.NumNets; if ~ctx.final, K = p.NumNetsTuning; end
o = struct('n_hidden', p.NumHidden, 'activation', p.Activation, 'C', p.C, 'input_scale', p.InputScale);
mdl.nets = cell(K, 1);
for k = 1:K, mdl.nets{k} = mpm_elm_train(X, z, o); end
mdl.form = mdl.nets{1}.form;
end

function [y, u] = predict_value(mdl, X)
X = double(X);
K = numel(mdl.nets);
P = zeros(size(X, 1), K);
for k = 1:K, P(:, k) = mpm_elm_predict(mdl.nets{k}, X); end
y = mean(P, 2) * mdl.y_sd + mdl.y_mean;
if nargout > 1, u = std(P, 0, 2) * mdl.y_sd; end
end

function txt = describe(mdl, p)
txt = sprintf('%d ELMs x %d %s hidden nodes, C = %.4g, InputScale = %.3g, %s solution', ...
    numel(mdl.nets), p.NumHidden, p.Activation, p.C, p.InputScale, mdl.form);
end
