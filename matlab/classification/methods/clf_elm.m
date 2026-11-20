function M = clf_elm()
%CLF_ELM Extreme learning machine (ELM) ensemble for prospectivity mapping.
%
%   Network  : single hidden layer of NumHidden RANDOM nodes g(w.x + b) (never trained) and one
%              linear output node (Huang, Zhu & Siew 2006). Only the output weights beta are
%              learned, in one linear solve - no back-propagation, no iterations.
%   Targets  : deposit = +1, background = -1 (binary ELM with a single output node; Huang et al.
%              2012, decision sign(f)). Chen & Wu (2017) used ELM in the same way - a regression
%              on the binary deposit / non-deposit label whose output ranks prospectivity.
%   Solution : regularised ELM  beta = (I/C + H'WH)^-1 H'WT  (N >= L)  or  H'(I/C + WHH')^-1 WT
%              (N < L), W = diagonal class weights of the weighted ELM (Zong, Huang & Chen 2013):
%                'W1'  w = 1/#class                     (classes count equally)
%                'W2'  w = 0.618/#class for the larger class, 1/#class for the smaller
%                'none' w = 1 (plain regularised ELM)
%              The weights are rescaled to mean 1 so that C keeps the meaning of the
%              unweighted ELM (only the ratio between classes matters for the solution).
%   Ensemble : NumNets ELMs with independent random hidden layers, outputs averaged - removes
%              most of the run-to-run variance due to the random nodes (ensemble ELMs: Liu &
%              Wang 2010; Cao et al. 2012). Uncertainty map = SD of the members' probabilities.
%   Output   : raw ensemble output during CV tuning (AUC only needs a ranking). Final model:
%              probability via a Platt (1999) sigmoid fitted to OUT-OF-FOLD ensemble outputs
%              (CalibFolds folds, buffered deposit cells kept in one fold), the same way
%              fitPosterior calibrates the SVM. The ELM output itself is a least-squares fit to
%              +/-1, not a probability.
%
%   Hyper-parameters (CV-tuned: NumHidden x C):
%     NumHidden       number of random hidden nodes L
%     C               regularisation (larger = closer fit; Inf = original pseudo-inverse ELM)
%     Activation      'sigmoid' (Huang 2006) | 'tanh' | 'sine' | 'hardlim' | 'tribas' | 'radbas'
%     InputScale      random input weights U(-a, a), a = InputScale*sqrt(3/P) (z-scored inputs)
%     Weighting       'W1' | 'W2' | 'none'   (weighted ELM scheme, see above)
%     NumNets         ensemble size of the final model; NumNetsTuning inside CV
%     CalibFolds      folds for the out-of-fold outputs of the Platt calibration
%
%   Implementation: common/model/mpm_elm_train.m, mpm_elm_predict.m, mpm_platt_fit.m
%   (MATLAB has no ELM).
%
%   References: Huang, G.-B., Zhu, Q.-Y., Siew, C.-K., 2006. Extreme learning machine: theory and
%   applications. Neurocomputing 70, 489-501.  Huang, G.-B., Zhou, H., Ding, X., Zhang, R., 2012.
%   Extreme learning machine for regression and multiclass classification. IEEE Trans. Syst. Man
%   Cybern. B 42, 513-529.  Zong, W., Huang, G.-B., Chen, Y., 2013. Weighted extreme learning
%   machine for imbalance learning. Neurocomputing 101, 229-242.  Chen, Y., Wu, W., 2017. Mapping
%   mineral prospectivity using an extreme learning machine regression. Ore Geol. Rev. 80,
%   200-213.  Liu, N., Wang, H., 2010. Ensemble based extreme learning machine. IEEE Signal
%   Process. Lett. 17, 754-757.  Cao, J., Lin, Z., Huang, G.-B., Liu, N., 2012. Voting based
%   extreme learning machine. Inf. Sci. 185, 66-77.  Platt, J.C., 1999 (see mpm_platt_fit).

M.key = 'elm';
M.name = 'Extreme learning machine';
M.task = 'classification';
M.description = 'Weighted regularised ELM ensemble (random hidden layer, analytic output weights), Platt-scaled.';
M.defaults = struct('NumHidden', 100, 'C', 1, 'Activation', 'sigmoid', 'InputScale', 1, ...
    'Weighting', 'W1', 'NumNets', 20, 'NumNetsTuning', 5, 'CalibFolds', 5);
M.grid = struct('NumHidden', [25 50 100 200 400], 'C', [1e-2 1e-1 1 10 100 1000]);
M.score_is_probability = true;
M.has_uncertainty = true;
M.train = @fit_model;
M.predict = @predict_score;
M.native_importance = [];          % random hidden layer -> permutation importance only
M.describe = @describe;
end

function mdl = fit_model(X, y, p, ctx)
X = double(X); y = logical(y);
K = p.NumNets; if ~ctx.final, K = p.NumNetsTuning; end
w = class_weights(y, p.Weighting);
mdl = ensemble(X, y, w, p, K);
mdl.platt = [];
if ctx.final
    % out-of-fold ensemble outputs -> Platt sigmoid (grouped folds: a deposit's buffer cells together)
    g = []; if isfield(ctx, 'groups'), g = ctx.groups; end
    kf = min(p.CalibFolds, min(sum(y), sum(~y)));
    if kf >= 2
        fold = mpm_kfold(numel(y), kf, RandStream.getGlobalStream, y, g);
        oof = nan(numel(y), 1);
        for f = 1:kf
            tr = fold ~= f; te = fold == f;
            if ~any(te) || all(y(tr)) || ~any(y(tr)), continue; end
            m = ensemble(X(tr, :), y(tr), class_weights(y(tr), p.Weighting), p, p.NumNetsTuning);
            oof(te) = mean(member_outputs(m, X(te, :)), 2);
        end
        ok = ~isnan(oof);
    else
        oof = mean(member_outputs(mdl, X), 2); ok = true(size(y));   % too few deposits: in-sample
    end
    [A, B] = mpm_platt_fit(oof(ok), y(ok), w(ok));
    mdl.platt = [A B]; mdl.calib_n = sum(ok); mdl.calib_oof = kf >= 2;
end
end

function mdl = ensemble(X, y, w, p, K)
t = 2 * double(y) - 1;                                  % +1 deposit, -1 background
o = struct('n_hidden', p.NumHidden, 'activation', p.Activation, 'C', p.C, ...
    'input_scale', p.InputScale, 'weights', w);
mdl.nets = cell(K, 1);
for k = 1:K
    mdl.nets{k} = mpm_elm_train(X, t, o);
end
mdl.form = mdl.nets{1}.form;
end

function F = member_outputs(mdl, X)
K = numel(mdl.nets);
F = zeros(size(X, 1), K);
for k = 1:K, F(:, k) = mpm_elm_predict(mdl.nets{k}, X); end
end

function [s, u] = predict_score(mdl, X)
F = member_outputs(mdl, double(X));
if isempty(mdl.platt)                                   % inside CV: raw output (ranking)
    s = mean(F, 2);
    if nargout > 1, u = std(F, 0, 2); end
    return;
end
sig = @(f) 1 ./ (1 + exp(mdl.platt(1) * f + mdl.platt(2)));
s = sig(mean(F, 2));
if nargout > 1, u = std(sig(F), 0, 2); end
end

function w = class_weights(y, scheme)
n1 = sum(y); n0 = sum(~y); w = ones(numel(y), 1);
switch upper(scheme)
    case 'NONE', return;
    case 'W1', w(y) = 1 / n1; w(~y) = 1 / n0;
    case 'W2'
        w(y) = 1 / n1; w(~y) = 1 / n0;
        if n0 > n1, w(~y) = 0.618 / n0; elseif n1 > n0, w(y) = 0.618 / n1; end
    otherwise, error('mpm:elm', 'Weighting must be W1, W2 or none.');
end
w = w / mean(w);
end

function txt = describe(mdl, p)
txt = sprintf('%d ELMs x %d %s hidden nodes, C = %.4g, InputScale = %.3g, weighting %s, %s solution', ...
    numel(mdl.nets), p.NumHidden, p.Activation, p.C, p.InputScale, p.Weighting, mdl.form);
if ~isempty(mdl.platt)
    src = 'out-of-fold'; if ~mdl.calib_oof, src = 'in-sample'; end
    txt = sprintf('%s; Platt A = %.3g, B = %.3g (%s, n = %d)', txt, mdl.platt(1), mdl.platt(2), src, mdl.calib_n);
end
end
