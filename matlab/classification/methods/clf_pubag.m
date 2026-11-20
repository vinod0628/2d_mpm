function M = clf_pubag()
%CLF_PUBAG Positive-unlabelled (PU) bagging for prospectivity mapping (Mordelet & Vert 2014).
%
%   Idea      : in MPM only the deposits are labelled; every other cell is UNLABELLED (barren or an
%               undiscovered deposit). Random "negatives" are therefore contaminated by unknown
%               positives. PU learning drops the negative class: the known deposits P are learned
%               against random subsamples of unlabelled cells U of the whole study area - no
%               distance buffer around deposits (Mordelet & Vert 2014; MPM use: Farahbakhsh et
%               al. 2023).
%   Algorithm : T bags; bag t trains a classifier of P against a random subsample U_t of K
%               unlabelled cells (K = SubsampleRatio x number of positives; the paper's safe choice
%               is K = N_P); the score is the average of the T classifiers (inductive bagging PU,
%               Alg. 1). Out-of-bag scores of the unlabelled training cells (Alg. 2) are kept in
%               the model. Engine: common/model/mpm_pubag_train.m / mpm_pubag_predict.m.
%   Base      : 'svm'  biased SVM with Gaussian kernel, C+ n+ = C- n- (the paper's classifier)
%               'tree' classification tree (CART), balanced prior
%   Pipeline  : M.learning = 'pu' -> the training "negatives" are cfg.pu.n_unlabelled random
%               cells of the study area (only training deposits and test background excluded);
%               CV tunes on the AUC of positives vs. unlabelled, which is a linear increasing
%               function of the true positive-vs-negative AUC, so it selects the same model.
%               Test deposits vs. test background validate as for every other method.
%   Score     : mean member output (SVM decision value / tree class-1 fraction) - ranks
%               P(deposit | x) but is not a probability -> Youden threshold.
%   Uncertainty map: standard deviation of the member outputs (bagging spread).
%
%   Hyper-parameters (CV grid: KernelScale x BoxConstraint x SubsampleRatio for 'svm'; for 'tree'
%   set e.g. cfg.tune.grid = struct('MinLeafSize', [1 3 5], 'SubsampleRatio', [1 3])):
%     BaseLearner     'svm' | 'tree'
%     NumBags         T of the final model (paper: little gain beyond ~100); NumBagsTuning in CV
%     SubsampleRatio  K / N_P
%     Replace         draw U_t with replacement (bootstrap)
%     BoxConstraint   SVM C;  KernelScale  MULTIPLIER of sqrt(P) (same convention as clf_svm)
%     MinLeafSize     tree
%
%   References: Mordelet, F., Vert, J.-P., 2014. A bagging SVM to learn from positive and
%   unlabeled examples. Pattern Recognit. Lett. 37, 201-209.  Elkan, C., Noto, K., 2008. Learning
%   classifiers from only positive and unlabeled data. KDD '08, 213-220.  Farahbakhsh, E.,
%   Maughan, J., Mueller, R.D., 2023. Prospectivity modelling of critical mineral deposits using a
%   generative adversarial network with oversampling and positive-unlabelled bagging. Ore Geol.
%   Rev., 105665 (doi 10.1016/j.oregeorev.2023.105665).

M.key = 'pubag';
M.name = 'PU bagging';
M.task = 'classification';
M.learning = 'pu';
M.description = 'Positive-unlabelled bagging (Mordelet & Vert 2014): deposits vs. random unlabelled cells, biased-SVM ensemble.';
M.defaults = struct('BaseLearner', 'svm', 'NumBags', 100, 'NumBagsTuning', 30, 'SubsampleRatio', 1, ...
    'Replace', true, 'BoxConstraint', 1, 'KernelScale', 1, 'MinLeafSize', 1);
M.grid = struct('KernelScale', [0.5 1 2 4 8], 'BoxConstraint', [0.1 1 10], 'SubsampleRatio', [1 3]);
M.score_is_probability = false;
M.score_name = 'PU score';
M.has_uncertainty = true;
M.train = @fit_model;
M.predict = @predict_score;
M.native_importance = [];
M.describe = @describe;
end

function mdl = fit_model(X, s, p, ctx)
s = logical(s);
T = p.NumBags; if ~ctx.final, T = p.NumBagsTuning; end
mdl = mpm_pubag_train(X, s, struct('n_bags', T, 'subsample', p.SubsampleRatio * sum(s), ...
    'replace', p.Replace, 'base', p.BaseLearner, 'box', p.BoxConstraint, ...
    'kernel_scale', p.KernelScale * sqrt(size(X, 2)), 'min_leaf', p.MinLeafSize, 'oob', ctx.final));
end

function [f, u] = predict_score(mdl, X)
if nargout > 1
    [f, u] = mpm_pubag_predict(mdl, X);
else
    f = mpm_pubag_predict(mdl, X);
end
end

function txt = describe(mdl, p)
switch mdl.base
    case 'svm'
        b = sprintf('biased SVM (Gaussian, C = %.3g, KernelScale = %.4g = x%.3g sqrt(P)), %.1f support vectors per bag, %d distinct', ...
            p.BoxConstraint, mdl.kernel_scale, p.KernelScale, mean(mdl.n_sv), size(mdl.Z, 1));
    case 'tree'
        b = sprintf('CART (MinLeafSize %d), %.1f leaves per bag', p.MinLeafSize, mean(mdl.n_leaves));
end
r = 'with'; if ~mdl.replace, r = 'without'; end
txt = sprintf('%d bags: %d positives vs. K = %d of %d unlabelled cells (%s replacement); %s', ...
    mdl.n_bags, mdl.n_pos, mdl.K, mdl.n_unl, r, b);
if isfield(mdl, 'oob_count')
    txt = sprintf('%s; out-of-bag scores for %d of %d unlabelled cells', txt, sum(mdl.oob_count > 0), mdl.n_unl);
end
end
