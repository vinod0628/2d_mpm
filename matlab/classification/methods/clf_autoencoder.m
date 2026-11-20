function M = clf_autoencoder()
%CLF_AUTOENCODER Deep autoencoder anomaly detection for prospectivity mapping (unsupervised).
%
%   Idea (Xiong & Zuo 2016): an autoencoder trained to reproduce the evidence vectors of the
%   whole study area learns the dominant (background) patterns; rare samples - e.g. the
%   multivariate signature of mineralisation - contribute little to training and are
%   reconstructed badly. The reconstruction error is the anomaly score.
%
%   Network  : symmetric fully connected autoencoder
%                P -> HiddenSizes -> Bottleneck (linear code layer) -> fliplr(HiddenSizes) -> P
%              Activation in the hidden layers, linear code and output layers (deep autoencoder
%              of Hinton & Salakhutdinov 2006). Loss: mean squared reconstruction error of the
%              z-scored evidence vector, L2 weight decay.
%   Training : Adam mini-batch back-propagation (common/model/mpm_autoencoder_*.m) from random
%              initialisation. Hinton & Salakhutdinov pre-trained deep stacks layer-wise with RBMs;
%              that is not needed for these shallow networks (a few layers, tens of units).
%              Own implementation instead of trainnet: ~50 x faster on such small networks.
%   Score    : mean squared reconstruction error  ||x - x_hat||^2 / P  per cell. Higher = more
%              anomalous = more prospective.
%   Fit      : on cfg.anomaly.n_fit random cells of the study area, without labels. Known deposits
%              only select the architecture (AUC of training deposits vs. training background),
%              the threshold (Youden index) and validate; test deposits stay independent.
%
%   Hyper-parameters (tuned: HiddenSizes x Bottleneck):
%     HiddenSizes   encoder layer widths before the code layer, e.g. 16 or [32 16]
%     Bottleneck    code size (must be < P, the number of features)
%     Activation    'tanh' | 'relu' | 'sigmoid'
%     L2            weight decay;  MaxEpochs, MiniBatchSize, LearnRate  (Adam)
%
%   References: Xiong, Y., Zuo, R., 2016. Recognition of geochemical anomalies using a deep
%   autoencoder network. Comput. Geosci. 86, 75-82.  Hinton, G.E., Salakhutdinov, R.R., 2006.
%   Reducing the dimensionality of data with neural networks. Science 313, 504-507.
%   Kingma, D.P., Ba, J., 2015. Adam: a method for stochastic optimization. ICLR.

M.key = 'autoencoder';
M.name = 'Autoencoder';
M.task = 'classification';
M.learning = 'unsupervised';
M.description = 'Unsupervised anomaly detection: deep autoencoder reconstruction error (Xiong & Zuo 2016).';
M.defaults = struct('HiddenSizes', 16, 'Bottleneck', 4, 'Activation', 'tanh', 'L2', 1e-4, ...
    'MaxEpochs', 50, 'MiniBatchSize', 256, 'LearnRate', 1e-3);
M.grid = struct('HiddenSizes', {{16, [32 16]}}, 'Bottleneck', [2 4 8]);
M.score_is_probability = false;
M.score_name = 'reconstruction error';
M.has_uncertainty = false;
M.train = @fit_model;
M.predict = @predict_score;
M.native_importance = [];
M.describe = @describe;
end

function mdl = fit_model(X, ~, p, ~)
X = double(X); P = size(X, 2);
if p.Bottleneck >= P
    error('mpm:autoencoder', 'Bottleneck (%d) must be smaller than the number of features (%d).', p.Bottleneck, P);
end
mdl = mpm_autoencoder_train(X, struct('hidden', p.HiddenSizes, 'code', p.Bottleneck, ...
    'activation', p.Activation, 'l2', p.L2, 'epochs', p.MaxEpochs, 'batch', p.MiniBatchSize, ...
    'lr', p.LearnRate));
mdl.fit_error = mean(mpm_autoencoder_predict(mdl, X));
mdl.n_fit = size(X, 1);
end

function s = predict_score(mdl, X)
s = mpm_autoencoder_predict(mdl, X);
end

function txt = describe(mdl, p)
txt = sprintf('autoencoder [%s] (%s, linear code), L2 = %.3g, %d epochs Adam; mean reconstruction MSE on %d fit cells = %.4f', ...
    num2str(mdl.sizes), p.Activation, p.L2, p.MaxEpochs, mdl.n_fit, mdl.fit_error);
end
